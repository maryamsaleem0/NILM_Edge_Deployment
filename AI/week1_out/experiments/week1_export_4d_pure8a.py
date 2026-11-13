import os, numpy as np, torch, torch.nn as nn, onnx, onnxruntime as ort
from collections import Counter
from onnx import shape_inference
from onnxsim import simplify

# --- patches so the model can be exported (same numbers) ---
_orig_diag = torch.diag
def _diag_patched(x, diagonal=0):
    if x.dim() == 1 and diagonal == 0:
        _i = torch.arange(x.shape[0], device=x.device)
        return (_i[:, None] == _i[None, :]).to(x.dtype)
    return _orig_diag(x, diagonal)
torch.diag = _diag_patched

_orig_einsum = torch.einsum
def _einsum_patched(eq, *ops):
    if len(ops) == 1 and isinstance(ops[0], (list, tuple)):
        ops = tuple(ops[0])
    if eq == "blhe,bshe->bhls":
        q, k = ops
        return torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1))
    if eq == "bhls,bshd->blhd":
        a, v = ops
        return torch.matmul(a, v.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
    return _orig_einsum(eq, *ops)
torch.einsum = _einsum_patched

from src.nilmformer.congif import NILMFormerConfig
from src.nilmformer.model import NILMFormer

BASE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(BASE, "week1_out") + os.sep
OUT = os.path.join(BASE, "week1_out_4d_pure8a") + os.sep
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0); np.random.seed(0)
model = NILMFormer(NILMFormerConfig()); model.eval()
L, B = 128, 1

# input and 3D reference (before any conversion)
if os.path.exists(IN + "test_input.npy"):
    x3 = torch.from_numpy(np.load(IN + "test_input.npy")); old = np.load(IN + "test_output.npy")
else:
    x3 = torch.rand(B, 9, L); x3[:, :1] *= 0.5; x3[:, 1:] = x3[:, 1:] * 2 - 1; old = None
with torch.no_grad():
    ref = model(x3).numpy()

# --- per-head attention: same numbers, one head's score matrix at a time ---
import torch.nn.functional as F
from src.nilmformer.layers.transformer import DiagonnalyMaskedSelfAttention

def _attn_per_head(self, x):
    B, T, _ = x.shape
    H, hd = self.n_heads, self.head_dim
    scale = 1.0 / hd ** 0.5
    i = torch.arange(T, device=x.device)
    diag = i[:, None] == i[None, :]                       # True on the diagonal
    out = None
    x0 = x
    for h in range(H):
        sl = slice(h * hd, (h + 1) * hd)
        x = x0 if out is None else x0 + (out - out)     # head h waits for head h-1 (RAM scheduling)
        q = F.linear(x, self.wq.weight[sl] * scale)       # (B, T, hd), softmax scale folded into Wq
        k = F.linear(x, self.wk.weight[sl])
        v = F.linear(x, self.wv.weight[sl])
        s = torch.matmul(q, k.transpose(1, 2))            # (B, T, T) for ONE head
        s = s.masked_fill(diag, -1e30)
        a = torch.softmax(s, dim=-1)
        y = F.linear(torch.matmul(a, v), self.wo.weight[:, sl])   # this head's share of Wo
        out = y if out is None else out + y
    return out
DiagonnalyMaskedSelfAttention.forward = _attn_per_head

# --- chunked feed-forward: same numbers, one chunk of the hidden layer at a time ---
from src.nilmformer.layers.transformer import PositionWiseFeedForward
N_CHUNKS = 4

def _ffn_chunked(self, x):
    Hd = self.layer1.weight.shape[0]
    assert Hd % N_CHUNKS == 0
    cs = Hd // N_CHUNKS
    x0 = x
    out = None
    for c in range(N_CHUNKS):
        sl = slice(c * cs, (c + 1) * cs)
        xc = x0          # chunk c waits for chunk c-1
        b1 = None if self.layer1.bias is None else self.layer1.bias[sl]
        h = self.activation(F.linear(xc, self.layer1.weight[sl], b1))      # (B, T, cs)
        y = F.linear(h, self.layer2.weight[:, sl])                         # this chunk's share of layer2
        out = y if out is None else out + y
    if self.layer2.bias is not None:
        out = out + self.layer2.bias
    return out
PositionWiseFeedForward.forward = _ffn_chunked

# --- in-place conversion: Conv1d -> Conv2d (1,k), BatchNorm1d -> BatchNorm2d ---
def to_conv2d(c):
    k, d = c.kernel_size[0], c.dilation[0]
    if c.padding == "same":
        assert (d * (k - 1)) % 2 == 0
        p = d * (k - 1) // 2
    else:
        p = c.padding[0]
    n = nn.Conv2d(c.in_channels, c.out_channels, (1, k), stride=(1, c.stride[0]),
                  padding=(0, p), dilation=(1, d), groups=c.groups,
                  bias=c.bias is not None, padding_mode=c.padding_mode)
    with torch.no_grad():
        n.weight.copy_(c.weight.unsqueeze(2))
        if c.bias is not None: n.bias.copy_(c.bias)
    return n

def to_bn2d(b):
    n = nn.BatchNorm2d(b.num_features, eps=b.eps, momentum=b.momentum,
                       affine=b.affine, track_running_stats=b.track_running_stats)
    n.load_state_dict(b.state_dict())
    return n

def convert(mod):
    for name, ch in list(mod.named_children()):
        if isinstance(ch, nn.Conv1d): setattr(mod, name, to_conv2d(ch))
        elif isinstance(ch, nn.BatchNorm1d): setattr(mod, name, to_bn2d(ch))
        else: convert(ch)
convert(model)
assert not any(isinstance(m, (nn.Conv1d, nn.BatchNorm1d)) for m in model.modules())

# --- new forward: (B, 9, 1, L) stays 4D through the conv parts ---
class Model4D(nn.Module):
    def __init__(self, m, B, L):
        super().__init__()
        self.m, self.B, self.L, self.D = m, B, L, m.NFConfig.d_model
        self.pick = nn.Conv2d(L + 1, L + 2, kernel_size=1, bias=False)
        with torch.no_grad():
            self.pick.weight.zero_()
            self.pick.weight[0, 0, 0, 0] = 1.0
            self.pick.weight[L + 1, L - 1, 0, 0] = 1.0
            for i in range(L):
                self.pick.weight[i + 1, i, 0, 0] = 1.0
        self.pick.weight.requires_grad_(False)
        h = m.DownstreamTaskHead
        self.head = nn.Conv2d(h.in_channels, h.out_channels, h.kernel_size, stride=h.stride,
                              dilation=h.dilation, bias=h.bias is not None)
        self.head.load_state_dict(h.state_dict())
    def forward(self, x):
        m, B, L, D = self.m, self.B, self.L, self.D
        enc = x[:, 1:]                                            # (B, 8, 1, L)
        p = x[:, :1]                                              # (B, 1, 1, L)
        mean = p.mean(dim=-1, keepdim=True).detach()              # (B, 1, 1, 1)
        std = torch.sqrt(torch.var(p, dim=-1, keepdim=True, unbiased=False) + 1e-6).detach()
        p = m.EmbedBlock((p - mean) / std)                        # (B, 72, 1, L)
        enc = m.ProjEmbedding(enc)                                # (B, 24, 1, L)
        t = torch.cat([p, enc], dim=1)                            # (B, 96, 1, L)
        t = t.permute(0, 2, 3, 1).reshape(B, L, D)                # -> 3D for the transformer
        tok = m.ProjStats1(torch.cat([mean, std], dim=1).reshape(B, 1, 2))   # (B, 1, 96)
        t = m.EncoderBlock(torch.cat([t, tok], dim=1))
        t = t.reshape(B, L + 1, 1, D)
        t = self.pick(t)            # (B, L, 96)
        t = t.permute(0, 3, 2, 1)                # -> back to 4D
        t = self.head(t)                               # (B, 1, 1, L)
        so = m.ProjStats2(tok).reshape(B, 2, 1, 1)                # [mean_out, std_out]
        return t * so[:, 1:2] + so[:, 0:1]

wrap = Model4D(model, B, L).eval()
x4 = x3.unsqueeze(2)                                              # (1, 9, 1, 128)
with torch.no_grad():
    y4 = wrap(x4).numpy()
print("PyTorch 3D vs 4D max diff:", float(np.abs(y4[:, :, 0, :] - ref).max()))

P = OUT + "nilmformer_untrained_w128_4d_conv2d_v8a.onnx"
kw = dict(input_names=["input"], output_names=["output"], opset_version=13, do_constant_folding=True)
try:
    torch.onnx.export(wrap, (x4,), P, dynamo=False, **kw)
except TypeError:
    torch.onnx.export(wrap, (x4,), P, **kw)

m = onnx.load(P)
for d, v in zip(m.graph.output[0].type.tensor_type.shape.dim, [1, 1, 1, 128]):
    d.dim_value = v
del m.graph.value_info[:]
m = shape_inference.infer_shapes(m)
ms, ok = simplify(m)
onnx.checker.check_model(ms)
onnx.save(ms, P)

bad = [d for vi in list(ms.graph.value_info) + list(ms.graph.input) + list(ms.graph.output)
       for d in vi.type.tensor_type.shape.dim if d.dim_param]
print("simplify ok:", ok, "| remaining symbolic dims:", len(bad))
for t in list(ms.graph.input) + list(ms.graph.output):
    print(t.name, [d.dim_value for d in t.type.tensor_type.shape.dim])
c = Counter(n.op_type for n in ms.graph.node)
print("Squeeze/Unsqueeze/Reshape/Transpose:", {k: c[k] for k in ["Squeeze", "Unsqueeze", "Reshape", "Transpose"]})
print("Einsum left in file:", "Einsum" in c, "| nodes:", len(ms.graph.node))
print("ops:", sorted(c))
inits = {i.name: len(i.dims) for i in ms.graph.initializer}
print("Conv weight ranks (4 = Conv2d):", sorted({inits.get(n.input[1]) for n in ms.graph.node if n.op_type == "Conv"}, key=str))

o = ort.InferenceSession(P).run(None, {"input": x4.numpy()})[0]
print("ONNX shape:", o.shape, "| allclose vs PyTorch 3D:", np.allclose(o[:, :, 0, :], ref, atol=1e-4),
      "| max diff:", float(np.abs(o[:, :, 0, :] - ref).max()))
if old is not None:
    print("same weights as the earlier 3D file:", np.allclose(ref, old, atol=1e-6))
np.save(OUT + "test_input_4d.npy", x4.numpy()); np.save(OUT + "test_output_4d.npy", o)
print("ONNX size (KB):", round(os.path.getsize(P) / 1024, 1), "| files:", os.listdir(OUT))
