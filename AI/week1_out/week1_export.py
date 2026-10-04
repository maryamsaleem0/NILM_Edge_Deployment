import os, shutil
import numpy as np
import torch
import onnxruntime as ort
from src.nilmformer.congif import NILMFormerConfig
from src.nilmformer.model import NILMFormer


# --- patch: ONNX cannot export torch.diag, so build the same diagonal mask with eye ---
_orig_diag = torch.diag
def _diag_patched(x, diagonal=0):
    if x.dim() == 1 and diagonal == 0:
        _i = torch.arange(x.shape[0], device=x.device)
        return (_i[:, None] == _i[None, :]).to(x.dtype)
    return _orig_diag(x, diagonal)
torch.diag = _diag_patched
# ------------------------------------------------------------------------------------

torch.manual_seed(0)
np.random.seed(0)

OUT = "/content/NILMFormer/week1_out"
os.makedirs(OUT, exist_ok=True)
ONNX_PATH = f"{OUT}/nilmformer_untrained_w128.onnx"

# 1) Build the model (defaults = paper settings), random weights
cfg = NILMFormerConfig()
model = NILMFormer(cfg)
# patch: padding="same" + dilation is not supported by onnxruntime, use explicit padding
for m in model.modules():
    if isinstance(m, torch.nn.Conv1d) and m.padding == "same":
        k = m.kernel_size[0]; d = m.dilation[0]
        assert (d * (k - 1)) % 2 == 0
        m.padding = (d * (k - 1) // 2,)
model.eval()   # IMPORTANT: turns off dropout

n_params = sum(p.numel() for p in model.parameters())
print("Parameters:", n_params, "(expected about 385,000)")

# 2) Make one random test input: (1, 1 + 8, 128)
L = 128
x = torch.rand(1, 1 + cfg.c_embedding, L)
x[:, :1, :] = x[:, :1, :] * 0.5      # power channel, scaled range 0..0.5
x[:, 1:, :] = x[:, 1:, :] * 2 - 1    # time features, range -1..1

with torch.no_grad():
    y = model(x)
print("Input shape :", tuple(x.shape))
print("Output shape:", tuple(y.shape))

# 3) Export to ONNX (try a few opset versions until one works)
def export(opset):
    kwargs = dict(input_names=["input"], output_names=["output"],
                  opset_version=opset, do_constant_folding=True)
    try:
        torch.onnx.export(model, (x,), ONNX_PATH, dynamo=False, **kwargs)
    except TypeError:   # older torch has no 'dynamo' argument
        torch.onnx.export(model, (x,), ONNX_PATH, **kwargs)

used_opset = None
for opset in [13, 14, 17]:
    try:
        export(opset)
        used_opset = opset
        print("Export OK with opset", opset)
        break
    except Exception as e:
        print(f"Export failed with opset {opset}:", repr(e)[:300])
if used_opset is None:
    raise SystemExit("ONNX export failed for all opsets - send me the error.")

# 4) Check ONNX output against PyTorch output
sess = ort.InferenceSession(ONNX_PATH)
onnx_out = sess.run(None, {"input": x.numpy()})[0]
torch_out = y.numpy()
print("allclose:", np.allclose(onnx_out, torch_out, atol=1e-4))
print("max abs difference:", float(np.abs(onnx_out - torch_out).max()))

# 5) Save test input and output
np.save(f"{OUT}/test_input.npy", x.numpy())
np.save(f"{OUT}/test_output.npy", torch_out)
print("ONNX size (KB):", round(os.path.getsize(ONNX_PATH) / 1024, 1))

# 6) Copy to Google Drive if it is mounted
drive_dir = "/content/drive/MyDrive/NILM_FYP/week1"
if os.path.isdir("/content/drive/MyDrive"):
    os.makedirs(drive_dir, exist_ok=True)
    for f in os.listdir(OUT):
        shutil.copy(f"{OUT}/{f}", drive_dir)
    print("Copied to Drive:", drive_dir)
print("Files:", os.listdir(OUT))
