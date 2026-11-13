import os, sys, numpy as np, onnx, onnxruntime as ort
from collections import Counter
from onnxruntime.quantization import (quantize_static, CalibrationDataReader,
                                      QuantFormat, QuantType, CalibrationMethod)

FP32, INT8 = sys.argv[1], sys.argv[2]
EXPECTED = int(sys.argv[3]) if len(sys.argv) > 3 else None     # attention MatMuls we expect
D = os.path.dirname(FP32) + os.sep

# 1) give every node a unique name, then find MatMuls with NO constant input (activation x activation)
m = onnx.load(FP32)
for i, n in enumerate(m.graph.node):
    if not n.name: n.name = f"{n.op_type}_auto_{i}"
tmp = D + "_named_tmp.onnx"; onnx.save(m, tmp)
const = {i.name for i in m.graph.initializer} | {o for n in m.graph.node if n.op_type == "Constant" for o in n.output}
exclude = [n.name for n in m.graph.node if n.op_type == "MatMul" and not any(x in const for x in n.input)]
cons = {}
for n in m.graph.node:
    for x in n.input: cons.setdefault(x, []).append(n)
byname = {n.name: n for n in m.graph.node}
extra = {c.name for nm in exclude for o in byname[nm].output for c in cons.get(o, []) if c.op_type == "MatMul"}
exclude = sorted(set(exclude) | extra)
print("excluded: attention matmuls + output projections:", len(exclude), "| expected:", EXPECTED)
if EXPECTED is not None and len(exclude) != EXPECTED: print("WARNING: count differs from expected - check before sending")

class Calib(CalibrationDataReader):                      # random windows: fine for size/speed, NOT for accuracy
    def __init__(self, n=64, seed=1):
        rng = np.random.default_rng(seed); self.data = []
        for _ in range(n):
            x = np.empty((1, 9, 1, 128), np.float32)
            x[:, 0] = rng.random((1, 1, 128)) * rng.choice([0.1, 0.3, 0.6])
            x[:, 1:] = rng.uniform(-1, 1, (1, 8, 1, 128))
            self.data.append({"input": x})
        self.it = iter(self.data)
    def get_next(self): return next(self.it, None)

quantize_static(tmp, INT8, Calib(), quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, per_channel=True,
                op_types_to_quantize=["Conv", "MatMul", "Gemm"], nodes_to_exclude=exclude,
                calibrate_method=CalibrationMethod.MinMax)
os.remove(tmp)

q = onnx.load(INT8); onnx.checker.check_model(q)
prod = {o: n for n in q.graph.node for o in n.output}
bad = sum(1 for n in q.graph.node if n.op_type == "Where" for x in n.input
          if x in prod and prod[x].op_type == "DequantizeLinear")
c = Counter(n.op_type for n in q.graph.node)
print("sizes KB  fp32 / int8:", round(os.path.getsize(FP32) / 1024), "/", round(os.path.getsize(INT8) / 1024))
print("Q / DQ nodes:", c["QuantizeLinear"], "/", c["DequantizeLinear"], "| Where fed by a DequantizeLinear:", bad, "(must be 0)")
x = np.load(D + "test_input_4d.npy")
a = ort.InferenceSession(FP32).run(None, {"input": x})[0]
b = ort.InferenceSession(INT8).run(None, {"input": x})[0]
print("int8 vs fp32 | shape:", b.shape, "| max abs diff:", float(np.abs(a - b).max()))
np.save(os.path.splitext(INT8)[0] + "_out.npy", b)
