# NILMFormer untrained ONNX (Person 1, AI): Week 1 deliverable

Model: NILMFormer (Petralia et al., KDD 2025), official code commit `e73a975a42fbebed6f9d7e90d75f7f48ae02fed9`.
The weights are UNTRAINED (seed 0). Every reference output in this folder is for checking the deployment chain only
(ONNX vs PyTorch, board vs PC). They say nothing about accuracy.

## 1. Files (folder `week1_out_4d_pure7`)

| File | What it is |
|---|---|
| `nilmformer_untrained_w128_4d_conv2d_v7.onnx` | float32 model, opset 13, static shapes (about 1.7 MB) |
| `nilmformer_untrained_w128_4d_conv2d_v7_int8_qdq.onnx` | INT8 (QDQ) version (about 0.7 MB) |
| `test_input_4d.npy` | random test input, shape (1, 9, 1, 128), float32 |
| `test_output_4d.npy` | float reference output for that input, shape (1, 1, 1, 128) |
| `nilmformer_untrained_w128_4d_conv2d_v7_int8_qdq_out.npy` | INT8 model output for the same input |
| `week1_export_4d_pure7.py` | export script (PyTorch to float ONNX) |
| `quant_int8_v2.py` | quantization script (float ONNX to INT8 QDQ) |

Why "4d" in the names: the plan names `nilmformer_untrained_w128.onnx` with a 3D input (1, 9, 128). The ST tool needed a 4D
input (1, 9, 1, 128), so the file names carry "_4d". The 3D PyTorch model and the 4D ONNX give the same numbers
(max difference about 1e-8). The folder `experiments/` holds the diagnosis scripts (window 64/96, chunked feed-forward);
they are not part of the working path.

## 2. Model settings (src/nilmformer/congif.py)

| Setting | Value |
|---|---|
| Input channels | 1 power + 8 timestamp channels = 9 |
| Window length | 128 (the repo default in configs/expes.yaml is 256; we use 128 as in the plan) |
| Embedding | 4 dilated conv blocks (dilations 1, 2, 4, 8, kernel 3), 72 filters, plus a 1x1 conv for the timestamp channels (24 channels) |
| d_model | 96 (72 + 24) |
| Transformer layers | 3 (pre-norm, diagonally masked self-attention, 8 heads, head dim 12) |
| Feed-forward | 96 to 384 to 96, GELU, ratio 4 |
| Dropout | 0.2 (inactive at inference) |
| Head | 1D conv, kernel 3, output 1 channel |

## 3. Input and output

**Input** `input`, float32, shape (1, 9, 1, 128): batch, channels, height 1, window length.

- Channel 0: aggregate (whole-house) power. Clip at 6000 W, then divide by 6000. One shared constant (plan Week 2).
  The repo default scaler (`MaxScaling`) divides by the training-set maximum instead; do not use it for deployment.
  In `src/helpers/dataset.py` the scaler accepts an integer (stat1 = 0, stat2 = that integer), so
  `power_scaling_type = 6000` gives exactly this. `appliance_scaling_type = SameAsPower` applies the same divisor
  to the appliance target.
- Channels 1 to 8: timestamp features, computed PER SAMPLE (each minute of the window has its own values).
  Timestamps = `pd.date_range(start = window start, periods = 128, freq = "1min")`. Values lie in [-1, 1].

  | Channel | Value | Formula (src/helpers/dataset.py, NILMDataset._create_exogene, cosinbase = True) |
  |---|---|---|
  | 1, 2 | sin, cos of minute | 2*pi*minute/60 |
  | 3, 4 | sin, cos of hour | 2*pi*hour/24 |
  | 5, 6 | sin, cos of day of week | 2*pi*(dayofweek - 1)/7, pandas dayofweek, Monday = 0 (note the -1) |
  | 7, 8 | sin, cos of month | 2*pi*(month - 1)/12, month 1..12 (note the -1) |

  Variable order comes from `list_exo_variables` in configs/expes.yaml: minute, hour, dow, month.
  `preprocessing.create_exogene` uses the same formulas WITHOUT the -1 for dow and month. It is only used in
  `metrics.py` (full-window energy evaluation), so the official repo is inconsistent. We follow `dataset.py` because
  training uses it. The board must match it. The `cosinbase=False` calls in expes.py belong to DiffNILM only.

**Output** `output`, float32, shape (1, 1, 1, 128): predicted power of ONE appliance per sample, in scaled units.
Multiply by 6000 to get watts. One model per appliance.

## 4. Inside the model vs in C on the board

Inside the ONNX (nothing to do on the board):

- instance normalization of channel 0 (subtract the window mean, divide by the window std, eps 1e-6, biased variance)
- embedding, mean/std token, Transformer encoder, convolutional head
- reverse normalization of the output (multiplied and shifted by the predicted std and mean)

In C on the board (Person 2):

- before inference: clip to 6000 W, divide by 6000
- build the 8 timestamp channels per sample (section 3), same order and formulas
- keep a window buffer of 128 samples
- after inference: multiply the output by 6000

## 5. Export notes (why the graph differs from the PyTorch code)

- Conv1d replaced by Conv2d (1, k). No Squeeze or Einsum. Static input and output shapes, no dynamic axes.
- Attention is written per head (8 separate 129 x 129 score matrices, scale folded into Wq, output projection as a
  sum of per-head slices of Wo). Same math as the original (max difference about 1e-7).
- The heads are chained with a zero dependency (head h reads x + 0 * output of head h-1) so the ST tool computes
  one head at a time. Without it activations were 726 KiB; with it 522 KiB.

## 6. INT8 notes

- ONNX QDQ, per-channel INT8 weights, INT8 activations, calibration with 64 RANDOM windows. INT8 accuracy is unproven
  until real calibration data is used (plan Week 5).
- 72 matmuls stay in float: the 48 attention matmuls (QK^T and attention x V, 8 heads x 3 layers x 2) and the 24
  per-head output projections. Reasons: the ST tool cannot run `Where` on INT8 input, and a float matmul feeding
  directly into a quantized layer gave 24 "number of I/O tensor is not coherent" errors.
- Max INT8 vs float difference on the random test input: about 0.002 (random data, not an accuracy result).

## 7. Status of the ST Edge AI tool and the board (X-CUBE-AI 10.2.0, ST Edge AI Core 2.2.0)

| Item | Result |
|---|---|
| v7 INT8, window 128 | analyzes: weights 649,028 B (633.82 KiB), activations 534,512 B (521.98 KiB) |
| Reported MACC (about 1.3 G) | counting artifact: the tool counts a runtime matmul as output size x (input size + 1). One QK^T matmul: reported 25,776,909, true 199,692, measured 3.794 ms (1,517,645 cycles at 400 MHz). Real compute is about 60 MMAC |
| Link | `.bss will not fit in region RAM_D1, overflowed by 194,568 B` (activations plus about 168 KB runtime data in the 512 KB AXI SRAM). Needs SRAM split or external SDRAM |
| Board | window 128 compiled and booted with SDRAM and MPU/cache configured; execution read timeout (> 10 s), cause not confirmed |
| Window 64 and 96 | crash the tool ("INTERNAL ERROR: edge ... not in graph"), float and INT8; open |
| Chunked feed-forward (experiments/) | crashes the tool even in float; do not use |
| Flash | one INT8 model is 634 KiB; three models are about 1.9 MB, so QSPI flash is needed (the plan allows it) |

## 8. How to reproduce (Colab, /content/NILMFormer)

```
uv run --with onnx --with onnxruntime --with onnxsim python week1_export_4d_pure7.py
uv run --with onnx --with onnxruntime python quant_int8_v2.py <float.onnx> <int8_qdq.onnx> 72
```

Checks to read in the output: max differences about 1e-7 (export), `excluded ... 72 | expected: 72` and
`Where fed by a DequantizeLinear: 0` (quantization). To use trained weights (Week 3 onwards), load the checkpoint into
the model before the per-head patch in the export script; the patch does not change the weights.

## 9. Known limitations

- Untrained weights; random calibration; no accuracy claim.
- Window 128 only (64 and 96 crash the tool). A different window means retraining and separate reporting.
- Full-model timing on the board is not measured yet (only one QK^T matmul).
