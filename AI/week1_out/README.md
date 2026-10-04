# NILMFormer untrained ONNX (Week 1) - for the feasibility test

**Weights are RANDOM (untrained).** Use this file only to check layer support, Flash, RAM and latency.

## Files
- nilmformer_untrained_w128.onnx : model (FP32, about 1.6 MB, opset 13)
- test_input.npy  : one random input, shape (1, 9, 128), float32
- test_output.npy : PyTorch output for that input, shape (1, 1, 128)
- week1_export.py : script that made these files

## Model settings (paper defaults)
3 Transformer layers, d_model 96, 8 heads, PFFN ratio 4, dropout 0.2 (off at inference),
4 dilated conv ResUnits (dilations 1, 2, 4, 8), 72 conv filters, window length 128.
Parameters: 383,283.

## Input (ONE input tensor, name "input")
Shape (1, 9, 128) = (batch, channels, time)
- channel 0      : aggregate power window, watts divided by 6000 (range 0..1)
- channels 1 to 8: 8 time-feature channels (exogenous encoding, same length 128)

## Output (name "output")
Shape (1, 1, 128): predicted appliance power per minute. Multiply by 6000 to get watts.

## What is INSIDE the model
- Mean/std normalisation of the power channel and the reverse step at the output
  (so the board does NOT need to do this).

## What the board must do in C
- Before the model: power / 6000, and build the 8 time-feature channels
  (exact definition will be confirmed in Week 2 from the data pipeline).
- After the model: output * 6000 to get watts.

## Operators to check in X-CUBE-AI
LayerNorm, GELU, Softmax, Einsum (attention), Conv1d with dilation 1, 2, 4, 8,
BatchNorm1d (in eval mode), Slice, Concat, Transpose.
Attention memory grows with window^2 (129 x 129 per head, 8 heads).

## Export changes (numerically identical, only for ONNX export)
1. Diagonal attention mask built with index comparison instead of torch.diag.
2. Conv1d padding="same" replaced by explicit padding (needed with dilation).

## Check
onnxruntime vs PyTorch: max abs difference 2.6e-08 (allclose True).
