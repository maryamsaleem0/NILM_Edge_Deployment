# Week 1 – Feasibility test

Model tested: nilmformer_untrained_w128_4d_conv2d_v4.onnx (untrained weights)
Board: STM32H747I-DISCO
Tools: ST Edge AI Core 2.2.0, X-CUBE-AI 10.2.0, STM32CubeIDE 1.x

Files:
- feasibility_report.pdf : 1-page summary and decision
- nilm_analyze_report.txt / analyze_log.txt : Analyze results (Flash, RAM, MACs)
- nilm_validate_report.txt / validation_log.txt : desktop validation (cos 0.99999)
- nilm_val_io.npz : input and output samples from the validation

Known issue: output positions 0 and 127 differ (replicate padding in head). Fix requested (v5).