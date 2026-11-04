# NILM Edge Deployment (Group 05)

**Current Status:** Hardware Integration & Benchmarking

Deploying the NILMFormer Transformer model onto an STM32H747I-DISCO board as an edge AI device to benchmark edge-versus-cloud performance metrics.

### Technical Milestones Achieved:
* **Model Quantization:** Successfully imported `v7_int8_qdq` ONNX model using X-CUBE-AI 10.2.0.
* **Memory Management:** Overcame internal 512 KiB `RAM_D1` limit by configuring the FMC driver to initialize 32 MB of external SDRAM. 
* **Linker Optimization:** Modified the GCC linker script (`STM32H747XIHX_FLASH.ld`) to map the 534.5 KiB activation pool to SDRAM (`0xD0000000`). 
* **Hardware Validation:** Target validation firmware successfully compiled, flashed, and booted on the Cortex-M7 at 400 MHz.
