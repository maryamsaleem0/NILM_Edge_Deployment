# Embedded Module - Week 4

## Hardware Platform Finalization
The target hardware platform has been officially finalized as the **STM32H747I-DISCO** microcontroller. The base `.ioc` configuration file has been generated and the edge deployment environment is now initialized using **STM32Cube.AI** for the Cortex-M7 core.

## Edge Deployment Challenges Investigated
During this phase, we analyzed the specific system-level challenges of deploying traditional NILM methods on resource-constrained devices:
* **Overlapping Appliance Operations:** Investigated the complexity of accurate load decomposition and instantaneous power calculations when multiple electrical appliances operate simultaneously.
* **MCU Memory Limits:** Evaluated strategies for designing a robust feature extraction algorithm that successfully separates overlapping appliance signatures while remaining strictly within the STM32 SRAM limits.
