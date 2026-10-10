# Benchmark Notes & Hardware Information

**Test Machine Specifications**:
- **Laptop Model/CPU**: 13th Gen Intel(R) Core(TM) i5-13420H (12 vCPUs, up to 4.60 GHz)
- **Discrete GPU**: NVIDIA GeForce RTX 2050 Laptop GPU (4 GB GDDR6 VRAM, Ampere)
- **Integrated GPU**: Intel(R) UHD Graphics (Gen-12LP, Shared System RAM)
- **RAM**: 8 GB
- **OS**: Windows (x64)

---

Date: 10/08/2026
Setup: Transformers.js WebGPU / fp32 / nvidia-chrome
Hardware: NVIDIA RTX 2050 (4 GB VRAM)
Outcome: Model failed to initialize (ERROR_CODE: 6, ERROR_MESSAGE: std::bad_alloc).

Date: 10/09/2026
Setup: Transformers.js WebGPU / fp32 / intel-chrome
Hardware: Intel UHD Graphics (Gen-12LP, Shared System RAM)
Outcome: Model failed to initialize (ERROR_CODE: 6, ERROR_MESSAGE: std::bad_alloc).