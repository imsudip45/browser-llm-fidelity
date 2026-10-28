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

---

### Diagnostic Notes on Runtime Behavior

1. **Transformers.js WebGPU (`fp16` & `q4f16`) Generation Collapse:**
   Across both Intel UHD Graphics and NVIDIA GeForce RTX 2050, executing `onnx-community/Qwen2.5-0.5B-Instruct` via Transformers.js 4.3.0 WebGPU (`fp16` and `q4f16`) resulted in severe generation degradation (`0.00–0.01` accuracy on GSM8K, `0.02–0.07` on MMLU, diverging immediately at token `0`). Repeating the NVIDIA `fp16` run (`nvidia-chrome-repeat`) produced identical results (`0.01` GSM8K, `0.07` MMLU), confirming the failure is deterministic rather than transient.

2. **Browser CPU WASM Sanity Check (`browser/tjs-wasm/q4/cpu-chrome-check`, $N=20$):**
   To isolate whether the failure originated from the ONNX model weights/harness or specifically from the WebGPU execution backend, we executed a 20-item diagnostic subset (10 GSM8K + 10 MMLU) using the same `site/harness.js` runner on the browser WebAssembly (WASM) CPU backend (`q4`). On GSM8K, browser WASM achieved `0.40` accuracy—matching the native `fp32` CPU reference accuracy (`0.40`) on the same 10 problems—confirming that the prompt harness and ONNX graph are valid and isolating the collapse to the Transformers.js WebGPU execution path.