# Benchmark Notes & Hardware Information

**Test Machine Specifications**:
- **Laptop Model/CPU**: 13th Gen Intel(R) Core(TM) i5-13420H (12 vCPUs, up to 4.60 GHz)
- **Discrete GPU**: NVIDIA GeForce RTX 2050 Laptop GPU (4 GB GDDR6 VRAM, Ampere)
- **Integrated GPU**: Intel(R) UHD Graphics (Gen-12LP, Shared System RAM)
- **RAM**: 8 GB
- **OS**: Windows (x64)

---

### Evaluation Inventory & Archival Scope

The benchmark suite consists of **14 run result files** (28 task evaluations, totaling **3,920 individual item evaluations**):
- **13 Full Runs ($N=300$ items each, 200 MMLU + 100 GSM8K = 3,900 item evaluations):**
  - 4 Native Node.js CPU runs: `fp32`, `fp16`, `q4`, `q4f16`
  - 5 Transformers.js WebGPU runs: `fp16` on Intel UHD, `fp16` on NVIDIA RTX 2050, `fp16` repeat on NVIDIA RTX 2050, `q4f16` on Intel UHD, `q4f16` on NVIDIA RTX 2050
  - 4 WebLLM WebGPU runs: `q4f16_1` on Intel UHD, `q4f16_1` on NVIDIA RTX 2050, `q4f32_1` on Intel UHD, `q4f32_1` on NVIDIA RTX 2050
- **1 Browser CPU Diagnostic Run ($N=20$ items, 10 MMLU + 10 GSM8K = 20 item evaluations):**
  - `browser/tjs-wasm/q4/cpu-chrome-check` on browser WebAssembly CPU backend.

*Zenodo Archival Note:* The Zenodo release DOI [`10.5281/zenodo.23760946`](https://doi.org/10.5281/zenodo.23760946) permanently archives the raw `v1.0.0` benchmark JSON execution outputs, dataset questions, and the original execution harness. The updated multi-table audit pipeline and static reconciliation scripts (`v1.1.x`) are version-controlled directly in this repository commit and subsequent repository releases.

*Weight Identifier & Hash Audit Note:* Across comparisons, model repository identifiers and precision dtypes were strictly matched (`onnx-community/Qwen2.5-0.5B-Instruct` for Transformers.js; `Qwen2.5-0.5B-Instruct-q4f16_1-MLC` / `q4f32_1-MLC` for WebLLM). However, SHA-256 file hashes were not pinned across the Oct 8–9 and Oct 28 run dates.

---

### Runtime Initialization & Memory Failures

- **Date:** 10/08/2026
  - **Setup:** Transformers.js WebGPU / fp32 / nvidia-chrome
  - **Hardware:** NVIDIA RTX 2050 (4 GB VRAM)
  - **Outcome:** Model failed to initialize (`ERROR_CODE: 6, ERROR_MESSAGE: std::bad_alloc`).
- **Date:** 10/09/2026
  - **Setup:** Transformers.js WebGPU / fp32 / intel-chrome
  - **Hardware:** Intel UHD Graphics (Gen-12LP, Shared System RAM)
  - **Outcome:** Model failed to initialize (`ERROR_CODE: 6, ERROR_MESSAGE: std::bad_alloc`).

*Runtime Failure Qualification:* In both cases, the tested unquantized `fp32` ONNX deployment failed during model initialization with `ERROR_CODE: 6 (std::bad_alloc)` on this laptop and loader stack. This confirms an allocation failure on the host environment under the tested runtime stack; the exact allocation site (host system RAM, WebAssembly heap boundary, or GPU buffer allocation) was not isolated via kernel-level memory instrumentation.

---

### Diagnostic Notes on Runtime Behavior

1. **Transformers.js WebGPU (`fp16` & `q4f16`) Generation Collapse & Causal Framing:**
   Across both Intel UHD Graphics and NVIDIA GeForce RTX 2050, executing `onnx-community/Qwen2.5-0.5B-Instruct` via Transformers.js 4.3.0 WebGPU (`fp16` and `q4f16`) resulted in severe generation degradation (`0.00–0.01` accuracy on GSM8K, `0.02–0.07` on MMLU, diverging immediately at token `0` in $\ge 82\%$ of GSM8K and $\ge 89\%$ of MMLU runs). Repeating the NVIDIA `fp16` run (`nvidia-chrome-repeat`) produced identical results (`0.01` GSM8K, `0.07` MMLU), confirming the failure is deterministic rather than transient.
   
   *Causal Qualification:* This degradation is strictly qualified as a failure mode of the evaluated Transformers.js v4.3.0 / ONNX Runtime Web WebGPU execution path and shader translation pipeline on this host, rather than an inherent limitation of the WebGPU standard itself. Specifically, the independent WebLLM runtime operates over WebGPU on the exact same hardware without generation collapse (achieving $0.16–0.19$ on GSM8K and $0.42–0.445$ on MMLU).

2. **Cross-Adapter Precision Drift (MLC `model_dtype` `float16` vs. `float32`):**
   Comparing WebLLM runs on Intel UHD Graphics versus NVIDIA GeForce RTX 2050 reveals that `webllm/q4f16_1` produces inter-GPU numerical drift (diverging on 10/200 MMLU texts and 54/100 GSM8K texts). When using `webllm/q4f32_1`, the cross-GPU disagreement drops to zero: **100.00% exact text agreement (200/200 MMLU and 100/100 GSM8K)**. Crucially, `q4f16_1` and `q4f32_1` differ in MLC `model_dtype` (`float16` vs. `float32`), which affects intermediate activations and accumulation throughout the MLC compilation pipeline.

3. **Browser CPU WASM Sanity Check (`browser/tjs-wasm/q4/cpu-chrome-check`, $N=20$):**
   To isolate whether the failure originated from the ONNX model weights/harness or specifically from the WebGPU execution backend, we executed a 20-item diagnostic subset (10 GSM8K + 10 MMLU) using the same `site/harness.js` runner on the browser WebAssembly (WASM) CPU backend (`q4`).
   
   *WASM Parity Finding:* Comparing `browser/tjs-wasm/q4/cpu-chrome-check` against `native-node-cpu/q4` on this matched subset yields:
   - **20/20 (100.00%) exact character-for-character text match** (`10/10` on GSM8K, `10/10` on MMLU).
   - **20/20 (100.00%) exact raw token-ID match** (`10/10` on GSM8K, `10/10` on MMLU).
   - **0 token divergences** (`n_diverged = 0`).
   On GSM8K, browser WASM achieved `0.40` accuracy—matching the native `fp32` CPU reference accuracy (`0.40`) on the same 10 problems—confirming that the prompt harness and ONNX graph are valid and isolating the collapse specifically to the Transformers.js WebGPU execution path.

4. **Parsing & Truncation Audit (Permissive vs. Strict Extraction):**
   - **MMLU Parsing & Token Caps:** In `native-node-cpu/fp32`, 39 out of 200 MMLU questions were unparsed by the letter-only extractor. Detailed token-length analysis reveals that **38 of these 39 items reached the 5-token cap** without isolating a single letter, while **1 item (`mmlu-1554`) terminated early at 3 tokens** (`"WEP"` + EOS, failing letter-only extraction).
   - **GSM8K Extraction & Truncation:** Standard evaluation uses a regex fallback that extracts the last parsed number from the model's output. In collapsed or drifting runs, model responses often exhaust the 320-token limit without emitting `"The answer is N."` (e.g., 76% of Intel `fp16` runs hit the 320-token cap). The permissive regex occasionally captured an arbitrary trailing integer, yielding spurious matches (e.g., 0.01 accuracy). Enforcing a strict parser (`answer is <num>`) results in 0.0%–1.0% accuracy across all setups because `Qwen2.5-0.5B-Instruct` rarely generates that exact phrasing verbatim.
   - **Valid-Only vs. All-Outcome Agreement:** On MMLU, counting `None == None` as agreement inflates reported agreement (`all_outcome_agree = 50.0%–55.0%` for WebLLM). Restricting agreement strictly to items where both reference and test emitted valid parsed answers (`valid_agree`) reveals true answer concordance: WebLLM achieves **61.5%–64.6% valid agreement** with `native-node-cpu/fp32` (`99/161` to `104/161`).
   - **Flips Decomposition:** In `webllm/q4f16_1/nvidia-chrome`, 30 questions were classified as wrong-to-correct (`w2c`) flips relative to `fp32` reference. However, 12 of these 30 flips were items where the `fp32` CPU reference failed to produce a valid parse (`unparsed_ref_to_correct = 12`), while the remaining 18 were true answer flips from a wrong reference prediction to a correct test prediction.
