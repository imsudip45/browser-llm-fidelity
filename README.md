# Does Running a Small LLM in the Browser Change Its Answers? Comparing Native and WebGPU Inference Across GPUs

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23760946.svg)](https://doi.org/10.5281/zenodo.23760946)

**Authors:** Sudip Niroula (Lead / First Author) and Mandip Pokharel (Co-Author)  
**Status:** Completed benchmark dataset and reproducible test harness (October 2026)  
**Model Tested:** `Qwen2.5-0.5B-Instruct`  
**Setups Compared:** Native Node.js on CPU (`fp32`, `fp16`, `q4`, `q4f16`) vs. Web Browser (`Transformers.js v4.3.0` WASM & WebGPU, and `WebLLM v0.2.85` WebGPU)  
**Test Hardware:** Single Windows 11 Laptop (13th Gen Intel Core i5-13420H CPU, NVIDIA GeForce RTX 2050 4 GB discrete GPU, and Intel UHD Gen-12LP integrated GPU, 8 GB RAM)  

---

## Summary

Most studies on browser AI only measure how fast a model runs. Developers usually assume that loading the same model in a web browser gives the same answers as running it normally on a computer. To test this assumption, we evaluated `Qwen2.5-0.5B-Instruct` across 14 run logs comprising **3,920 individual item evaluations** (13 full runs of 300 questions—200 MMLU multiple-choice questions and 100 GSM8K math problems—plus a 20-question diagnostic browser WASM CPU run) with randomness turned off (greedy decoding). Using one shared prompt and scoring harness (`site/harness.js`) across 28 task configurations on the same laptop, we compared native CPU baselines against browser runs across two GPUs (NVIDIA GeForce RTX 2050 discrete and Intel UHD Graphics integrated).

## What We Found

1. **Same weight file, reproducible browser WebGPU degradation (Transformers.js):** When executing `fp16` and `q4f16` ONNX model files natively on the CPU, generation behaves normally (`fp16` achieves 98.1% valid agreement with `fp32` on MMLU and 31.0% accuracy on GSM8K). However, loading those identical ONNX graphs into Transformers.js v4.3.0 WebGPU causes severe output degradation on both Intel and NVIDIA GPUs (0%–1% accuracy on GSM8K, 2%–7% on MMLU, diverging immediately at token 0 in $\ge 82\%$ of cases). Running a matched 20-item diagnostic check on browser WebAssembly CPU (`tjs-wasm/q4/cpu-chrome-check`) produces **100% exact text (20/20) and exact token-ID (20/20) parity** against native Node.js CPU `q4`, confirming that the model weights and prompt harness are identical and isolating the collapse specifically to the Transformers.js WebGPU execution path on this host.
2. **Overall accuracy hides answer flips and unparsed responses:** On MMLU multiple-choice questions, WebLLM builds achieve 42.0%–44.5% overall accuracy compared to 39.0% for native `fp32` CPU. However, looking at individual questions reveals that native `fp32` CPU fails to produce an isolated letter on 39/200 questions (80.5% parse rate). Restricting comparison strictly to valid parsed pairs (`valid_agree`), WebLLM agrees on **61.5%–64.6%** of valid answers. In `webllm/q4f16_1/nvidia-chrome`, 30 questions flipped from wrong to correct (`w2c`), but 12 of those 30 were items where the CPU reference failed to parse (`unparsed_ref_to_correct = 12`).
3. **32-bit intermediate math eliminates cross-GPU drift:** In WebLLM, 16-bit intermediate math (`q4f16_1`) causes subtle numerical discrepancies between Intel integrated and NVIDIA discrete GPUs, resulting in divergent text on 5% of MMLU items and 54% of GSM8K items. Upgrading to 32-bit accumulation (`q4f32_1`) achieves **100% exact text agreement (300/300 questions)** between Intel UHD and NVIDIA RTX 2050 GPUs.
4. **Host memory constraints on full-precision WebGPU:** Attempting to initialize unquantized `fp32` ONNX models in browser WebGPU results in an immediate out-of-memory abort (`std::bad_alloc`) on both GPUs under our host's 4 GB VRAM / 8 GB system RAM configuration, demonstrating why client-side deployment strictly mandates 16-bit or 4-bit compression.

---

## Research Questions

1. **RQ1 (Runtime effect):** With the exact same weight file, does running in the browser change answers compared with native inference?
2. **RQ2 (GPU vendor effect):** Does the same browser setup give different answers on an NVIDIA discrete GPU versus an Intel integrated GPU within the same machine?
3. **RQ3 (Deployment effect):** How far are ready-made browser builds (such as WebLLM `q4f16_1` and `q4f32_1`) from a full-precision native reference, and how much of that gap comes from compression versus the browser?
4. **RQ4 (Hidden answer flips):** Do overall accuracy numbers hide individual questions flipping between right and wrong?

For a detailed plain-language guide to the study design, read [docs/ABOUT_THE_RESEARCH.md](docs/ABOUT_THE_RESEARCH.md). The initial literature check and planning notes are in [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md), and hardware/diagnostic logs are in [results/hardware.md](results/hardware.md) and [results/notes.md](results/notes.md).

---

## Benchmark Results & Multi-Table Audit

Reference Baseline: `native-node-cpu/fp32`  
Host Hardware: Single Windows 11 Laptop — 13th Gen Intel Core i5-13420H, NVIDIA GeForce RTX 2050 (4 GB GDDR6), Intel UHD Graphics (Gen-12LP), 8 GB RAM  
Full machine-readable exports and parsing audits are available in [results/summary.md](results/summary.md) and [results/summary.csv](results/summary.csv).

## Table 1: Primary Comparison vs. Full-Precision Reference (`native-node-cpu/fp32`)

*Evaluates all 28 experimental configurations against the native CPU fp32 baseline. Separates unparsed predictions, reports valid-only vs. all-outcome agreement, decomposes flips, and provides divergence token distributions.*

| setup | task | n | acc | ref_acc | strict_acc | n_unparsed | parse_rate | acc_parsed | valid_agree | all_outcome_agree | agree_ci | n_both_valid | n_joint_invalid | flips_c2w | flips_w2c | unparsed_ref_to_correct | correct_to_unparsed_test | mcnemar_p | exact_text | median_first_div_token | n_diverged | n_div_at_0 | n_div_at_1 | n_truncated_or_max | gpu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| browser/tjs-wasm/q4/cpu-chrome-check | gsm8k | 10 | 0.4 | 0.4 | 0.0 | 0 | 1.0 | 0.4 | 0.2 | 0.2 | [0.000, 0.500] | 10 | 0 | 2 | 2 | 0 | 0 | 1.0 | 0.0 | 0 | 10 | 8 | 1 | 5 | cpu (wasm) |
| browser/tjs-webgpu/fp16/intel-chrome | gsm8k | 100 | 0.0 | 0.29 | 0.0 | 44 | 0.56 | 0.0 | 0.054 | 0.03 | [0.000, 0.070] | 56 | 0 | 29 | 0 | 0 | 12 | < 1e-4 | 0.0 | 0 | 100 | 87 | 4 | 76 | intel |
| browser/tjs-webgpu/fp16/nvidia-chrome | gsm8k | 100 | 0.01 | 0.29 | 0.0 | 49 | 0.51 | 0.02 | 0.039 | 0.02 | [0.000, 0.050] | 51 | 0 | 29 | 1 | 0 | 15 | < 1e-4 | 0.0 | 0 | 100 | 82 | 8 | 68 | nvidia |
| browser/tjs-webgpu/fp16/nvidia-chrome-repeat | gsm8k | 100 | 0.01 | 0.29 | 0.0 | 49 | 0.51 | 0.02 | 0.039 | 0.02 | [0.000, 0.050] | 51 | 0 | 29 | 1 | 0 | 15 | < 1e-4 | 0.0 | 0 | 100 | 82 | 8 | 68 | nvidia |
| browser/tjs-webgpu/q4f16/intel-chrome | gsm8k | 100 | 0.0 | 0.29 | 0.0 | 96 | 0.04 | 0.0 | 0.0 | 0.0 | [0.000, 0.000] | 4 | 0 | 29 | 0 | 0 | 26 | < 1e-4 | 0.0 | 0 | 100 | 99 | 1 | 89 | intel |
| browser/tjs-webgpu/q4f16/nvidia-chrome | gsm8k | 100 | 0.0 | 0.29 | 0.0 | 95 | 0.05 | 0.0 | 0.0 | 0.0 | [0.000, 0.000] | 5 | 0 | 29 | 0 | 0 | 27 | < 1e-4 | 0.0 | 0 | 100 | 99 | 1 | 91 | nvidia |
| browser/webllm/q4f16_1/intel-chrome | gsm8k | 100 | 0.16 | 0.29 | 0.0 | 0 | 1.0 | 0.16 | 0.15 | 0.15 | [0.080, 0.220] | 100 | 0 | 19 | 6 | 0 | 0 | 0.0146 | 0.0 | 1 | 100 | 0 | 69 | 50 | intel |
| browser/webllm/q4f16_1/nvidia-chrome | gsm8k | 100 | 0.19 | 0.29 | 0.0 | 0 | 1.0 | 0.19 | 0.19 | 0.19 | [0.120, 0.270] | 100 | 0 | 16 | 6 | 0 | 0 | 0.0525 | 0.0 | 1 | 100 | 0 | 69 | 49 | nvidia |
| browser/webllm/q4f32_1/intel-chrome | gsm8k | 100 | 0.18 | 0.29 | 0.01 | 0 | 1.0 | 0.18 | 0.2 | 0.2 | [0.120, 0.290] | 100 | 0 | 16 | 5 | 0 | 0 | 0.0266 | 0.0 | 1 | 100 | 0 | 70 | 45 | intel |
| browser/webllm/q4f32_1/nvidia-chrome | gsm8k | 100 | 0.18 | 0.29 | 0.01 | 0 | 1.0 | 0.18 | 0.2 | 0.2 | [0.120, 0.290] | 100 | 0 | 16 | 5 | 0 | 0 | 0.0266 | 0.0 | 1 | 100 | 0 | 70 | 45 | nvidia |
| native-node-cpu/fp16 | gsm8k | 100 | 0.31 | 0.29 | 0.0 | 0 | 1.0 | 0.31 | 0.74 | 0.74 | [0.650, 0.820] | 100 | 0 | 2 | 4 | 0 | 0 | 0.6875 | 0.54 | 84.5 | 46 | 0 | 3 | 41 | cpu |
| native-node-cpu/fp32 | gsm8k | 100 | 0.29 | 0.29 | 0.0 | 0 | 1.0 | 0.29 | 1.0 | 1.0 | [1.000, 1.000] | 100 | 0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 |  | 0 | 0 | 0 | 44 | cpu |
| native-node-cpu/q4 | gsm8k | 100 | 0.2 | 0.29 | 0.01 | 0 | 1.0 | 0.2 | 0.19 | 0.19 | [0.120, 0.270] | 100 | 0 | 13 | 4 | 0 | 0 | 0.049 | 0.0 | 0 | 100 | 72 | 5 | 50 | cpu |
| native-node-cpu/q4f16 | gsm8k | 100 | 0.24 | 0.29 | 0.01 | 0 | 1.0 | 0.24 | 0.23 | 0.23 | [0.150, 0.320] | 100 | 0 | 10 | 5 | 0 | 0 | 0.3018 | 0.0 | 0 | 100 | 74 | 7 | 50 | cpu |
| browser/tjs-wasm/q4/cpu-chrome-check | mmlu | 10 | 0.2 | 0.3 | 0.2 | 2 | 0.8 | 0.25 | 0.286 | 0.4 | [0.100, 0.700] | 7 | 2 | 1 | 0 | 0 | 0 | 1.0 | 0.3 | 0 | 7 | 6 | 1 |  | cpu (wasm) |
| browser/tjs-webgpu/fp16/intel-chrome | mmlu | 200 | 0.055 | 0.39 | 0.035 | 146 | 0.27 | 0.204 | 0.302 | 0.205 | [0.155, 0.260] | 43 | 28 | 73 | 6 | 2 | 56 | < 1e-4 | 0.0 | 0 | 200 | 196 | 2 |  | intel |
| browser/tjs-webgpu/fp16/nvidia-chrome | mmlu | 200 | 0.07 | 0.39 | 0.05 | 146 | 0.27 | 0.259 | 0.239 | 0.21 | [0.155, 0.270] | 46 | 31 | 72 | 8 | 3 | 56 | < 1e-4 | 0.005 | 0 | 199 | 196 | 2 |  | nvidia |
| browser/tjs-webgpu/fp16/nvidia-chrome-repeat | mmlu | 200 | 0.07 | 0.39 | 0.05 | 146 | 0.27 | 0.259 | 0.239 | 0.21 | [0.155, 0.270] | 46 | 31 | 72 | 8 | 3 | 56 | < 1e-4 | 0.005 | 0 | 199 | 196 | 2 |  | nvidia |
| browser/tjs-webgpu/q4f16/intel-chrome | mmlu | 200 | 0.025 | 0.39 | 0.005 | 182 | 0.09 | 0.278 | 0.357 | 0.2 | [0.150, 0.260] | 14 | 35 | 77 | 4 | 1 | 75 | < 1e-4 | 0.01 | 0 | 198 | 179 | 18 |  | intel |
| browser/tjs-webgpu/q4f16/nvidia-chrome | mmlu | 200 | 0.02 | 0.39 | 0.005 | 184 | 0.08 | 0.25 | 0.154 | 0.19 | [0.140, 0.250] | 13 | 36 | 78 | 4 | 0 | 74 | < 1e-4 | 0.0 | 0 | 200 | 180 | 19 |  | nvidia |
| browser/webllm/q4f16_1/intel-chrome | mmlu | 200 | 0.42 | 0.39 | 0.42 | 1 | 0.995 | 0.422 | 0.615 | 0.5 | [0.430, 0.570] | 161 | 1 | 23 | 29 | 11 | 0 | 0.4885 | 0.35 | 0 | 130 | 100 | 24 |  | intel |
| browser/webllm/q4f16_1/nvidia-chrome | mmlu | 200 | 0.43 | 0.39 | 0.43 | 0 | 1.0 | 0.43 | 0.621 | 0.5 | [0.430, 0.570] | 161 | 0 | 22 | 30 | 12 | 0 | 0.3317 | 0.345 | 0 | 131 | 100 | 28 |  | nvidia |
| browser/webllm/q4f32_1/intel-chrome | mmlu | 200 | 0.445 | 0.39 | 0.445 | 6 | 0.97 | 0.459 | 0.646 | 0.55 | [0.480, 0.620] | 161 | 6 | 19 | 30 | 10 | 0 | 0.1524 | 0.37 | 0 | 126 | 91 | 30 |  | intel |
| browser/webllm/q4f32_1/nvidia-chrome | mmlu | 200 | 0.445 | 0.39 | 0.445 | 6 | 0.97 | 0.459 | 0.646 | 0.55 | [0.480, 0.620] | 161 | 6 | 19 | 30 | 10 | 0 | 0.1524 | 0.37 | 0 | 126 | 91 | 30 |  | nvidia |
| native-node-cpu/fp16 | mmlu | 200 | 0.38 | 0.39 | 0.38 | 40 | 0.8 | 0.475 | 0.981 | 0.98 | [0.960, 0.995] | 160 | 39 | 2 | 0 | 0 | 0 | 0.5 | 0.91 | 1 | 18 | 5 | 12 |  | cpu |
| native-node-cpu/fp32 | mmlu | 200 | 0.39 | 0.39 | 0.39 | 39 | 0.805 | 0.484 | 1.0 | 1.0 | [1.000, 1.000] | 161 | 39 | 0 | 0 | 0 | 0 | 1.0 | 1.0 |  | 0 | 0 | 0 |  | cpu |
| native-node-cpu/q4 | mmlu | 200 | 0.455 | 0.39 | 0.455 | 14 | 0.93 | 0.489 | 0.752 | 0.675 | [0.610, 0.735] | 161 | 14 | 11 | 24 | 7 | 0 | 0.041 | 0.485 | 0 | 103 | 66 | 29 |  | cpu |
| native-node-cpu/q4f16 | mmlu | 200 | 0.465 | 0.39 | 0.465 | 15 | 0.925 | 0.503 | 0.752 | 0.68 | [0.615, 0.740] | 161 | 15 | 10 | 25 | 8 | 0 | 0.0167 | 0.475 | 0 | 105 | 65 | 32 |  | cpu |

---

## Table 2: Direct Matched-Dtype Comparisons: Native CPU vs. Browser (RQ1)

*Directly tests runtime effects by holding model weights and quantization format strictly constant while changing only the runtime environment (Native Node.js CPU vs. Browser WebGPU/WASM).*

| browser_setup | native_ref_setup | task | n | browser_acc | native_acc | valid_agree | all_outcome_agree | exact_text | exact_token_ids | median_first_div_token | n_diverged | n_div_at_0 | n_both_valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| browser/tjs-wasm/q4/cpu-chrome-check | native-node-cpu/q4 | gsm8k | 10 | 0.4 | 0.4 | 1.0 | 1.0 | 1.0 | 1.0 |  | 0 | 0 | 10 |
| browser/tjs-webgpu/fp16/intel-chrome | native-node-cpu/fp16 | gsm8k | 100 | 0.0 | 0.31 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 100 | 87 | 56 |
| browser/tjs-webgpu/fp16/nvidia-chrome | native-node-cpu/fp16 | gsm8k | 100 | 0.01 | 0.31 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 100 | 82 | 51 |
| browser/tjs-webgpu/fp16/nvidia-chrome-repeat | native-node-cpu/fp16 | gsm8k | 100 | 0.01 | 0.31 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 100 | 82 | 51 |
| browser/tjs-webgpu/q4f16/intel-chrome | native-node-cpu/q4f16 | gsm8k | 100 | 0.0 | 0.24 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 100 | 100 | 4 |
| browser/tjs-webgpu/q4f16/nvidia-chrome | native-node-cpu/q4f16 | gsm8k | 100 | 0.0 | 0.24 | 0.2 | 0.01 | 0.0 | 0.0 | 0 | 100 | 100 | 5 |
| browser/tjs-wasm/q4/cpu-chrome-check | native-node-cpu/q4 | mmlu | 10 | 0.2 | 0.2 | 1.0 | 1.0 | 1.0 | 1.0 |  | 0 | 0 | 8 |
| browser/tjs-webgpu/fp16/intel-chrome | native-node-cpu/fp16 | mmlu | 200 | 0.055 | 0.38 | 0.302 | 0.21 | 0.0 | 0.0 | 0 | 200 | 196 | 43 |
| browser/tjs-webgpu/fp16/nvidia-chrome | native-node-cpu/fp16 | mmlu | 200 | 0.07 | 0.38 | 0.244 | 0.21 | 0.005 | 0.005 | 0 | 199 | 196 | 45 |
| browser/tjs-webgpu/fp16/nvidia-chrome-repeat | native-node-cpu/fp16 | mmlu | 200 | 0.07 | 0.38 | 0.244 | 0.21 | 0.005 | 0.005 | 0 | 199 | 196 | 45 |
| browser/tjs-webgpu/q4f16/intel-chrome | native-node-cpu/q4f16 | mmlu | 200 | 0.025 | 0.465 | 0.176 | 0.085 | 0.005 | 0.005 | 0 | 199 | 193 | 17 |
| browser/tjs-webgpu/q4f16/nvidia-chrome | native-node-cpu/q4f16 | mmlu | 200 | 0.02 | 0.465 | 0.062 | 0.08 | 0.0 | 0.0 | 0 | 200 | 193 | 16 |

*Note on browser WASM check:* `browser/tjs-wasm/q4/cpu-chrome-check` demonstrates **100% exact text (20/20) and exact token-ID (20/20) match** against `native-node-cpu/q4` on the matched diagnostic subset, confirming the execution graph and prompt harness are identical between Node.js and browser WASM CPU.

---

## Table 3: Cross-GPU Pairwise Consistency: Intel UHD vs. NVIDIA RTX 2050 (RQ2)

*Directly compares the Intel UHD Graphics integrated GPU run against the NVIDIA GeForce RTX 2050 discrete GPU run on the same laptop for each browser engine build.*

| build | task | n | intel_acc | nvidia_acc | valid_agree | all_outcome_agree | exact_text | n_both_valid | n_diff_predictions | n_diff_text |
|---|---|---|---|---|---|---|---|---|---|---|
| tjs-webgpu/fp16 | gsm8k | 100 | 0.0 | 0.01 | 0.571 | 0.59 | 0.27 | 42 | 41 | 73 |
| tjs-webgpu/q4f16 | gsm8k | 100 | 0.0 | 0.0 | 1.0 | 0.97 | 0.36 | 3 | 3 | 64 |
| webllm/q4f16_1 | gsm8k | 100 | 0.16 | 0.19 | 0.59 | 0.59 | 0.46 | 100 | 41 | 54 |
| webllm/q4f32_1 | gsm8k | 100 | 0.18 | 0.18 | 1.0 | 1.0 | 1.0 | 100 | 0 | 0 |
| tjs-webgpu/fp16 | mmlu | 200 | 0.055 | 0.07 | 0.902 | 0.85 | 0.64 | 41 | 30 | 72 |
| tjs-webgpu/q4f16 | mmlu | 200 | 0.025 | 0.02 | 0.917 | 0.945 | 0.7 | 12 | 11 | 60 |
| webllm/q4f16_1 | mmlu | 200 | 0.42 | 0.43 | 0.975 | 0.97 | 0.95 | 199 | 6 | 10 |
| webllm/q4f32_1 | mmlu | 200 | 0.445 | 0.445 | 1.0 | 1.0 | 1.0 | 194 | 0 | 0 |

*Key takeaway:* Under 16-bit intermediate math (`webllm/q4f16_1`), inter-GPU numerical drift causes 5% of MMLU texts and 54% of GSM8K texts to diverge. Upgrading to 32-bit intermediate accumulation (`webllm/q4f32_1`) achieves **100% exact text agreement (300/300 items)** and 0 divergent predictions across Intel and NVIDIA GPUs.

---

## Repository Layout

| Path | What it is |
|---|---|
| `site/` | Browser runner page (`index.html`, `app.js`, `worker.js`) |
| `site/harness.js` | Shared prompts and answer extraction used by both native and browser runs |
| `site/questions_v1.json` | Fixed question set: 200 MMLU + 100 GSM8K, seed 2026 (also mirrored in `data/questions_v1.json`) |
| `run_native.mjs` | Native runner (Node.js, CPU via `onnxruntime-node`) |
| `runall.sh` | Runs all four native CPU baselines (`fp32`, `fp16`, `q4`, `q4f16`) |
| `analyze.py` | Multi-table audit script; evaluates all setups against baseline, exporting tables to `results/summary.md` and `results/summary.csv` |
| `scripts/make_questions.py` | Rebuilds the fixed 300-question set from the original Hugging Face datasets |
| `results/` | Raw JSON result logs (14 files), `hardware.md`, `notes.md`, `summary.md`, and `summary.csv` |
| `RUN_GUIDE.md` | Step-by-step instructions for reproducing the browser and native experiments |
| `CITATION.cff` | Citation metadata for Zenodo and academic references |

---

## Quick Start (How to Reproduce)

### 1. Native CPU Baselines

```bash
npm install
node run_native.mjs onnx-community/Qwen2.5-0.5B-Instruct fp32 all
node run_native.mjs onnx-community/Qwen2.5-0.5B-Instruct fp16 all
node run_native.mjs onnx-community/Qwen2.5-0.5B-Instruct q4 all
node run_native.mjs onnx-community/Qwen2.5-0.5B-Instruct q4f16 all
```

### 2. Browser Runs

```bash
python -m http.server 8000 --directory site
```

Open http://localhost:8000 in Google Chrome and follow [RUN_GUIDE.md](RUN_GUIDE.md).

### 3. Statistical Analysis & Summary Generation

```bash
pip install -r requirements.txt
python analyze.py results native-node-cpu/fp32
```

## Fixed Experimental Settings

- **System Prompt:** `"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."`
- **Decoding:** Greedy decoding (temperature = 0, no sampling), identical prompt templates across all setups.
- **MMLU (200 questions):** Stratified across all 57 subjects (capped at 4 per subject); answer with a single letter (A–D), max 5 new tokens.
- **GSM8K (100 problems):** Step-by-step math reasoning ending with "The answer is N.", max 320 new tokens.
- **Library Versions:** Transformers.js 4.3.0 (both native Node.js and browser), WebLLM 0.2.85, Chrome 139.0.0.0.

## Data & Model Sources

- **MMLU:** https://huggingface.co/datasets/cais/mmlu
- **GSM8K:** https://huggingface.co/datasets/openai/gsm8k
- **ONNX Weights:** https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct

## Authors & Contributions (CRediT)

- **Sudip Niroula** (Lead / First Author): Conceptualization, Methodology, Software (experiment harness and analysis pipeline), Formal Analysis, Writing – Original Draft, Documentation.
- **Mandip Pokharel** (Co-Author): Investigation (native and dual-GPU browser benchmark execution on the Intel Core i5-13420H / NVIDIA RTX 2050 / Intel UHD testbed), Data Curation, Validation.

## Citation

If you use this benchmark dataset or evaluation harness in your research, please cite:

> Niroula, S., & Pokharel, M. (2026). *Does Running a Small LLM in the Browser Change Its Answers? Comparing Native and WebGPU Inference Across GPUs*. Zenodo. https://doi.org/10.5281/zenodo.23760946

```bibtex
@software{niroula2026browserllmfidelity,
  author       = {Niroula, Sudip and Pokharel, Mandip},
  title        = {{Does Running a Small LLM in the Browser Change Its Answers? Comparing Native and WebGPU Inference Across GPUs}},
  month        = oct,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.1.0},
  doi          = {10.5281/zenodo.23760946},
  url          = {https://doi.org/10.5281/zenodo.23760946}
}
```

## License

MIT
