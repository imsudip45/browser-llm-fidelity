# Benchmark Summary & Multi-Table Audit

**Reference Baseline:** `native-node-cpu/fp32`  
**Host Hardware:** Single Windows 11 Laptop — 13th Gen Intel(R) Core(TM) i5-13420H CPU, NVIDIA GeForce RTX 2050 (4 GB GDDR6), Intel(R) UHD Graphics (Gen-12LP), 8 GB RAM  
**Software Environment:** Windows 11 x64, Google Chrome 139.0.0.0, Node.js v20+, Transformers.js 4.3.0, WebLLM 0.2.85  

---

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

## Table 4: Parsing & Truncation Audit (Strict vs. Permissive Extraction)

*Evaluates the effect of regex extraction fallbacks and generation token limits across tasks.*

| setup | task | n | standard_acc | strict_acc | acc_parsed | parse_rate | n_unparsed | n_truncated_or_max | gpu |
|---|---|---|---|---|---|---|---|---|---|
| browser/tjs-wasm/q4/cpu-chrome-check | gsm8k | 10 | 0.4 | 0.0 | 0.4 | 1.0 | 0 | 5 | cpu (wasm) |
| browser/tjs-webgpu/fp16/intel-chrome | gsm8k | 100 | 0.0 | 0.0 | 0.0 | 0.56 | 44 | 76 | intel |
| browser/tjs-webgpu/fp16/nvidia-chrome | gsm8k | 100 | 0.01 | 0.0 | 0.02 | 0.51 | 49 | 68 | nvidia |
| browser/tjs-webgpu/fp16/nvidia-chrome-repeat | gsm8k | 100 | 0.01 | 0.0 | 0.02 | 0.51 | 49 | 68 | nvidia |
| browser/tjs-webgpu/q4f16/intel-chrome | gsm8k | 100 | 0.0 | 0.0 | 0.0 | 0.04 | 96 | 89 | intel |
| browser/tjs-webgpu/q4f16/nvidia-chrome | gsm8k | 100 | 0.0 | 0.0 | 0.0 | 0.05 | 95 | 91 | nvidia |
| browser/webllm/q4f16_1/intel-chrome | gsm8k | 100 | 0.16 | 0.0 | 0.16 | 1.0 | 0 | 50 | intel |
| browser/webllm/q4f16_1/nvidia-chrome | gsm8k | 100 | 0.19 | 0.0 | 0.19 | 1.0 | 0 | 49 | nvidia |
| browser/webllm/q4f32_1/intel-chrome | gsm8k | 100 | 0.18 | 0.01 | 0.18 | 1.0 | 0 | 45 | intel |
| browser/webllm/q4f32_1/nvidia-chrome | gsm8k | 100 | 0.18 | 0.01 | 0.18 | 1.0 | 0 | 45 | nvidia |
| native-node-cpu/fp16 | gsm8k | 100 | 0.31 | 0.0 | 0.31 | 1.0 | 0 | 41 | cpu |
| native-node-cpu/fp32 | gsm8k | 100 | 0.29 | 0.0 | 0.29 | 1.0 | 0 | 44 | cpu |
| native-node-cpu/q4 | gsm8k | 100 | 0.2 | 0.01 | 0.2 | 1.0 | 0 | 50 | cpu |
| native-node-cpu/q4f16 | gsm8k | 100 | 0.24 | 0.01 | 0.24 | 1.0 | 0 | 50 | cpu |
| browser/tjs-wasm/q4/cpu-chrome-check | mmlu | 10 | 0.2 | 0.2 | 0.25 | 0.8 | 2 |  | cpu (wasm) |
| browser/tjs-webgpu/fp16/intel-chrome | mmlu | 200 | 0.055 | 0.035 | 0.204 | 0.27 | 146 |  | intel |
| browser/tjs-webgpu/fp16/nvidia-chrome | mmlu | 200 | 0.07 | 0.05 | 0.259 | 0.27 | 146 |  | nvidia |
| browser/tjs-webgpu/fp16/nvidia-chrome-repeat | mmlu | 200 | 0.07 | 0.05 | 0.259 | 0.27 | 146 |  | nvidia |
| browser/tjs-webgpu/q4f16/intel-chrome | mmlu | 200 | 0.025 | 0.005 | 0.278 | 0.09 | 182 |  | intel |
| browser/tjs-webgpu/q4f16/nvidia-chrome | mmlu | 200 | 0.02 | 0.005 | 0.25 | 0.08 | 184 |  | nvidia |
| browser/webllm/q4f16_1/intel-chrome | mmlu | 200 | 0.42 | 0.42 | 0.422 | 0.995 | 1 |  | intel |
| browser/webllm/q4f16_1/nvidia-chrome | mmlu | 200 | 0.43 | 0.43 | 0.43 | 1.0 | 0 |  | nvidia |
| browser/webllm/q4f32_1/intel-chrome | mmlu | 200 | 0.445 | 0.445 | 0.459 | 0.97 | 6 |  | intel |
| browser/webllm/q4f32_1/nvidia-chrome | mmlu | 200 | 0.445 | 0.445 | 0.459 | 0.97 | 6 |  | nvidia |
| native-node-cpu/fp16 | mmlu | 200 | 0.38 | 0.38 | 0.475 | 0.8 | 40 |  | cpu |
| native-node-cpu/fp32 | mmlu | 200 | 0.39 | 0.39 | 0.484 | 0.805 | 39 |  | cpu |
| native-node-cpu/q4 | mmlu | 200 | 0.455 | 0.455 | 0.489 | 0.93 | 14 |  | cpu |
| native-node-cpu/q4f16 | mmlu | 200 | 0.465 | 0.465 | 0.503 | 0.925 | 15 |  | cpu |
