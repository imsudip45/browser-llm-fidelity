# Benchmark Summary & Multi-Table Static Audit

**Reference Baseline:** `native-node-cpu/fp32`  
**Host Hardware:** Single Windows 11 Laptop — 13th Gen Intel(R) Core(TM) i5-13420H CPU, NVIDIA GeForce RTX 2050 (4 GB GDDR6), Intel(R) UHD Graphics (Gen-12LP), 8 GB RAM  
**Software Environment:** Windows 11 x64, Google Chrome 139.0.0.0, Node.js v20+, Transformers.js 4.3.0, WebLLM 0.2.85  
**Tokenizer Audit:** `Qwen/Qwen2.5-0.5B-Instruct` (Snapshot Commit: `7ae557604adf67be50417f59c2c2f167def9a775`)  

---

## Table 1: Primary Comparison vs. Full-Precision Reference (`native-node-cpu/fp32`)

*Evaluates all 28 experimental configurations against the native CPU fp32 baseline. Separates unparsed predictions, reports valid-only and all-outcome agreement with explicit fractions, decomposes flips, and tracks token sequence divergence across all paired items.*

| setup | task | n | acc | ref_acc | strict_acc | n_unparsed | parse_rate | acc_parsed | valid_agree | all_outcome_agree | agree_ci | n_both_valid | n_joint_invalid | flips_c2w | flips_w2c | unparsed_ref_to_correct | correct_to_unparsed_test | mcnemar_p | exact_text | median_first_div_token | n_diverged | n_div_at_0 | n_div_at_1 | div_source | n_at_token_cap | gpu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| browser/tjs-wasm/q4/cpu-chrome-check | gsm8k | 10 | 0.4 | 0.4 | 0.0 | 0 | 1.0 | 4/10 (40.00%) | 2/10 (20.00%) | 2/10 (20.00%) | [0.000, 0.500] | 10 | 0 | 2 | 2 | 0 | 0 | 1.0 | 0/10 (0.00%) | 0 | 10 | 8 | 1 | raw_token_ids | 5 | cpu (wasm) |
| browser/tjs-webgpu/fp16/intel-chrome | gsm8k | 100 | 0.0 | 0.29 | 0.0 | 44 | 0.56 | 0/56 (0.00%) | 3/56 (5.36%) | 3/100 (3.00%) | [0.000, 0.070] | 56 | 0 | 29 | 0 | 0 | 12 | < 1e-4 | 0/100 (0.00%) | 0 | 100 | 87 | 4 | raw_token_ids | 76 | intel |
| browser/tjs-webgpu/fp16/nvidia-chrome | gsm8k | 100 | 0.01 | 0.29 | 0.0 | 49 | 0.51 | 1/51 (1.96%) | 2/51 (3.92%) | 2/100 (2.00%) | [0.000, 0.050] | 51 | 0 | 29 | 1 | 0 | 15 | < 1e-4 | 0/100 (0.00%) | 0 | 100 | 82 | 8 | raw_token_ids | 68 | nvidia |
| browser/tjs-webgpu/fp16/nvidia-chrome-repeat | gsm8k | 100 | 0.01 | 0.29 | 0.0 | 49 | 0.51 | 1/51 (1.96%) | 2/51 (3.92%) | 2/100 (2.00%) | [0.000, 0.050] | 51 | 0 | 29 | 1 | 0 | 15 | < 1e-4 | 0/100 (0.00%) | 0 | 100 | 82 | 8 | raw_token_ids | 68 | nvidia |
| browser/tjs-webgpu/q4f16/intel-chrome | gsm8k | 100 | 0.0 | 0.29 | 0.0 | 96 | 0.04 | 0/4 (0.00%) | 0/4 (0.00%) | 0/100 (0.00%) | [0.000, 0.000] | 4 | 0 | 29 | 0 | 0 | 26 | < 1e-4 | 0/100 (0.00%) | 0 | 100 | 99 | 1 | raw_token_ids | 89 | intel |
| browser/tjs-webgpu/q4f16/nvidia-chrome | gsm8k | 100 | 0.0 | 0.29 | 0.0 | 95 | 0.05 | 0/5 (0.00%) | 0/5 (0.00%) | 0/100 (0.00%) | [0.000, 0.000] | 5 | 0 | 29 | 0 | 0 | 27 | < 1e-4 | 0/100 (0.00%) | 0 | 100 | 99 | 1 | raw_token_ids | 91 | nvidia |
| browser/webllm/q4f16_1/intel-chrome | gsm8k | 100 | 0.16 | 0.29 | 0.0 | 0 | 1.0 | 16/100 (16.00%) | 15/100 (15.00%) | 15/100 (15.00%) | [0.080, 0.220] | 100 | 0 | 19 | 6 | 0 | 0 | 0.0146 | 0/100 (0.00%) | 1 | 100 | 0 | 69 | retokenized_text | 50 | intel |
| browser/webllm/q4f16_1/nvidia-chrome | gsm8k | 100 | 0.19 | 0.29 | 0.0 | 0 | 1.0 | 19/100 (19.00%) | 19/100 (19.00%) | 19/100 (19.00%) | [0.120, 0.270] | 100 | 0 | 16 | 6 | 0 | 0 | 0.0525 | 0/100 (0.00%) | 1 | 100 | 0 | 69 | retokenized_text | 49 | nvidia |
| browser/webllm/q4f32_1/intel-chrome | gsm8k | 100 | 0.18 | 0.29 | 0.01 | 0 | 1.0 | 18/100 (18.00%) | 20/100 (20.00%) | 20/100 (20.00%) | [0.120, 0.290] | 100 | 0 | 16 | 5 | 0 | 0 | 0.0266 | 0/100 (0.00%) | 1 | 100 | 0 | 70 | retokenized_text | 45 | intel |
| browser/webllm/q4f32_1/nvidia-chrome | gsm8k | 100 | 0.18 | 0.29 | 0.01 | 0 | 1.0 | 18/100 (18.00%) | 20/100 (20.00%) | 20/100 (20.00%) | [0.120, 0.290] | 100 | 0 | 16 | 5 | 0 | 0 | 0.0266 | 0/100 (0.00%) | 1 | 100 | 0 | 70 | retokenized_text | 45 | nvidia |
| native-node-cpu/fp16 | gsm8k | 100 | 0.31 | 0.29 | 0.0 | 0 | 1.0 | 31/100 (31.00%) | 74/100 (74.00%) | 74/100 (74.00%) | [0.650, 0.820] | 100 | 0 | 2 | 4 | 0 | 0 | 0.6875 | 54/100 (54.00%) | 84.5 | 46 | 0 | 3 | raw_token_ids | 41 | cpu |
| native-node-cpu/fp32 | gsm8k | 100 | 0.29 | 0.29 | 0.0 | 0 | 1.0 | 29/100 (29.00%) | 100/100 (100.00%) | 100/100 (100.00%) | [1.000, 1.000] | 100 | 0 | 0 | 0 | 0 | 0 | 1.0 | 100/100 (100.00%) |  | 0 | 0 | 0 | raw_token_ids | 44 | cpu |
| native-node-cpu/q4 | gsm8k | 100 | 0.2 | 0.29 | 0.01 | 0 | 1.0 | 20/100 (20.00%) | 19/100 (19.00%) | 19/100 (19.00%) | [0.120, 0.270] | 100 | 0 | 13 | 4 | 0 | 0 | 0.049 | 0/100 (0.00%) | 0 | 100 | 72 | 5 | raw_token_ids | 50 | cpu |
| native-node-cpu/q4f16 | gsm8k | 100 | 0.24 | 0.29 | 0.01 | 0 | 1.0 | 24/100 (24.00%) | 23/100 (23.00%) | 23/100 (23.00%) | [0.150, 0.320] | 100 | 0 | 10 | 5 | 0 | 0 | 0.3018 | 0/100 (0.00%) | 0 | 100 | 74 | 7 | raw_token_ids | 50 | cpu |
| browser/tjs-wasm/q4/cpu-chrome-check | mmlu | 10 | 0.2 | 0.3 | 0.2 | 2 | 0.8 | 2/8 (25.00%) | 2/7 (28.57%) | 4/10 (40.00%) | [0.100, 0.700] | 7 | 2 | 1 | 0 | 0 | 0 | 1.0 | 3/10 (30.00%) | 0 | 7 | 6 | 1 | raw_token_ids | 2 | cpu (wasm) |
| browser/tjs-webgpu/fp16/intel-chrome | mmlu | 200 | 0.055 | 0.39 | 0.035 | 146 | 0.27 | 11/54 (20.37%) | 13/43 (30.23%) | 41/200 (20.50%) | [0.155, 0.260] | 43 | 28 | 73 | 6 | 2 | 56 | < 1e-4 | 0/200 (0.00%) | 0 | 200 | 196 | 2 | raw_token_ids | 180 | intel |
| browser/tjs-webgpu/fp16/nvidia-chrome | mmlu | 200 | 0.07 | 0.39 | 0.05 | 146 | 0.27 | 14/54 (25.93%) | 11/46 (23.91%) | 42/200 (21.00%) | [0.155, 0.270] | 46 | 31 | 72 | 8 | 3 | 56 | < 1e-4 | 1/200 (0.50%) | 0 | 199 | 196 | 2 | raw_token_ids | 187 | nvidia |
| browser/tjs-webgpu/fp16/nvidia-chrome-repeat | mmlu | 200 | 0.07 | 0.39 | 0.05 | 146 | 0.27 | 14/54 (25.93%) | 11/46 (23.91%) | 42/200 (21.00%) | [0.155, 0.270] | 46 | 31 | 72 | 8 | 3 | 56 | < 1e-4 | 1/200 (0.50%) | 0 | 199 | 196 | 2 | raw_token_ids | 187 | nvidia |
| browser/tjs-webgpu/q4f16/intel-chrome | mmlu | 200 | 0.025 | 0.39 | 0.005 | 182 | 0.09 | 5/18 (27.78%) | 5/14 (35.71%) | 40/200 (20.00%) | [0.150, 0.260] | 14 | 35 | 77 | 4 | 1 | 75 | < 1e-4 | 2/200 (1.00%) | 0 | 198 | 179 | 18 | raw_token_ids | 155 | intel |
| browser/tjs-webgpu/q4f16/nvidia-chrome | mmlu | 200 | 0.02 | 0.39 | 0.005 | 184 | 0.08 | 4/16 (25.00%) | 2/13 (15.38%) | 38/200 (19.00%) | [0.140, 0.250] | 13 | 36 | 78 | 4 | 0 | 74 | < 1e-4 | 0/200 (0.00%) | 0 | 200 | 180 | 19 | raw_token_ids | 160 | nvidia |
| browser/webllm/q4f16_1/intel-chrome | mmlu | 200 | 0.42 | 0.39 | 0.42 | 1 | 0.995 | 84/199 (42.21%) | 99/161 (61.49%) | 100/200 (50.00%) | [0.430, 0.570] | 161 | 1 | 23 | 29 | 11 | 0 | 0.4885 | 70/200 (35.00%) | 0 | 130 | 100 | 24 | retokenized_text | 39 | intel |
| browser/webllm/q4f16_1/nvidia-chrome | mmlu | 200 | 0.43 | 0.39 | 0.43 | 0 | 1.0 | 86/200 (43.00%) | 100/161 (62.11%) | 100/200 (50.00%) | [0.430, 0.570] | 161 | 0 | 22 | 30 | 12 | 0 | 0.3317 | 69/200 (34.50%) | 0 | 131 | 100 | 28 | retokenized_text | 36 | nvidia |
| browser/webllm/q4f32_1/intel-chrome | mmlu | 200 | 0.445 | 0.39 | 0.445 | 6 | 0.97 | 89/194 (45.88%) | 104/161 (64.60%) | 110/200 (55.00%) | [0.480, 0.620] | 161 | 6 | 19 | 30 | 10 | 0 | 0.1524 | 74/200 (37.00%) | 0 | 126 | 91 | 30 | retokenized_text | 40 | intel |
| browser/webllm/q4f32_1/nvidia-chrome | mmlu | 200 | 0.445 | 0.39 | 0.445 | 6 | 0.97 | 89/194 (45.88%) | 104/161 (64.60%) | 110/200 (55.00%) | [0.480, 0.620] | 161 | 6 | 19 | 30 | 10 | 0 | 0.1524 | 74/200 (37.00%) | 0 | 126 | 91 | 30 | retokenized_text | 40 | nvidia |
| native-node-cpu/fp16 | mmlu | 200 | 0.38 | 0.39 | 0.38 | 40 | 0.8 | 76/160 (47.50%) | 157/160 (98.12%) | 196/200 (98.00%) | [0.960, 0.995] | 160 | 39 | 2 | 0 | 0 | 0 | 0.5 | 182/200 (91.00%) | 1 | 18 | 5 | 12 | raw_token_ids | 91 | cpu |
| native-node-cpu/fp32 | mmlu | 200 | 0.39 | 0.39 | 0.39 | 39 | 0.805 | 78/161 (48.45%) | 161/161 (100.00%) | 200/200 (100.00%) | [1.000, 1.000] | 161 | 39 | 0 | 0 | 0 | 0 | 1.0 | 200/200 (100.00%) |  | 0 | 0 | 0 | raw_token_ids | 104 | cpu |
| native-node-cpu/q4 | mmlu | 200 | 0.455 | 0.39 | 0.455 | 14 | 0.93 | 91/186 (48.92%) | 121/161 (75.16%) | 135/200 (67.50%) | [0.610, 0.735] | 161 | 14 | 11 | 24 | 7 | 0 | 0.041 | 97/200 (48.50%) | 0 | 103 | 66 | 29 | raw_token_ids | 50 | cpu |
| native-node-cpu/q4f16 | mmlu | 200 | 0.465 | 0.39 | 0.465 | 15 | 0.925 | 93/185 (50.27%) | 121/161 (75.16%) | 136/200 (68.00%) | [0.615, 0.740] | 161 | 15 | 10 | 25 | 8 | 0 | 0.0167 | 95/200 (47.50%) | 0 | 105 | 65 | 32 | raw_token_ids | 47 | cpu |

---

## Table 2: Direct Matched-Dtype Comparisons: Native CPU vs. Browser (RQ1)

*Evaluates runtime effects by matching model repository identifiers and dtypes between Native Node.js CPU and Browser WebGPU/WASM across all paired item evaluations. (Note: Repository IDs and dtypes were strictly matched; file SHA-256 hashes were not pinned across the Oct 8–9 and Oct 28 run dates).*

| browser_setup | native_ref_setup | task | n | browser_acc | native_acc | valid_agree | all_outcome_agree | exact_text | exact_token_ids | median_first_div_token | n_diverged | n_div_at_0 | n_div_at_1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| browser/tjs-wasm/q4/cpu-chrome-check | native-node-cpu/q4 | gsm8k | 10 | 0.4 | 0.4 | 10/10 (100.00%) | 10/10 (100.00%) | 10/10 (100.00%) | 10/10 (100.00%) |  | 0 | 0 | 0 |
| browser/tjs-webgpu/fp16/intel-chrome | native-node-cpu/fp16 | gsm8k | 100 | 0.0 | 0.31 | 0/56 (0.00%) | 0/100 (0.00%) | 0/100 (0.00%) | 0/100 (0.00%) | 0 | 100 | 87 | 4 |
| browser/tjs-webgpu/fp16/nvidia-chrome | native-node-cpu/fp16 | gsm8k | 100 | 0.01 | 0.31 | 0/51 (0.00%) | 0/100 (0.00%) | 0/100 (0.00%) | 0/100 (0.00%) | 0 | 100 | 82 | 8 |
| browser/tjs-webgpu/fp16/nvidia-chrome-repeat | native-node-cpu/fp16 | gsm8k | 100 | 0.01 | 0.31 | 0/51 (0.00%) | 0/100 (0.00%) | 0/100 (0.00%) | 0/100 (0.00%) | 0 | 100 | 82 | 8 |
| browser/tjs-webgpu/q4f16/intel-chrome | native-node-cpu/q4f16 | gsm8k | 100 | 0.0 | 0.24 | 0/4 (0.00%) | 0/100 (0.00%) | 0/100 (0.00%) | 0/100 (0.00%) | 0 | 100 | 100 | 0 |
| browser/tjs-webgpu/q4f16/nvidia-chrome | native-node-cpu/q4f16 | gsm8k | 100 | 0.0 | 0.24 | 1/5 (20.00%) | 1/100 (1.00%) | 0/100 (0.00%) | 0/100 (0.00%) | 0 | 100 | 100 | 0 |
| browser/tjs-wasm/q4/cpu-chrome-check | native-node-cpu/q4 | mmlu | 10 | 0.2 | 0.2 | 8/8 (100.00%) | 10/10 (100.00%) | 10/10 (100.00%) | 10/10 (100.00%) |  | 0 | 0 | 0 |
| browser/tjs-webgpu/fp16/intel-chrome | native-node-cpu/fp16 | mmlu | 200 | 0.055 | 0.38 | 13/43 (30.23%) | 42/200 (21.00%) | 0/200 (0.00%) | 0/200 (0.00%) | 0 | 200 | 196 | 3 |
| browser/tjs-webgpu/fp16/nvidia-chrome | native-node-cpu/fp16 | mmlu | 200 | 0.07 | 0.38 | 11/45 (24.44%) | 42/200 (21.00%) | 1/200 (0.50%) | 1/200 (0.50%) | 0 | 199 | 196 | 2 |
| browser/tjs-webgpu/fp16/nvidia-chrome-repeat | native-node-cpu/fp16 | mmlu | 200 | 0.07 | 0.38 | 11/45 (24.44%) | 42/200 (21.00%) | 1/200 (0.50%) | 1/200 (0.50%) | 0 | 199 | 196 | 2 |
| browser/tjs-webgpu/q4f16/intel-chrome | native-node-cpu/q4f16 | mmlu | 200 | 0.025 | 0.465 | 3/17 (17.65%) | 17/200 (8.50%) | 1/200 (0.50%) | 1/200 (0.50%) | 0 | 199 | 193 | 6 |
| browser/tjs-webgpu/q4f16/nvidia-chrome | native-node-cpu/q4f16 | mmlu | 200 | 0.02 | 0.465 | 1/16 (6.25%) | 16/200 (8.00%) | 0/200 (0.00%) | 0/200 (0.00%) | 0 | 200 | 193 | 7 |

*Browser WASM CPU Verification:* `browser/tjs-wasm/q4/cpu-chrome-check` achieves **10/10 (100.00%) exact text and 10/10 (100.00%) exact token-ID match** on GSM8K, and **10/10 (100.00%) exact text and 10/10 (100.00%) exact token-ID match** on MMLU (with 8/8 valid parsed agreement) against `native-node-cpu/q4` on the matched diagnostic subset.

---

## Table 3: Cross-Adapter Pairwise Consistency: Intel UHD vs. NVIDIA RTX 2050 (RQ2)

*Directly compares the Intel UHD Graphics integrated GPU run against the NVIDIA GeForce RTX 2050 discrete GPU run on the same laptop for each browser engine build.*

| build | task | n | intel_acc | nvidia_acc | valid_agree | all_outcome_agree | exact_text | n_both_valid | n_diff_predictions | n_diff_text | median_first_div_token | n_diverged | div_source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tjs-webgpu/fp16 | gsm8k | 100 | 0.0 | 0.01 | 24/42 (57.14%) | 59/100 (59.00%) | 27/100 (27.00%) | 42 | 41 | 73 | 2 | 73 | raw_token_ids |
| tjs-webgpu/q4f16 | gsm8k | 100 | 0.0 | 0.0 | 3/3 (100.00%) | 97/100 (97.00%) | 36/100 (36.00%) | 3 | 3 | 64 | 6 | 64 | raw_token_ids |
| webllm/q4f16_1 | gsm8k | 100 | 0.16 | 0.19 | 59/100 (59.00%) | 59/100 (59.00%) | 46/100 (46.00%) | 100 | 41 | 54 | 60.5 | 54 | retokenized_text |
| webllm/q4f32_1 | gsm8k | 100 | 0.18 | 0.18 | 100/100 (100.00%) | 100/100 (100.00%) | 100/100 (100.00%) | 100 | 0 | 0 |  | 0 | retokenized_text |
| tjs-webgpu/fp16 | mmlu | 200 | 0.055 | 0.07 | 37/41 (90.24%) | 170/200 (85.00%) | 128/200 (64.00%) | 41 | 30 | 72 | 1 | 72 | raw_token_ids |
| tjs-webgpu/q4f16 | mmlu | 200 | 0.025 | 0.02 | 11/12 (91.67%) | 189/200 (94.50%) | 140/200 (70.00%) | 12 | 11 | 60 | 1 | 60 | raw_token_ids |
| webllm/q4f16_1 | mmlu | 200 | 0.42 | 0.43 | 194/199 (97.49%) | 194/200 (97.00%) | 190/200 (95.00%) | 199 | 6 | 10 | 0 | 10 | retokenized_text |
| webllm/q4f32_1 | mmlu | 200 | 0.445 | 0.445 | 194/194 (100.00%) | 200/200 (100.00%) | 200/200 (100.00%) | 194 | 0 | 0 |  | 0 | retokenized_text |

*Key takeaway:* Under `webllm/q4f16_1` (where MLC `model_dtype` is `float16`), cross-GPU numerical drift produces divergent text on 10/200 (5.0%) MMLU items and 54/100 (54.0%) GSM8K items. Under `webllm/q4f32_1` (where MLC `model_dtype` is `float32`), cross-adapter consistency reaches **100.00% exact text agreement (200/200 MMLU and 100/100 GSM8K)** with 0 divergent predictions.

---

## Table 4: Parsing & Truncation Audit (Strict vs. Permissive Extraction)

*Evaluates regex extraction parse rates, accuracy restricted to valid parsed responses, strict marker extraction (`r'\b([ABCD])\b'` on MMLU, `r'answer is\s*([\-0-9,\.]+)'` on GSM8K), and token limit saturation (5 tokens for MMLU, 320 tokens for GSM8K).*

| setup | task | n | task_acc | n_unparsed | parse_rate | acc_parsed | strict_acc | n_at_token_cap | cap_source | gpu |
|---|---|---|---|---|---|---|---|---|---|---|
| browser/tjs-wasm/q4/cpu-chrome-check | gsm8k | 10 | 0.4 | 0 | 1.0 | 4/10 (40.00%) | 0/10 (0.00%) | 5 | raw_token_ids | cpu (wasm) |
| browser/tjs-webgpu/fp16/intel-chrome | gsm8k | 100 | 0.0 | 44 | 0.56 | 0/56 (0.00%) | 0/100 (0.00%) | 76 | raw_token_ids | intel |
| browser/tjs-webgpu/fp16/nvidia-chrome | gsm8k | 100 | 0.01 | 49 | 0.51 | 1/51 (1.96%) | 0/100 (0.00%) | 68 | raw_token_ids | nvidia |
| browser/tjs-webgpu/fp16/nvidia-chrome-repeat | gsm8k | 100 | 0.01 | 49 | 0.51 | 1/51 (1.96%) | 0/100 (0.00%) | 68 | raw_token_ids | nvidia |
| browser/tjs-webgpu/q4f16/intel-chrome | gsm8k | 100 | 0.0 | 96 | 0.04 | 0/4 (0.00%) | 0/100 (0.00%) | 89 | raw_token_ids | intel |
| browser/tjs-webgpu/q4f16/nvidia-chrome | gsm8k | 100 | 0.0 | 95 | 0.05 | 0/5 (0.00%) | 0/100 (0.00%) | 91 | raw_token_ids | nvidia |
| browser/webllm/q4f16_1/intel-chrome | gsm8k | 100 | 0.16 | 0 | 1.0 | 16/100 (16.00%) | 0/100 (0.00%) | 50 | retokenized_text | intel |
| browser/webllm/q4f16_1/nvidia-chrome | gsm8k | 100 | 0.19 | 0 | 1.0 | 19/100 (19.00%) | 0/100 (0.00%) | 49 | retokenized_text | nvidia |
| browser/webllm/q4f32_1/intel-chrome | gsm8k | 100 | 0.18 | 0 | 1.0 | 18/100 (18.00%) | 1/100 (1.00%) | 45 | retokenized_text | intel |
| browser/webllm/q4f32_1/nvidia-chrome | gsm8k | 100 | 0.18 | 0 | 1.0 | 18/100 (18.00%) | 1/100 (1.00%) | 45 | retokenized_text | nvidia |
| native-node-cpu/fp16 | gsm8k | 100 | 0.31 | 0 | 1.0 | 31/100 (31.00%) | 0/100 (0.00%) | 41 | raw_token_ids | cpu |
| native-node-cpu/fp32 | gsm8k | 100 | 0.29 | 0 | 1.0 | 29/100 (29.00%) | 0/100 (0.00%) | 44 | raw_token_ids | cpu |
| native-node-cpu/q4 | gsm8k | 100 | 0.2 | 0 | 1.0 | 20/100 (20.00%) | 1/100 (1.00%) | 50 | raw_token_ids | cpu |
| native-node-cpu/q4f16 | gsm8k | 100 | 0.24 | 0 | 1.0 | 24/100 (24.00%) | 1/100 (1.00%) | 50 | raw_token_ids | cpu |
| browser/tjs-wasm/q4/cpu-chrome-check | mmlu | 10 | 0.2 | 2 | 0.8 | 2/8 (25.00%) | 2/10 (20.00%) | 2 | raw_token_ids | cpu (wasm) |
| browser/tjs-webgpu/fp16/intel-chrome | mmlu | 200 | 0.055 | 146 | 0.27 | 11/54 (20.37%) | 7/200 (3.50%) | 180 | raw_token_ids | intel |
| browser/tjs-webgpu/fp16/nvidia-chrome | mmlu | 200 | 0.07 | 146 | 0.27 | 14/54 (25.93%) | 10/200 (5.00%) | 187 | raw_token_ids | nvidia |
| browser/tjs-webgpu/fp16/nvidia-chrome-repeat | mmlu | 200 | 0.07 | 146 | 0.27 | 14/54 (25.93%) | 10/200 (5.00%) | 187 | raw_token_ids | nvidia |
| browser/tjs-webgpu/q4f16/intel-chrome | mmlu | 200 | 0.025 | 182 | 0.09 | 5/18 (27.78%) | 1/200 (0.50%) | 155 | raw_token_ids | intel |
| browser/tjs-webgpu/q4f16/nvidia-chrome | mmlu | 200 | 0.02 | 184 | 0.08 | 4/16 (25.00%) | 1/200 (0.50%) | 160 | raw_token_ids | nvidia |
| browser/webllm/q4f16_1/intel-chrome | mmlu | 200 | 0.42 | 1 | 0.995 | 84/199 (42.21%) | 84/200 (42.00%) | 39 | retokenized_text | intel |
| browser/webllm/q4f16_1/nvidia-chrome | mmlu | 200 | 0.43 | 0 | 1.0 | 86/200 (43.00%) | 86/200 (43.00%) | 36 | retokenized_text | nvidia |
| browser/webllm/q4f32_1/intel-chrome | mmlu | 200 | 0.445 | 6 | 0.97 | 89/194 (45.88%) | 89/200 (44.50%) | 40 | retokenized_text | intel |
| browser/webllm/q4f32_1/nvidia-chrome | mmlu | 200 | 0.445 | 6 | 0.97 | 89/194 (45.88%) | 89/200 (44.50%) | 40 | retokenized_text | nvidia |
| native-node-cpu/fp16 | mmlu | 200 | 0.38 | 40 | 0.8 | 76/160 (47.50%) | 76/200 (38.00%) | 91 | raw_token_ids | cpu |
| native-node-cpu/fp32 | mmlu | 200 | 0.39 | 39 | 0.805 | 78/161 (48.45%) | 78/200 (39.00%) | 104 | raw_token_ids | cpu |
| native-node-cpu/q4 | mmlu | 200 | 0.455 | 14 | 0.93 | 91/186 (48.92%) | 91/200 (45.50%) | 50 | raw_token_ids | cpu |
| native-node-cpu/q4f16 | mmlu | 200 | 0.465 | 15 | 0.925 | 93/185 (50.27%) | 93/200 (46.50%) | 47 | raw_token_ids | cpu |

*Unparsed Response Findings:* In `native-node-cpu/fp32` on MMLU, 39/200 items failed letter-only extraction: 38 saturated the 5-token limit without emitting a standalone letter, and 1 terminated early at 3 tokens (`"WEP"` + EOS, ID `mmlu-1554`).

---

## Table 5: Holm Step-Down Multiple-Testing Correction (22 Hypotheses)

*Controls family-wise error rate across all 22 primary experimental comparisons against `native-node-cpu/fp32` (11 full runs $\times$ 2 tasks, excluding the reference run, repeatability check, and diagnostic 20-item subset).*

| setup | task | flips | mcnemar_p_unadjusted | holm_rank | mcnemar_p_holm | significant_at_0_05 |
|---|---|---|---|---|---|---|
| browser/tjs-webgpu/q4f16/nvidia-chrome | mmlu | c2w=78, w2c=4 | 7.614297e-19 | 1 | 1.675145e-17 | Yes |
| browser/tjs-webgpu/q4f16/intel-chrome | mmlu | c2w=77, w2c=4 | 1.449536e-18 | 2 | 3.044026e-17 | Yes |
| browser/tjs-webgpu/fp16/intel-chrome | mmlu | c2w=73, w2c=6 | 9.995152e-16 | 3 | 1.999030e-14 | Yes |
| browser/tjs-webgpu/fp16/nvidia-chrome | mmlu | c2w=72, w2c=8 | 5.375096e-14 | 4 | 1.021268e-12 | Yes |
| browser/tjs-webgpu/fp16/intel-chrome | gsm8k | c2w=29, w2c=0 | 3.725290e-09 | 5 | 6.705523e-08 | Yes |
| browser/tjs-webgpu/q4f16/intel-chrome | gsm8k | c2w=29, w2c=0 | 3.725290e-09 | 6 | 6.705523e-08 | Yes |
| browser/tjs-webgpu/q4f16/nvidia-chrome | gsm8k | c2w=29, w2c=0 | 3.725290e-09 | 7 | 6.705523e-08 | Yes |
| browser/tjs-webgpu/fp16/nvidia-chrome | gsm8k | c2w=29, w2c=1 | 5.774200e-08 | 8 | 8.661300e-07 | Yes |
| browser/webllm/q4f16_1/intel-chrome | gsm8k | c2w=19, w2c=6 | 0.0146 | 9 | 0.2049 | No |
| native-node-cpu/q4f16 | mmlu | c2w=10, w2c=25 | 0.0167 | 10 | 0.2168 | No |
| browser/webllm/q4f32_1/intel-chrome | gsm8k | c2w=16, w2c=5 | 0.0266 | 11 | 0.3192 | No |
| browser/webllm/q4f32_1/nvidia-chrome | gsm8k | c2w=16, w2c=5 | 0.0266 | 12 | 0.3192 | No |
| native-node-cpu/q4 | mmlu | c2w=11, w2c=24 | 0.0410 | 13 | 0.4096 | No |
| native-node-cpu/q4 | gsm8k | c2w=13, w2c=4 | 0.0490 | 14 | 0.4414 | No |
| browser/webllm/q4f16_1/nvidia-chrome | gsm8k | c2w=16, w2c=6 | 0.0525 | 15 | 0.4414 | No |
| browser/webllm/q4f32_1/intel-chrome | mmlu | c2w=19, w2c=30 | 0.1524 | 16 | 1.0000 | No |
| browser/webllm/q4f32_1/nvidia-chrome | mmlu | c2w=19, w2c=30 | 0.1524 | 17 | 1.0000 | No |
| native-node-cpu/q4f16 | gsm8k | c2w=10, w2c=5 | 0.3018 | 18 | 1.0000 | No |
| browser/webllm/q4f16_1/nvidia-chrome | mmlu | c2w=22, w2c=30 | 0.3317 | 19 | 1.0000 | No |
| browser/webllm/q4f16_1/intel-chrome | mmlu | c2w=23, w2c=29 | 0.4885 | 20 | 1.0000 | No |
| native-node-cpu/fp16 | mmlu | c2w=2, w2c=0 | 0.5000 | 21 | 1.0000 | No |
| native-node-cpu/fp16 | gsm8k | c2w=2, w2c=4 | 0.6875 | 22 | 1.0000 | No |
