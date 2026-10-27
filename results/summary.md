Reference: native-node-cpu/fp32
Hardware: Laptop with Intel Core i5-13th Gen H processor (i5-13420H), NVIDIA GeForce RTX 2050 (4 GB VRAM), and Intel UHD Graphics

| setup | task | n | acc | ref_acc | agree | agree_ci | flips_c2w | flips_w2c | mcnemar_p | exact_text | median_first_div_token | n_diverged | gpu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| browser/tjs-wasm/q4/cpu-chrome-check | gsm8k | 10 | 0.4 | 0.4 | 0.2 | [0.000, 0.500] | 2 | 2 | 1.0 | 0.0 | 0 | 10 | intel |
| browser/tjs-webgpu/fp16/intel-chrome | gsm8k | 100 | 0.0 | 0.29 | 0.03 | [0.000, 0.070] | 29 | 0 | 0.0 | 0.0 | 0 | 100 | intel |
| browser/tjs-webgpu/fp16/nvidia-chrome | gsm8k | 100 | 0.01 | 0.29 | 0.02 | [0.000, 0.050] | 29 | 1 | 0.0 | 0.0 | 0 | 100 | nvidia |
| browser/tjs-webgpu/fp16/nvidia-chrome-repeat | gsm8k | 100 | 0.01 | 0.29 | 0.02 | [0.000, 0.050] | 29 | 1 | 0.0 | 0.0 | 0 | 100 | nvidia |
| browser/tjs-webgpu/q4f16/intel-chrome | gsm8k | 100 | 0.0 | 0.29 | 0.0 | [0.000, 0.000] | 29 | 0 | 0.0 | 0.0 | 0 | 100 | intel |
| browser/tjs-webgpu/q4f16/nvidia-chrome | gsm8k | 100 | 0.0 | 0.29 | 0.0 | [0.000, 0.000] | 29 | 0 | 0.0 | 0.0 | 0 | 100 | nvidia |
| browser/webllm/q4f16_1/intel-chrome | gsm8k | 100 | 0.16 | 0.29 | 0.15 | [0.080, 0.220] | 19 | 6 | 0.0146 | 0.0 | 1 | 100 | intel |
| browser/webllm/q4f16_1/nvidia-chrome | gsm8k | 100 | 0.19 | 0.29 | 0.19 | [0.120, 0.270] | 16 | 6 | 0.0525 | 0.0 | 1 | 100 | nvidia |
| browser/webllm/q4f32_1/intel-chrome | gsm8k | 100 | 0.18 | 0.29 | 0.2 | [0.120, 0.290] | 16 | 5 | 0.0266 | 0.0 | 1 | 100 | intel |
| browser/webllm/q4f32_1/nvidia-chrome | gsm8k | 100 | 0.18 | 0.29 | 0.2 | [0.120, 0.290] | 16 | 5 | 0.0266 | 0.0 | 1 | 100 | nvidia |
| native-node-cpu/fp32 | gsm8k | 100 | 0.29 | 0.29 | 1.0 | [1.000, 1.000] | 0 | 0 | 1.0 | 1.0 |  | 0 | cpu |
| browser/tjs-wasm/q4/cpu-chrome-check | mmlu | 10 | 0.2 | 0.3 | 0.4 | [0.100, 0.700] | 1 | 0 | 1.0 | 0.3 | 0 | 7 | intel |
| browser/tjs-webgpu/fp16/intel-chrome | mmlu | 200 | 0.055 | 0.39 | 0.205 | [0.155, 0.260] | 73 | 6 | 0.0 | 0.0 | 0 | 200 | intel |
| browser/tjs-webgpu/fp16/nvidia-chrome | mmlu | 200 | 0.07 | 0.39 | 0.21 | [0.155, 0.270] | 72 | 8 | 0.0 | 0.005 | 0 | 199 | nvidia |
| browser/tjs-webgpu/fp16/nvidia-chrome-repeat | mmlu | 200 | 0.07 | 0.39 | 0.21 | [0.155, 0.270] | 72 | 8 | 0.0 | 0.005 | 0 | 199 | nvidia |
| browser/tjs-webgpu/q4f16/intel-chrome | mmlu | 200 | 0.025 | 0.39 | 0.2 | [0.150, 0.260] | 77 | 4 | 0.0 | 0.01 | 0 | 198 | intel |
| browser/tjs-webgpu/q4f16/nvidia-chrome | mmlu | 200 | 0.02 | 0.39 | 0.19 | [0.140, 0.250] | 78 | 4 | 0.0 | 0.0 | 0 | 200 | nvidia |
| browser/webllm/q4f16_1/intel-chrome | mmlu | 200 | 0.42 | 0.39 | 0.5 | [0.430, 0.570] | 23 | 29 | 0.4885 | 0.35 | 0 | 130 | intel |
| browser/webllm/q4f16_1/nvidia-chrome | mmlu | 200 | 0.43 | 0.39 | 0.5 | [0.430, 0.570] | 22 | 30 | 0.3317 | 0.345 | 0 | 131 | nvidia |
| browser/webllm/q4f32_1/intel-chrome | mmlu | 200 | 0.445 | 0.39 | 0.55 | [0.480, 0.620] | 19 | 30 | 0.1524 | 0.37 | 0 | 126 | intel |
| browser/webllm/q4f32_1/nvidia-chrome | mmlu | 200 | 0.445 | 0.39 | 0.55 | [0.480, 0.620] | 19 | 30 | 0.1524 | 0.37 | 0 | 126 | nvidia |
| native-node-cpu/fp32 | mmlu | 200 | 0.39 | 0.39 | 1.0 | [1.000, 1.000] | 0 | 0 | 1.0 | 1.0 |  | 0 | cpu |
