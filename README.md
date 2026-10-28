# Browser LLM Fidelity

Does running a small language model inside a web browser change its answers?

This repository contains the full experiment harness for a short empirical study comparing the answers of the same small LLM (Qwen2.5-0.5B-Instruct) when run:

- natively (Transformers.js on Node.js with onnxruntime-node, CPU), and
- in the browser (Transformers.js with WebGPU or WASM, and WebLLM with WebGPU),
- across GPU vendors on a single unified host machine (NVIDIA GeForce RTX 2050 4 GB discrete GPU vs. Intel UHD Gen-12LP integrated GPU).

Status: work in progress (October 2026). Results are not final.

## Research questions

1. With the same weight file, does running in the browser change answers compared with native inference?
2. Does the same browser setup give different answers on NVIDIA discrete versus Intel integrated GPUs?
3. How far are ready-made browser builds (for example WebLLM q4f16_1) from a full-precision reference?
4. Do overall accuracy numbers hide answer flips?

For a detailed plain-language explanation of the study, read [docs/ABOUT_THE_RESEARCH.md](docs/ABOUT_THE_RESEARCH.md). The working plan and related-work check are in [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md).

## Repository layout

| Path | What it is |
|---|---|
| `site/` | Browser runner page (`index.html`, `app.js`, `worker.js`) |
| `site/harness.js` | Shared prompts and answer extraction, used by both native and browser runs |
| `site/questions_v1.json` | Fixed question set: 200 MMLU + 100 GSM8K, seed 2026 |
| `run_native.mjs` | Native runner (Node.js, CPU) |
| `runall.sh` | Runs the native reference (fp32) and native q4 / fp16 |
| `analyze.py` | Compares all result files with the reference; writes `results/summary.md` and `.csv` |
| `scripts/make_questions.py` | Rebuilds the question set from the original datasets |
| `results/` | Result JSON files (one per setup) |
| `RUN_GUIDE.md` | Step-by-step guide for running the browser experiments |

## Quick start

Native reference:

```
npm install
node run_native.mjs onnx-community/Qwen2.5-0.5B-Instruct fp32 all
```

Browser runs:

```
python -m http.server 8000 --directory site
```

Open http://localhost:8000 in Chrome and follow [RUN_GUIDE.md](RUN_GUIDE.md).

Analysis:

```
pip install scipy
python analyze.py results native-node-cpu/fp32
```

## Preliminary Results Summary

Reference: `native-node-cpu/fp32`  
Hardware: Laptop with Intel Core i5-13420H, NVIDIA GeForce RTX 2050 (4 GB VRAM), and Intel UHD Graphics (Windows 11, Chrome 139)

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

### Key Takeaways

1. **`fp32` In-Browser Memory Ceiling:** Full unquantized `fp32` models cannot allocate within consumer browser WebGPU environments due to heap/buffer limits (`std::bad_alloc`, error code 6) on both 4 GB dGPU and shared-memory iGPU targets, establishing quantization as a mandatory prerequisite for browser deployment.
2. **Hidden Answer Flips in WebLLM (24.5%–26.0% churn on MMLU):** While WebLLM aggregate accuracy matches or slightly exceeds the native reference (`0.42–0.445` vs. `0.39`), this parity masks widespread bilateral answer flips. Between 49 and 52 out of 200 questions flipped (19–23 correct-to-wrong, 29–30 wrong-to-correct), demonstrating that overall benchmark accuracy hides significant semantic churn.
3. **`q4f32_1` Eliminates Intel vs. NVIDIA Cross-GPU Drift:** In `q4f16_1`, slight inter-vendor divergence emerged between Intel UHD and NVIDIA RTX 2050 on both MMLU (`0.42` vs. `0.43`) and GSM8K (`0.16` vs. `0.19`). Upgrading intermediate accumulation precision to `q4f32_1` completely eliminated this cross-GPU divergence, producing identical accuracy (`0.18` on GSM8K, `0.445` on MMLU) and identical flip profiles across vendors.
4. **Transformers.js WebGPU vs. WASM Contrast:** While Transformers.js 4.3.0 WebGPU collapsed deterministically across both GPUs (`0.00–0.01` on GSM8K, diverging at token 0), running the exact same model weights and test harness through the browser WASM CPU backend (`tjs-wasm/q4`) scored `0.40` on GSM8K—matching native `fp32` CPU performance on the same subset. This confirms that the test harness and ONNX weights are intact, isolating the failure specifically to the browser WebGPU execution pipeline.

## Fixed settings

- Greedy decoding (no sampling), identical system prompt and chat template.
- MMLU: answer with a single letter, max 5 new tokens.
- GSM8K: step-by-step, final "The answer is N.", max 320 new tokens.
- Transformers.js 4.3.0 (native and browser), WebLLM 0.2.85.

## Data sources

- MMLU: https://huggingface.co/datasets/cais/mmlu
- GSM8K: https://huggingface.co/datasets/openai/gsm8k
- Model (ONNX): https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct

## Authors

- **Sudip Niroula** (Lead / First Author): Conceptualization, Methodology, Software (experiment harness and analysis pipeline), Formal Analysis, Writing – Original Draft, Documentation.
- **Mandip Pokharel** (Co-Author): Investigation (native and dual-GPU browser benchmark execution on the Intel Core i5-13420H / NVIDIA RTX 2050 / Intel UHD testbed), Data Curation, Validation.

## License

MIT
