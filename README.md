# Does Running a Small LLM in the Browser Change Its Answers? Comparing Native and WebGPU Inference Across GPUs

**Authors:** Sudip Niroula (Lead / First Author) and Mandip Pokharel (Co-Author)  
**Status:** Completed benchmark dataset and reproducible test harness (October 2026)  
**Model Tested:** `Qwen2.5-0.5B-Instruct`  
**Setups Compared:** Native Node.js on CPU (`fp32`, `fp16`, `q4`, `q4f16`) vs. Web Browser (`Transformers.js v4.3.0` WASM & WebGPU, and `WebLLM v0.2.85` WebGPU)  
**Test Hardware:** Single Windows 11 Laptop (13th Gen Intel Core i5-13420H CPU, NVIDIA GeForce RTX 2050 4 GB discrete GPU, and Intel UHD Gen-12LP integrated GPU, 8 GB RAM)  

---

## Summary

Most studies on browser AI only measure how fast a model runs. Developers usually assume that loading the same model in a web browser gives the same answers as running it normally on a computer. To test this assumption, we asked `Qwen2.5-0.5B-Instruct` the exact same 300 questions (200 MMLU multiple-choice questions and 100 GSM8K math problems) with randomness turned off (greedy decoding). Using one shared prompt and scoring script (`site/harness.js`) across 28 setups on the same laptop, we compared native CPU runs against browser runs across two GPUs (NVIDIA RTX 2050 and Intel UHD).

## What We Found

1. **Same weight file, broken browser WebGPU answers (Transformers.js):** When we run the `fp16` and `q4f16` ONNX model files natively on the CPU, they work normally (`fp16` matches the full-precision `fp32` baseline on 98% of MMLU questions and scores 31% on GSM8K math). However, loading those exact same `fp16` and `q4f16` files into Transformers.js WebGPU in Chrome causes the model's math answers to collapse to 0%–1% accuracy, changing outputs from the very first token (`token 0`). Running a 20-question check in browser CPU WASM (`q4`) restores normal GSM8K accuracy (40% vs. 40% reference), proving the bug is inside the WebGPU execution path rather than the weights or prompt script.
2. **Overall accuracy hides changed answers ("answer flips"):** On MMLU multiple-choice questions, 4-bit models look just as accurate overall as the full-precision `fp32` reference (scoring 42.0%–46.5% vs. 39.0%). However, checking individual questions shows that native 4-bit compression alone flips 17.5% of the answers (from right to wrong or wrong to right, agreeing with `fp32` on 68% of questions), and running in the browser with WebLLM flips 24.5%–26.0% of the answers (agreeing with `fp32` on only 50%–55% of questions).
3. **32-bit math fixes differences between Intel and NVIDIA GPUs:** In WebLLM, using 16-bit math (`q4f16_1`) causes the Intel integrated GPU and NVIDIA discrete GPU to give different answers on the same questions (16% vs. 19% on GSM8K). Switching to 32-bit math (`q4f32_1`) makes the Intel and NVIDIA GPUs give 100% identical accuracy (18.0% on GSM8K, 44.5% on MMLU) and identical answer flip counts.
4. **Full-precision (`fp32`) does not fit in browser WebGPU memory:** Running even a tiny 0.5B model in full 32-bit precision inside browser WebGPU crashes with an out-of-memory error (`std::bad_alloc`) on a 4 GB NVIDIA GPU and 8 GB system RAM, showing why browser apps are forced to use 16-bit or 4-bit compression.

---

## Research Questions

1. **RQ1 (Runtime effect):** With the exact same weight file, does running in the browser change answers compared with native inference?
2. **RQ2 (GPU vendor effect):** Does the same browser setup give different answers on an NVIDIA discrete GPU versus an Intel integrated GPU within the same machine?
3. **RQ3 (Deployment effect):** How far are ready-made browser builds (such as WebLLM `q4f16_1` and `q4f32_1`) from a full-precision native reference, and how much of that gap comes from compression versus the browser?
4. **RQ4 (Hidden answer flips):** Do overall accuracy numbers hide individual questions flipping between right and wrong?

For a detailed plain-language guide to the study design, read [docs/ABOUT_THE_RESEARCH.md](docs/ABOUT_THE_RESEARCH.md). The initial literature check and planning notes are in [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md), and hardware/diagnostic logs are in [results/hardware.md](results/hardware.md) and [results/notes.md](results/notes.md).

---

## Full Benchmark Results (28 Setups)

Reference: `native-node-cpu/fp32`  
Hardware: Single Windows 11 Laptop — Intel Core i5-13420H, NVIDIA GeForce RTX 2050 (4 GB VRAM), and Intel UHD Graphics (8 GB System RAM)

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
| native-node-cpu/fp16 | gsm8k | 100 | 0.31 | 0.29 | 0.74 | [0.650, 0.820] | 2 | 4 | 0.6875 | 0.54 | 88 | 46 | cpu |
| native-node-cpu/fp32 | gsm8k | 100 | 0.29 | 0.29 | 1.0 | [1.000, 1.000] | 0 | 0 | 1.0 | 1.0 |  | 0 | cpu |
| native-node-cpu/q4 | gsm8k | 100 | 0.2 | 0.29 | 0.19 | [0.120, 0.270] | 13 | 4 | 0.049 | 0.0 | 0 | 100 | cpu |
| native-node-cpu/q4f16 | gsm8k | 100 | 0.24 | 0.29 | 0.23 | [0.150, 0.320] | 10 | 5 | 0.3018 | 0.0 | 0 | 100 | cpu |
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
| native-node-cpu/fp16 | mmlu | 200 | 0.38 | 0.39 | 0.98 | [0.960, 0.995] | 2 | 0 | 0.5 | 0.91 | 1 | 18 | cpu |
| native-node-cpu/fp32 | mmlu | 200 | 0.39 | 0.39 | 1.0 | [1.000, 1.000] | 0 | 0 | 1.0 | 1.0 |  | 0 | cpu |
| native-node-cpu/q4 | mmlu | 200 | 0.455 | 0.39 | 0.675 | [0.610, 0.735] | 11 | 24 | 0.041 | 0.485 | 0 | 103 | cpu |
| native-node-cpu/q4f16 | mmlu | 200 | 0.465 | 0.39 | 0.68 | [0.615, 0.740] | 10 | 25 | 0.0167 | 0.475 | 0 | 105 | cpu |

---

## Repository Layout

| Path | What it is |
|---|---|
| `site/` | Browser runner page (`index.html`, `app.js`, `worker.js`) |
| `site/harness.js` | Shared prompts and answer extraction used by both native and browser runs |
| `site/questions_v1.json` | Fixed question set: 200 MMLU + 100 GSM8K, seed 2026 (also mirrored in `data/questions_v1.json`) |
| `run_native.mjs` | Native runner (Node.js, CPU via `onnxruntime-node`) |
| `runall.sh` | Runs all four native CPU baselines (`fp32`, `fp16`, `q4`, `q4f16`) |
| `analyze.py` | Compares all result files against the reference; writes `results/summary.md` and `results/summary.csv` |
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

- **Decoding:** Greedy decoding (temperature = 0, no sampling), identical system prompt and chat template across all setups.
- **MMLU (200 questions):** Stratified across all 57 subjects; answer with a single letter (A–D), max 5 new tokens.
- **GSM8K (100 problems):** Step-by-step math reasoning ending with "The answer is N.", max 320 new tokens.
- **Library Versions:** Transformers.js 4.3.0 (both native Node.js and browser), WebLLM 0.2.85, Chrome 139.0.0.0.

## Data & Model Sources

- **MMLU:** https://huggingface.co/datasets/cais/mmlu
- **GSM8K:** https://huggingface.co/datasets/openai/gsm8k
- **ONNX Weights:** https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct

## Authors & Contributions (CRediT)

- **Sudip Niroula** (Lead / First Author): Conceptualization, Methodology, Software (experiment harness and analysis pipeline), Formal Analysis, Writing – Original Draft, Documentation.
- **Mandip Pokharel** (Co-Author): Investigation (native and dual-GPU browser benchmark execution on the Intel Core i5-13420H / NVIDIA RTX 2050 / Intel UHD testbed), Data Curation, Validation.

## License

MIT
