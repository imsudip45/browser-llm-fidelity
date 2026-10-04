# Browser LLM Fidelity

Does running a small language model inside a web browser change its answers?

This repository contains the full experiment harness for a short empirical study comparing the answers of the same small LLM (Qwen2.5-0.5B-Instruct) when run:

- natively (Transformers.js on Node.js with onnxruntime-node, CPU), and
- in the browser (Transformers.js with WebGPU or WASM, and WebLLM with WebGPU),
- across GPU vendors (NVIDIA GTX 1650 Ti, AMD Radeon integrated, NVIDIA RTX 4060).

Status: work in progress (October 2026). Results are not final.

## Research questions

1. With the same weight file, does running in the browser change answers compared with native inference?
2. Does the same browser setup give different answers on NVIDIA versus AMD GPUs?
3. How far are ready-made browser builds (for example WebLLM q4f16_1) from a full-precision reference?
4. Do overall accuracy numbers hide answer flips?

See [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) for the full plan and the related-work check.

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

Sudip Niroula (lead). Co-authors to be added with agreed contribution roles (CRediT).

## License

MIT
