> **Note**: This document reflects the initial study design drafted on 3 October 2026. Final experiments were consolidated onto a single dual-GPU Windows 11 testbed (Intel Core i5-13420H, NVIDIA RTX 2050 4 GB dGPU, and Intel UHD Gen-12LP iGPU) to hold host hardware and OS variables strictly constant; see results/hardware.md and results/summary.md for final executed configurations.

# Research Plan: Does Running a Small LLM in the Browser Change Its Answers?

Working title: "Same Model, Different Answers? Measuring Answer Fidelity of In-Browser LLM Inference Across Frameworks and GPU Vendors"

Authors: Sudip Niroula (first author) and Mandip Pokharel (second author)
Prepared: 3 October 2026
Target: preprint on TechRxiv by end of November 2026; code and data on Zenodo and GitHub

---

## 1. Novelty audit (what already exists)

I checked the closest work on 3 October 2026. Summary: browser LLM papers measure speed; quantization papers measure answer changes, but on servers, not in browsers. Nobody I found measures whether the same general-purpose LLM gives the same answers in the browser versus natively, across browser frameworks and GPU vendors.

| Work | What it measured | Gap left for us |
|---|---|---|
| WebLLM paper, 2024 ([arXiv 2412.15803](https://arxiv.org/html/2412.15803v2)) | Decoding speed of WebLLM versus native MLC-LLM on an Apple M3 Max | No accuracy or output comparison |
| WebGPU dispatch overhead study, 2026 ([arXiv 2604.02344](https://arxiv.org/html/2604.02344)) | Speed and latency across 4 GPU vendors, 3 browsers; WebLLM with Qwen2.5-0.5B/1.5B | No output-quality comparison |
| Llamas on the Web (LlamaWeb), May 2026 ([arXiv 2605.20706](https://arxiv.org/html/2605.20706v1)) | Memory and speed of llama.cpp's WebGPU backend across 16 devices; kernel-level numeric tests only | No end-to-end accuracy. Reports that f16 accumulation made Qwen2.5 produce incoherent output on Apple GPUs during development, which motivates our question |
| GenomeOcean Anywhere, 2026 ([arXiv 2609.35882](https://arxiv.org/html/2609.35882v1)) | Browser llama.cpp WebGPU matches native for a genome MoE model, single NVIDIA H100 | Not a general LLM, one framework, one GPU, no answer-level benchmark |
| Hints-In-Browser, NeurIPS 2024 ([paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/34cc2ded6daba59357134c0b9fb06bfe-Paper-Datasets_and_Benchmarks_Track.pdf)) | Feedback quality of 16-bit versus web-converted 4-bit models | Mixes quantization with browser effects; no comparison across browser frameworks or GPU vendors (authors list WebLLM vs ONNX as future work) |
| EdgeBench, undergraduate thesis, UFAM ([PDF](https://riu.ufam.edu.br/bitstream/prefix/10027/8/TCC_MatheusUchoa.pdf)) | WebLLM speed on Intel and NVIDIA; tiny 5-question MMLU screen | No paired accuracy comparison browser vs native |
| "Accuracy is Not All You Need" ([OpenReview](https://openreview.net/pdf/7ecf82af167db2fa1a4c2972d42f903ce9507ee9.pdf)) | "Flips": compressed models change individual answers even when total accuracy looks the same | Server-side compression only; we borrow the flip metric |
| "How Divergence Becomes Decision Flips", Oct 2026 ([arXiv 2610.00694](https://arxiv.org/html/2610.00694v1)) | Links KL divergence to decision flips in compressed models on H100 | Not browser or cross-vendor; we cite it as method background |

Honest caveat: this is a focused check, not a guarantee. Re-run a search one week before posting, because this area is moving fast.

## 2. Research questions

- RQ1 (runtime effect): With the exact same weight file, does running in the browser (WebGPU) change answers compared with native inference?
- RQ2 (vendor effect): Does the same browser setup give different answers on an NVIDIA GPU versus an AMD integrated GPU?
- RQ3 (deployment effect): How far are the ready-made browser model builds (for example WebLLM's q4f16 builds) from the full-precision reference, and how much of that is quantization versus the runtime?
- RQ4 (visibility): Do overall accuracy numbers hide answer flips? (Same accuracy, but different questions right and wrong.)

## 3. Why the design is clean: hold the weights fixed

The main trap is mixing up two causes: model compression (quantization) and the runtime/browser. Design comparisons in pairs where only one thing changes:

| Comparison | Same | Different | Answers |
|---|---|---|---|
| A. llama.cpp native (CUDA / CPU) vs llama.cpp WebGPU in Chrome (same GGUF file) | Weights, engine source | Runtime (native vs browser) | RQ1 |
| B. ONNX Runtime native (Python) vs Transformers.js WebGPU vs Transformers.js WASM (same ONNX file) | Weights | Runtime | RQ1 |
| C. Same browser setup on GTX 1650 Ti vs Radeon iGPU (vs friend's RTX 4060) | Weights, runtime | GPU vendor / generation | RQ2 |
| D. Full-precision reference (PyTorch, fp32 on CPU) vs each quantized build | Model | Quantization + runtime | RQ3 |
| E. Chrome vs Edge on same GPU | Everything except browser | Browser | Minor check |

Note: Chrome and Edge both use Chromium's WebGPU (Dawn), so E is expected to show little difference. Firefox has WebGPU too; include it if it runs.

## 4. Models (small enough for a 4 GB GPU)

Pick 3 models available in all frameworks. Verify availability during setup week:

- Qwen2.5-0.5B-Instruct
- Qwen2.5-1.5B-Instruct (or a newer Qwen small model if available in all three formats)
- Llama-3.2-1B-Instruct
- Optional: Gemma-3-1B-it

Formats needed for each: original Hugging Face weights (reference), a GGUF build (Q4_K_M and Q8_0), an ONNX build (fp16 and q4), and a WebLLM prebuilt (q4f16_1).

## 5. Tasks and data

| Benchmark | Size | Why |
|---|---|---|
| MMLU, stratified sample | 500 questions | Multiple choice; scoring by letter; standard |
| GSM8K test sample | 300 problems | Math; long generation; errors accumulate |
| ARC-Challenge (optional) | 300 questions | Second multiple-choice test |

Fixed settings: greedy decoding (temperature 0), identical prompt template and chat template, fixed maximum length (for example 512 tokens for GSM8K), fixed random seed, same answer extraction code for every setup.

## 6. Metrics

1. Accuracy per setup.
2. Agreement with reference: % of questions where the final answer equals the reference answer.
3. Flip rate: % of questions that go from correct to wrong, and from wrong to correct.
4. Exact output match: % of generations that are character-identical to the native run with the same weights.
5. First divergence position: the token index where outputs first differ (median, distribution).
6. Optional: top-1 next-token agreement on fixed text (teacher forcing) where logits are accessible (llama.cpp gives logits).

Statistics: McNemar's test for paired accuracy differences; bootstrap 95% confidence intervals for agreement and flip rates. Report run-to-run repeatability too: run each setup twice to see whether the same setup is even deterministic.

## 7. Expected results (any outcome is publishable)

- If browser equals native with the same weights: "safe to deploy; differences come from quantization, not the browser." Useful reassurance with evidence.
- If browser differs: "the browser itself changes answers, especially on some GPUs." Useful warning.
- Most likely: small but real differences in long math outputs, near-zero differences in multiple choice. That nuance is itself the finding.

## 8. Timeline (around IELTS and applications)

| Dates | Work |
|---|---|
| 4–13 Oct | IELTS only. Optional: install Python, llama.cpp, Node. |
| 14–25 Oct | Setup: download models in all formats; build a small browser test page; confirm WebGPU works on both GPUs (chrome://gpu); pilot with 20 questions. |
| 26 Oct–8 Nov | Full runs: reference on CPU; native runs; browser runs on both GPUs. Friend runs the same scripts on the RTX 4060. |
| 9–15 Nov | Analysis: tables, flip charts, statistics. |
| 16–28 Nov | Writing (6–8 pages, IEEE two-column format). Teacher reviews. |
| By 30 Nov | Post on TechRxiv; code and data on GitHub + Zenodo (DOI). |
| Early Dec | Cite it in EMAI application. |

Laptop notes: GTX 1650 Ti has 4 GB VRAM, so keep models at 1.5B or below in GPU runs. The fp32 reference runs on CPU (Ryzen 5 4600H, 24 GB RAM); a 1.5B model in fp32 needs about 6 GB RAM. Run long jobs overnight on mains power with the laptop cooled.

## 9. Collaborator & Co-Authorship

Mandip Pokharel is the collaborator and co-author (second author) who executed the full benchmark suite on the dual-GPU (NVIDIA RTX 2050 + Intel UHD) testbed and curated the result logs. Roles are structured following the CRediT taxonomy:

- **Sudip Niroula (First Author)**: Study design, test harness architecture (`site/` and `run_native.mjs`), analysis pipeline (`analyze.py`), statistical evaluations, and primary drafting.
- **Mandip Pokharel (Second Author)**: Executed the full benchmark suite across all runtime configurations on the dual-GPU (NVIDIA RTX 2050 + Intel UHD) testbed, curated the result logs in `results/`, and contributed to validation and draft review.

Responsibilities:

| Sudip Niroula | Mandip Pokharel |
|---|---|
| Design, test harness, analysis scripts | Benchmark execution across native CPU and browser runs (WASM, WebGPU, WebLLM) on the dual-GPU testbed |
| Statistical analysis and evaluation | Curating result logs in `results/` and verifying run configurations |
| Writing original draft and documentation | Output validation, reviewing, and editing drafts |

## 10. Risks and fixes

| Risk | Fix |
|---|---|
| A model is not available in one format | Swap to another model available in all formats; document conversion steps |
| WebGPU fails on the Radeon iGPU or a driver issue | Update drivers; record the failure (it is itself a finding); continue with remaining GPUs |
| Browser runs are slow | Use 0.5B and 1.5B only; run overnight; reduce GSM8K to 200 |
| Someone publishes the same study first | Re-check literature before posting; our multi-framework + AMD/NVIDIA + flip-rate angle can still differ |
| Results show "no difference" | Still publishable as a careful null result with confidence intervals |

## 11. What to read first (2–3 hours total)

1. WebLLM paper: https://arxiv.org/html/2412.15803v2
2. Llamas on the Web (LlamaWeb): https://arxiv.org/html/2605.20706v1
3. Accuracy is Not All You Need (flips): https://openreview.net/pdf/7ecf82af167db2fa1a4c2972d42f903ce9507ee9.pdf
4. Transformers.js WebGPU guide: https://huggingface.co/docs/transformers.js/en/guides/webgpu
5. TechRxiv submission info: https://innovate.ieee.org/techrxiv/
