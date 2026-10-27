# About This Research

Working title: "Same Model, Different Answers? Measuring Answer Fidelity of In-Browser LLM Inference Across Frameworks and GPU Vendors"

Authors: Sudip Niroula (lead author), Mandip Pokharel (co-author)
Status: in progress (October 2026)
Planned output: preprint on TechRxiv, with code and data in this repository (and an archived copy on Zenodo)

This document explains, in plain language, what the study is about, why it matters, how it is designed, and what we will and will not claim. It is written so that a co-author, a reviewer, or an admissions committee can understand the project without reading the code.

---

## 1. The one-sentence summary

We take the same small AI language model, ask it the same 300 questions in exactly the same way, and check whether its answers change when it runs inside a web browser instead of running normally on the computer, and whether they change between different GPUs.

## 2. Why this question matters

### 2.1 AI is moving into the browser

Small language models (around 0.5 to 3 billion parameters) can now run directly inside a web page, using the computer's own graphics card through a browser technology called WebGPU. Libraries such as WebLLM and Transformers.js make this practical. The appeal is clear:

- Privacy: the user's text never leaves their device.
- Cost: no server and no paid API.
- Offline use: it works without internet once the model is downloaded.

The WebLLM authors report that in-browser inference can retain up to about 80% of native decoding speed on the same device ([WebLLM paper](https://arxiv.org/html/2412.15803v2)).

### 2.2 But everyone measures speed, not correctness

Research on browser LLMs has focused on how fast they run. For example:

- The WebLLM paper evaluates decoding throughput against native MLC-LLM ([WebLLM paper](https://arxiv.org/html/2412.15803v2)).
- A 2026 study characterises WebGPU overhead for LLM inference across four GPU vendors, three backends, and three browsers, measuring speed and latency ([WebGPU dispatch overhead study](https://arxiv.org/html/2604.02344)).
- Llamas on the Web (LlamaWeb), a WebGPU backend for llama.cpp, reports memory and performance across many devices, with kernel-level numerical tests ([LlamaWeb](https://arxiv.org/html/2605.20706v1)).

What these papers do not measure is whether the model gives the same answers in the browser as it does natively. Developers implicitly assume "same model file means same behaviour". That assumption has not been tested carefully for general-purpose language models.

### 2.3 There are reasons to suspect differences

- Computers do arithmetic with limited precision. Adding the same numbers in a different order can give very slightly different results. Different software (native versus browser) and different GPUs often add numbers in different orders.
- Browser GPU code is written in a different shading language (WGSL) and goes through extra translation layers (for example Chrome's Dawn, then Direct3D 12 or Vulkan).
- Some browser kernels use 16-bit floating point to save memory and time, which is less precise. The LlamaWeb authors report that using 16-bit accumulation in one kernel caused some models, including Qwen2.5, to produce incoherent output on Apple M-series GPUs during development, so they switched that kernel to 32-bit ([LlamaWeb](https://arxiv.org/html/2605.20706v1)).
- Even on servers, running the same prompt with greedy decoding on different GPUs or batch sizes can change answers; one study reports accuracy swings of up to about 9% for a reasoning model under BF16 ([Numerical Sources of Nondeterminism in LLM Inference](https://zirui-ray-liu.github.io/projects/mlsys-numerics/numerical-nondeterminism/)).

A language model generates text one token at a time. If one early token changes, every following token can change. So a tiny numerical difference can, in principle, turn a correct answer into a wrong one.

### 2.4 Why the result is useful either way

- If browser answers match native answers, developers get evidence that in-browser deployment is safe in terms of correctness, and that any quality loss comes from model compression rather than the browser.
- If they differ, developers and researchers learn that the browser and GPU are hidden variables that must be reported, tested, and controlled, especially in sensitive uses such as education or health.

## 3. Key ideas explained simply

| Term | Meaning in this project |
|---|---|
| Language model (LLM) | A program that predicts the next piece of text (token). We use Qwen2.5-0.5B-Instruct, a small open model. |
| Inference | Running a trained model to get an answer (no training happens in this project). |
| Native inference | Running the model with normal software on the computer, here Transformers.js on Node.js with onnxruntime-node on the CPU. |
| In-browser inference | Running the model inside a web page, here using Transformers.js (onnxruntime-web) or WebLLM. |
| WebGPU | A browser standard that lets web pages use the GPU for computation. |
| WASM (WebAssembly) | A way to run fast code in the browser on the CPU, without the GPU. |
| Precision / quantization | How many bits are used to store each model number. fp32 = 32-bit (reference), fp16 = 16-bit, q4 / q4f16 = about 4-bit compressed weights. Fewer bits means smaller and faster, but possibly less accurate. |
| ONNX | A standard model file format. Transformers.js uses ONNX files, so the native and browser runs can load the exact same file. |
| WebLLM build (q4f16_1, q4f32_1) | Pre-compiled 4-bit versions of the model made by the WebLLM / MLC project for browsers. |
| Greedy decoding | Always choosing the single most likely next token. This removes randomness, so any difference between setups comes from computation, not chance. |
| Flip | A question where one setup is correct and another is wrong (or the other way round). |

## 4. Research questions

- RQ1, runtime effect: With exactly the same model file and code, does running in the browser change answers compared with native inference?
- RQ2, GPU vendor effect: Does the same browser setup give different answers on an NVIDIA discrete GPU (GeForce RTX 2050) versus an Intel integrated GPU (UHD Graphics Gen-12LP) within the same machine?
- RQ3, deployment effect: How far are ready-made browser model builds (for example WebLLM's 4-bit builds) from a full-precision reference, and how much of that gap is explained by compression rather than by the browser?
- RQ4, hidden changes: Do overall accuracy numbers hide answer flips? Two setups can have the same accuracy while getting different questions right.

## 5. How the study is designed

### 5.1 The main principle: change only one thing at a time

The most common mistake in comparisons like this is to change several things at once (for example, a different compression level and a different runtime) and then not know which one caused a difference. We avoid this by building comparisons in pairs:

| Comparison | Kept the same | Changed | Answers |
|---|---|---|---|
| Native Node.js (CPU) fp32 vs browser WebGPU fp32 | Model file, library version, prompts | Runtime and hardware path | RQ1 |
| Native Node.js q4 vs browser WASM q4 | Model file, library version, prompts | Runtime (native CPU vs browser CPU) | RQ1 |
| Browser WebGPU on NVIDIA vs on Intel | Everything except the GPU | GPU vendor | RQ2 |
| Browser fp16 run repeated twice on the same GPU | Everything | Nothing (repeatability check) | Baseline noise |
| Full-precision reference vs each compressed build (fp16, q4f16, WebLLM q4f16_1, q4f32_1) | Model, prompts | Compression and runtime | RQ3 |
| Chrome vs Edge on the same GPU (optional) | Everything except the browser | Browser | Minor check |

### 5.2 Fixed and shared components

- One shared file (`site/harness.js`) builds every prompt and extracts every answer. The native runner and the browser runner both import it, so prompts and scoring are identical.
- The same Transformers.js version (4.3.0) is used natively and in the browser.
- An explicit system prompt is included so that every runtime receives the same conversation.
- All runs use greedy decoding and fixed maximum output lengths.
- Each results file records the exact setup: library versions, model, precision, browser user agent, and the GPU reported by WebGPU.

### 5.3 Model

Qwen2.5-0.5B-Instruct, using the ONNX files published at https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct (fp32, fp16, q4, q4f16) and the WebLLM builds Qwen2.5-0.5B-Instruct-q4f16_1-MLC and Qwen2.5-0.5B-Instruct-q4f32_1-MLC.

We chose a 0.5B model because it fits comfortably on a 4 GB GPU, and because small models are what people actually run in browsers. A second model (for example a 1–1.5B model) may be added if time allows.

### 5.4 Questions

| Benchmark | Number | Why |
|---|---|---|
| MMLU ([dataset](https://huggingface.co/datasets/cais/mmlu)) | 200 questions, about 4 per subject across all 57 subjects | Multiple choice; the answer is a single letter, so it tests short outputs |
| GSM8K ([dataset](https://huggingface.co/datasets/openai/gsm8k)) | 100 problems | Grade-school maths with step-by-step reasoning; long outputs, where small differences can grow |

The questions were sampled once with a fixed random seed (2026) and saved in `site/questions_v1.json`. `scripts/make_questions.py` rebuilds them exactly.

### 5.5 Hardware

| Component | Specification |
| :--- | :--- |
| **Device Type** | Laptop (Windows PC) |
| **Operating System** | Windows 11 (64-bit, Build x86_64) |
| **Processor (CPU)** | 13th Gen Intel(R) Core(TM) i5-13420H (12 vCPUs, up to 4.60 GHz) |
| **Discrete GPU (dGPU)** | NVIDIA GeForce RTX 2050 Laptop GPU (4 GB GDDR6 VRAM, Ampere architecture) |
| **Integrated GPU (iGPU)** | Intel(R) UHD Graphics (Gen-12LP architecture, shared system memory) |
| **System Memory (RAM)** | 8 GB |
| **Browser Tested** | Google Chrome 139.0.0.0 (Windows NT 10.0; Win64; x64) |
| **Host Runtime Baseline** | Node.js v20+ (`native-node-cpu/fp32`) |

Having NVIDIA and Intel GPUs in the same laptop is useful: the CPU, memory, operating system, and browser stay the same while only the GPU changes.

## 6. What we measure

For every setup and task:

1. Accuracy: the share of questions answered correctly.
2. Agreement with the reference: the share of questions where the setup gives the same final answer as the full-precision native reference (right or wrong).
3. Flips: correct-to-wrong and wrong-to-correct counts compared with the reference.
4. Exact output match: the share of answers whose full text is character-for-character identical to the reference.
5. First divergence position: when outputs differ, the token position where they first differ. Early divergence means the difference starts at the beginning of the answer; late divergence means it accumulates.
6. Repeatability: whether the same setup gives identical outputs when run twice.

Statistics:

- McNemar's exact test checks whether accuracy differences between two setups on the same questions are larger than chance.
- Bootstrap 95% confidence intervals are reported for agreement rates.

The flip idea comes from work on model compression showing that even when compressed models keep similar accuracy, individual answers change in both directions ([Accuracy is Not All You Need](https://openreview.net/pdf/7ecf82af167db2fa1a4c2972d42f903ce9507ee9.pdf)). Recent work also links internal output differences to such decision flips ([How Divergence Becomes Decision Flips in Compressed Language Models](https://arxiv.org/html/2610.00694v1)). We apply this lens to browser deployment, which, as far as we found, these works do not study.

## 7. Related work and what is new here

| Work | What it did | How this study differs |
|---|---|---|
| WebLLM ([arXiv 2412.15803](https://arxiv.org/html/2412.15803v2)) | In-browser engine; throughput vs native | We measure answers, not speed |
| WebGPU dispatch overhead ([arXiv 2604.02344](https://arxiv.org/html/2604.02344)) | Speed and latency across GPU vendors and browsers | We measure answer fidelity across vendors |
| LlamaWeb ([arXiv 2605.20706](https://arxiv.org/html/2605.20706v1)) | WebGPU backend for llama.cpp; memory, speed, kernel-level numeric checks | We test end-to-end answers on benchmarks |
| GenomeOcean Anywhere ([arXiv 2609.35882](https://arxiv.org/html/2609.35882v1)) | Browser vs native match for a genome model on one server GPU | We test a general chat model, multiple frameworks, consumer GPUs from two vendors |
| Hints-In-Browser, NeurIPS 2024 ([paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/34cc2ded6daba59357134c0b9fb06bfe-Paper-Datasets_and_Benchmarks_Track.pdf)) | Quality of web-converted 4-bit models for programming feedback | We separate compression effects from runtime effects |
| Nondeterminism in LLM inference ([project page](https://zirui-ray-liu.github.io/projects/mlsys-numerics/numerical-nondeterminism/)) | Answer changes across server GPUs and batch sizes | We study browsers and consumer GPUs |

Expected contribution: a controlled, reproducible measurement of whether in-browser inference changes the answers of a small LLM, separating runtime, GPU vendor, and compression effects, with an open harness others can reuse.

This related-work check was done on 3 October 2026 and will be repeated before posting the preprint.

## 8. What we will not claim

- We will not claim that results hold for all models, all browsers, or all GPUs. We test one small model (possibly two), Chromium-based browsers, and two GPU targets on our testbed (NVIDIA discrete and Intel integrated).
- We will not claim that one framework is "better". Frameworks use different model builds; we report differences, not rankings.
- We will not claim causes we did not test directly. If browser and native differ, we can say where and how much, and suggest likely reasons (for example precision of arithmetic), but not prove the exact low-level cause without kernel-level experiments.
- Benchmark accuracy for a 0.5B model is low in absolute terms. That is expected and not the point; the point is whether answers stay the same.

## 9. Known limitations

- Small sample: 300 questions per setup. Enough to detect clear differences, not tiny ones.
- Answer extraction: in some MMLU questions the small model writes text instead of a letter. This affects all setups equally because scoring code is shared, but it is reported.
- WebLLM uses its own model builds and chat handling, so WebLLM comparisons answer RQ3 (deployment effect), not RQ1.
- Browser and driver versions change often; we record versions in every results file.

## 10. Ethics and data

- No human participants and no personal data. All datasets are public benchmarks.
- No model training. All computation runs on personal laptops; no paid services.
- Code, question sets, and results are released openly so others can check and reproduce them.

## 11. Workflow and timeline

| Stage | Dates (planned) |
|---|---|
| Harness built, native reference started | 3–4 October 2026 |
| Browser runs across NVIDIA discrete and Intel integrated GPUs | October 2026 |
| Analysis (tables, figures, statistics) | Late October – early November |
| Writing (6–8 pages, IEEE format) and internal review | November |
| Preprint on TechRxiv; archive on Zenodo | By 30 November 2026 |

## 12. Authorship and contributions (CRediT taxonomy)

- **Sudip Niroula (First Author):** Conceptualisation, Methodology, Software (`site/` browser runner, `run_native.mjs`, `analyze.py`, and dataset sampling scripts), Formal Analysis, Visualisation, and Writing – Original Draft & Documentation.
- **Mandip Pokharel (Second Author):** Investigation (configuring hardware GPU assignments and executing all native CPU, browser WASM, Transformers.js WebGPU, and WebLLM benchmark runs on the Intel Core i5-13420H / NVIDIA GeForce RTX 2050 / Intel UHD Graphics testbed), Data Curation (`results/`), and Validation.

Every listed author must make a real contribution and approve the final paper.

## 13. How to take part or reproduce

1. Read [RUN_GUIDE.md](../RUN_GUIDE.md).
2. Serve the runner page locally (`python -m http.server 8000 --directory site`) and open http://localhost:8000 in Chrome.
3. Run the setups listed in the guide; place result files in `results/`.
4. Run `python analyze.py results native-node-cpu/fp32` to produce `results/summary.md`.

Questions or problems: open an issue in this repository.
