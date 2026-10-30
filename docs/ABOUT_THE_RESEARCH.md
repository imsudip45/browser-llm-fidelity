# About This Research

Title: "Does Running a Small LLM in the Browser Change Its Answers? Comparing Native and WebGPU Inference Across GPUs"

Authors: Sudip Niroula (lead author), Mandip Pokharel (co-author)
Status: Completed (October 2026)
Archived DOI: https://doi.org/10.5281/zenodo.23760946 (archives the raw v1.0.0 benchmark outputs and harness; the v1.1.x multi-table audit pipeline is maintained in this repository)
Planned output: preprint on TechRxiv, with code and data in this repository (and an archived copy on Zenodo)

This document explains, in plain language, what the study is about, why it matters, how it is designed, and what we will and will not claim. It is written so that a co-author, a reviewer, or an admissions committee can understand the project without reading the code.

---

## 1. The one-sentence summary

We take the same small AI language model, ask it the same 300 questions in exactly the same way, and check whether its answers change when it runs inside a web browser instead of running normally on the computer, and whether they change between different GPUs.

## 2. Why this question matters

### 2.1 AI is moving into the browser

Small language models (around 0.5 to 3 billion parameters) can now run directly inside a web page, using the computer's own graphics card through a browser technology called WebGPU. Libraries such as WebLLM and Transformers.js make this practical. Client-side inference can avoid per-request inference-server computation and can keep prompts local when the application does not transmit them. Once weights are cached locally, it can also operate offline.

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
| Sampling disabled | Deterministic inference: sampling was disabled (`do_sample = false`) in Transformers.js, and WebLLM used `temperature = 0` with `seed = 0`. |
| Flip | A question where one setup is correct and another is wrong (or the other way round). |

## 4. Research questions

- RQ1, runtime effect: With exactly the same model file and code, does running in the browser change answers compared with native inference?
- RQ2, GPU vendor effect: Does the same browser setup give different answers on Intel-labelled integrated GPU versus NVIDIA-labelled discrete GPU runs on the same machine? (Adapter labels reflect a main-page `navigator.gpu.requestAdapter()` check after switching Windows Graphics Settings rather than internal worker attestation).
- RQ3, deployment effect: How far are ready-made browser model builds (for example WebLLM's 4-bit builds) from a full-precision reference, and how much of that gap is explained by compression rather than by the browser?
- RQ4, hidden changes: Do overall accuracy numbers hide answer flips? Two setups can have the same accuracy while getting different questions right.

## 5. How the study is designed

### 5.1 The main principle: change only one thing at a time

The most common mistake in comparisons like this is to change several things at once (for example, a different compression level and a different runtime) and then not know which one caused a difference. We avoid this by building comparisons in pairs:

| Comparison | Kept the same | Changed | Answers |
|---|---|---|---|
| Native Node.js (CPU) fp32 vs browser WebGPU fp32 | Model repository ID/dtype, library version, prompts | Runtime and hardware path | RQ1 |
| Native Node.js q4 vs browser WASM q4 | Model repository ID/dtype, library version, prompts | Runtime (native CPU vs browser CPU) | RQ1 |
| Browser WebGPU Intel-labelled vs NVIDIA-labelled | Everything except adapter assignment | GPU adapter | RQ2 |
| Browser fp16 run repeated twice on the same GPU | Everything | Nothing (repeatability check) | Baseline noise |
| Full-precision reference vs each compressed build (fp16, q4f16, WebLLM q4f16_1, q4f32_1) | Model, prompts | Compression and runtime | RQ3 |
| Chrome vs Edge on the same GPU (optional) | Everything except the browser | Browser | Minor check |

*(Note on weight pinning: Model repository identifiers and dtypes were strictly matched across runs, while SHA-256 file hashes were not pinned across the Oct 8–9 and Oct 28 run dates).*

### 5.2 Fixed and shared components

- One shared file (`site/harness.js`) builds every prompt and extracts every answer. The native runner and the browser runner both import it, so prompts and scoring are identical.
- The exact system prompt across all setups is: `"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."`
- The same Transformers.js version (4.3.0) is used natively and in the browser; WebLLM runs on version 0.2.85.
- For deterministic evaluation, sampling was disabled (`do_sample = false`) in Transformers.js, and WebLLM used `temperature = 0` with `seed = 0`. Fixed maximum output lengths were set to 5 new tokens for MMLU and 320 new tokens for GSM8K.
- Each results file records the exact setup: library versions, model, precision, browser user agent, and the GPU reported by WebGPU.

### 5.3 Model

Qwen2.5-0.5B-Instruct, using the ONNX files published at https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct (fp32, fp16, q4, q4f16) and the WebLLM builds Qwen2.5-0.5B-Instruct-q4f16_1-MLC and Qwen2.5-0.5B-Instruct-q4f32_1-MLC.

We chose a 0.5B model because it fits comfortably on a 4 GB GPU, and because small models are what people actually run in browsers. A second model (for example a 1–1.5B model) may be added if time allows.

### 5.4 Questions and Evaluation Inventory

| Benchmark | Number | Why |
|---|---|---|
| MMLU ([dataset](https://huggingface.co/datasets/cais/mmlu)) | 200 questions across all 57 subjects (capped at 4 per subject: 34 subjects with 4 items, 18 with 3 items, 5 with 2 items) | Multiple choice; the answer is a single letter, so it tests short outputs |
| GSM8K ([dataset](https://huggingface.co/datasets/openai/gsm8k)) | 100 problems | Grade-school maths with step-by-step reasoning; long outputs, where small differences can grow |

The questions were sampled once with a fixed random seed (2026) and saved in `site/questions_v1.json`. `scripts/make_questions.py` rebuilds them exactly.

In total, across 13 full runs ($N=300$) and one 20-item diagnostic WebAssembly check ($N=20$), the benchmark suite evaluates **14 run result files, 28 task evaluations, and 3,920 individual item evaluations**.

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

1. **Accuracy (`acc`):** the share of all $n$ questions answered correctly under standard parsing.
2. **Valid agreement (`valid_agree`):** the share of questions where *both* the test setup and the reference produced non-null parsed predictions (`pred_ref is not None and pred_test is not None`) and their final answers match. This isolates genuine answer concordance from unparsed failures.
3. **All-outcome agreement (`all_outcome_agree`):** the share of all $n$ questions where predictions match, including shared unparsed outcomes (`None == None`).
4. **Flips decomposition:** 
   - `flips_c2w`: reference correct, test wrong.
   - `flips_w2c`: reference wrong, test correct.
   - `unparsed_ref_to_correct`: reference unparsed/null, test correct (a subset of `w2c`).
   - `correct_to_unparsed_test`: reference correct, test unparsed/null.
5. **Exact output match (`exact_text`):** the share of answers whose full generated text is character-for-character identical to the reference under literal string comparison without whitespace trimming.
6. **Token divergence distribution:** for diverged responses, the token position where text first differs, evaluated via Hugging Face tokenization. We report the interpolated median (`median_first_div_token` via `statistics.median`), as well as counts of immediate divergence at token 0 (`n_div_at_0`) and token 1 (`n_div_at_1`).
7. **Parsing and truncation audit:** we measure standard parse rate (`parse_rate`), unparsed count (`n_unparsed`), accuracy among valid responses (`acc_parsed`), strict accuracy (`strict_acc` requiring exact regex patterns like `answer is <num>`), and truncation count (`n_truncated_or_max` hitting generation caps).
8. **Repeatability:** whether the same setup gives identical outputs when run twice on the same hardware (`nvidia-chrome` vs `nvidia-chrome-repeat`).

Statistics:

- McNemar's exact test checks whether accuracy differences between two setups on the same questions are larger than chance (formatted as `< 1e-4` when $p < 0.0001$).
- Bootstrap 95% confidence intervals (`agree_ci`) reflect item-resampling variability across the fixed 300-question set (where an observed 0% sample yields a degenerate `[0.000, 0.000]` percentile interval) and do not capture variance across devices, drivers, or sessions.

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

- We will not claim that results hold for all models, all browsers, or all GPUs. We test one small model (`Qwen2.5-0.5B-Instruct`), Chromium-based browsers, and two GPU targets on our testbed (NVIDIA GeForce RTX 2050 discrete and Intel UHD Graphics integrated).
- We will not claim that Transformers.js WebGPU degradation represents a diagnosed shader defect or an inherent flaw in the WebGPU standard. The collapse observed in Transformers.js v4.3.0 WebGPU (`fp16`/`q4f16`) is a reproducible failure of the tested Transformers.js v4.3.0 / ONNX Runtime Web WebGPU path relative to requested-dtype CPU controls, without claiming identical historical weight bytes (since SHA-256 weight hashes were not pinned across the Oct 8–9 and Oct 28 runs) or a diagnosed shader defect. The independent WebLLM runtime operates over WebGPU on the exact same hardware without generation collapse.
- We will not claim that differences between native CPU Transformers.js and browser WebLLM are purely runtime effects. WebLLM uses its own pre-compiled TVM model builds, proprietary quantization kernels, and execution engine; these comparisons evaluate end-to-end deployment stacks (RQ3) rather than isolated runtime differences alone.
- We will not claim causes we did not test directly. If browser and native differ, we report the exact token divergence points and behavioral differences, and suggest likely numerical or allocation causes, but do not prove low-level driver or kernel mechanics without kernel-level profiling.
- We will not claim that WebLLM `q4f32_1` vs `q4f16_1` isolates "32-bit accumulation" alone; these configurations differ across the MLC compilation stack in `model_dtype` (`float16` vs `float32`).
- We will not claim that `std::bad_alloc` proves full-precision models cannot physically fit in browser WebGPU memory generally or that compression is universally mandatory in browsers; rather, the tested `fp32` ONNX deployment failed during initialization with `ERROR_CODE: 6 (std::bad_alloc)` on this laptop under this loader stack, with the exact allocation site (host RAM, WASM heap, or GPU buffer) undetermined.
- Benchmark accuracy for a 0.5B model is modest in absolute terms. That is expected; the focus is measuring answer fidelity across execution environments.

## 9. Known limitations

- Fixed question sample: 300 questions per setup (200 MMLU, 100 GSM8K), totaling 3,920 item evaluations across 14 logs. While sufficient to detect statistically significant discrepancies and immediate token divergences, larger samples would provide finer confidence intervals.
- Answer extraction sensitivity: Qwen2.5-0.5B sometimes outputs reasoning text without isolating a single letter (MMLU) or hits the 320-token limit without emitting `"The answer is N."` (GSM8K). In native `fp32` CPU on MMLU, 38 of 39 unparsed items saturated the 5-token cap, while 1 item terminated early at 3 tokens (`"WEP"` + EOS, failing letter-only extraction). We report parse rates and strict extraction metrics explicitly to quantify this limitation.
- Memory constraints: Running unquantized `fp32` models under Transformers.js WebGPU fails during initialization on consumer laptops with 4 GB VRAM and 8 GB RAM (`std::bad_alloc`), requiring compressed formats for browser deployment.
- Model weight hashes: Model repository identifiers and dtypes were strictly matched, but SHA-256 file hashes were not pinned across the Oct 8–9 and Oct 28 run dates.
- Software drift: Browser, driver, and WebGPU implementation versions evolve rapidly; exact software versions are preserved in every JSON artifact. The Zenodo DOI `10.5281/zenodo.23760946` archives the raw `v1.0.0` benchmark data, whereas the updated audit pipeline (`v1.1.x`) is maintained in the git repository.

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
