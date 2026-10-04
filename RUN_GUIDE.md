# How to Run the Browser Experiments on Your Laptop

Two ways to open the runner page:

- Hosted (only works when logged in as Sudip): https://www.perplexity.ai/computer/a/browser-llm-fidelity-runner-4_26W29pSsGx0nK9REqGUA
- Local (works for anyone, including co-authors): clone this repository, then in the repository folder run

  ```
  python -m http.server 8000 --directory site
  ```

  and open http://localhost:8000 in Chrome. (WebGPU works on localhost. Do not open index.html by double-clicking; it must be served.)

Model: Qwen2.5-0.5B-Instruct. Questions: 200 MMLU + 100 GSM8K (fixed, seed 2026). Greedy decoding.
Every run downloads one JSON file. Put all result files into the `results/` folder of this repository (commit them, or send them to Sudip).

---

## 1. One-time preparation

1. Plug the laptop into power. Set Windows power mode to "Best performance". Close games and heavy apps.
2. Update Chrome to the latest version.
3. Check WebGPU: open `chrome://gpu` and confirm "WebGPU: Hardware accelerated".
4. Free disk space: about 5 GB (model files are cached by the browser).
5. Internet: first runs download about 4 GB in total (fp32 2 GB, fp16 1 GB, q4f16 0.5 GB, WebLLM builds about 0.3–0.5 GB each). Use Wi-Fi without a data cap.

## 2. Choosing which GPU the browser uses

Windows decides which GPU Chrome uses:

1. Settings > System > Display > Graphics.
2. Find Google Chrome (add it with "Browse" if missing: `C:\Program Files\Google\Chrome\Application\chrome.exe`).
3. Click Options:
   - "High performance" = NVIDIA GeForce GTX 1650 Ti
   - "Power saving" = AMD Radeon Graphics
4. Close Chrome completely (all windows) and reopen.
5. Open the runner page. Section 1 shows the detected GPU. Check that it says nvidia or amd as expected before every run.

## 3. Runs to do (in this order)

Before each run: choose Engine and Precision, keep Task = All, leave Limit blank, type the Run label exactly as below, press Start. Do not close the tab or let the laptop sleep. The page downloads the results file automatically at the end.

### A. NVIDIA GTX 1650 Ti (Chrome, "High performance")

| # | Engine | Precision | Run label |
|---|---|---|---|
| 1 | Transformers.js, WebGPU | q4f16 | nvidia-chrome |
| 2 | Transformers.js, WebGPU | fp16 | nvidia-chrome |
| 3 | Transformers.js, WebGPU | fp32 | nvidia-chrome |
| 4 | WebLLM | q4f16_1 | nvidia-chrome |
| 5 | WebLLM | q4f32_1 | nvidia-chrome |

### B. AMD Radeon integrated (Chrome, "Power saving")

Same 5 runs, label `amd-chrome`.

### C. CPU only (Chrome, any GPU setting)

| # | Engine | Precision | Run label |
|---|---|---|---|
| 11 | Transformers.js, WASM (CPU) | q4 | cpu-chrome |

This one is slow. Run it overnight.

### D. Repeatability check (NVIDIA, Chrome)

Repeat run 2 (WebGPU fp16) with label `nvidia-chrome-repeat`. This shows whether the same setup gives identical outputs twice.

### E. Optional: Edge browser (NVIDIA)

Repeat runs 2 and 4 in Microsoft Edge with label `nvidia-edge`.

### F. Friend's RTX 4060 laptop

Your friend repeats section A with label `rtx4060-chrome`, and sends you the files.

## 4. Quick test first

Before the full runs, do one quick test: Engine Transformers.js WebGPU, Precision q4f16, Limit 3, label `test`. It should finish within a few minutes after the download. If you see an ERROR in the log box, copy it and send it to me.

## 5. Time estimates (rough, will vary)

- WebGPU q4f16 / WebLLM on NVIDIA: about 10–30 minutes per run.
- fp32 / fp16 WebGPU: slower, maybe 30–60 minutes.
- AMD integrated GPU: maybe 2–3 times slower than NVIDIA.
- CPU WASM: several hours.

## 6. If something goes wrong

- "WebGPU not available": update Chrome; check `chrome://gpu`; update GPU drivers (NVIDIA app, AMD Adrenalin).
- Page freezes or the tab crashes on fp32: note it (it is a result too), then continue with the other runs.
- Stopped halfway: press "Download results" to keep partial results, and note it.
- Record anything unusual in a text file: date, browser version (`chrome://version`), driver versions.
