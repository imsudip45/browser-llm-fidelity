const $ = (id) => document.getElementById(id);
const log = (s) => { $("log").textContent += s + "\n"; $("log").scrollTop = 1e9; };
let gpuInfo = { available: false };
let lastRun = null;
let worker = null;
window.__bfid = { state: "idle", lastRun: null };

const DTYPES = {
  "tjs-webgpu": ["fp32", "fp16", "q4f16", "q4"],
  "tjs-wasm": ["q4", "fp32", "fp16"],
  webllm: ["q4f16_1", "q4f32_1"],
};
function fillDtypes() {
  $("dtype").innerHTML = DTYPES[$("engine").value].map((d) => `<option>${d}</option>`).join("");
}
$("engine").addEventListener("change", fillDtypes);
fillDtypes();

async function detectGpu() {
  if (!("gpu" in navigator)) { $("gpu").innerHTML = "<b>WebGPU</b> not available in this browser."; return; }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) { $("gpu").textContent = "No WebGPU adapter found."; return; }
    const info = adapter.info || {};
    gpuInfo = {
      available: true, vendor: info.vendor || "", architecture: info.architecture || "",
      device: info.device || "", description: info.description || "",
      shader_f16: adapter.features.has("shader-f16"), isFallbackAdapter: !!(info.isFallbackAdapter || adapter.isFallbackAdapter),
    };
    $("gpu").innerHTML =
      `<div><b>Vendor</b>${gpuInfo.vendor || "(hidden)"}</div>` +
      `<div><b>Architecture</b>${gpuInfo.architecture || "(hidden)"}</div>` +
      `<div><b>Description</b>${gpuInfo.description || "(hidden)"}</div>` +
      `<div><b>shader-f16</b>${gpuInfo.shader_f16}</div>`;
  } catch (e) { $("gpu").textContent = "Could not get WebGPU adapter: " + e; }
}
detectGpu();

function download(obj, name) {
  const blob = new Blob([JSON.stringify(obj)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = name; a.click();
}

$("dl").addEventListener("click", () => lastRun && download(lastRun.data, lastRun.name));
$("stop").addEventListener("click", () => worker && worker.postMessage({ type: "stop" }));

$("run").addEventListener("click", () => {
  $("run").disabled = true; $("stop").disabled = false; $("dl").disabled = true;
  $("log").textContent = ""; $("pbar").style.width = "0";
  const engineKey = $("engine").value, dtype = $("dtype").value, task = $("task").value;
  const limit = Number($("limit").value) || Infinity;
  const label = ($("label").value || "run").replace(/[^a-z0-9_-]/gi, "-");
  const name = `browser_${engineKey}_${dtype}_${label}_${task}.json`;
  if (worker) worker.terminate();
  worker = new Worker("./worker.js", { type: "module" });
  window.__bfid.state = "running";
  worker.onmessage = (ev) => {
    const m = ev.data;
    if (m.type === "log") log(m.s);
    else if (m.type === "status") $("status").textContent = m.s;
    else if (m.type === "progress") {
      $("pbar").style.width = `${(m.done / m.total) * 100}%`;
      $("status").textContent = `${m.done}/${m.total} done, running accuracy ${m.acc}%, elapsed ${m.elapsed}s`;
    } else if (m.type === "partial") {
      lastRun = { name, data: m.data }; window.__bfid.lastRun = lastRun; $("dl").disabled = false;
    } else if (m.type === "done") {
      lastRun = { name, data: m.data }; window.__bfid.lastRun = lastRun; window.__bfid.state = "done";
      $("dl").disabled = false; $("run").disabled = false; $("stop").disabled = true;
      log(m.data.meta.complete ? "Finished. Downloading results file..." : "Stopped. Partial results can be downloaded.");
      if (m.data.meta.complete) download(m.data, name);
    } else if (m.type === "error") {
      window.__bfid.state = "error"; log("ERROR: " + m.s);
      $("run").disabled = false; $("stop").disabled = true;
    }
  };
  worker.postMessage({ type: "run", engineKey, dtype, task, limit, label, gpu: gpuInfo,
    userAgent: navigator.userAgent, hardwareConcurrency: navigator.hardwareConcurrency });
});
