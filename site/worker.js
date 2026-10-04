import * as H from "./harness.js";

const TJS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0";
const WEBLLM_URL = "https://esm.run/@mlc-ai/web-llm@0.2.85";
const TJS_MODEL = "onnx-community/Qwen2.5-0.5B-Instruct";
const WEBLLM_MODELS = {
  q4f16_1: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC",
  q4f32_1: "Qwen2.5-0.5B-Instruct-q4f32_1-MLC",
};
const post = (type, data) => self.postMessage({ type, ...data });
const log = (s) => post("log", { s });
const status = (s) => post("status", { s });
let stopFlag = false;

async function makeTjs(device, dtype) {
  const T = await import(TJS_URL);
  T.env.allowLocalModels = false;
  log(`Loading ${TJS_MODEL} (${dtype}) on ${device}... first time downloads the model.`);
  const tok = await T.AutoTokenizer.from_pretrained(TJS_MODEL);
  const model = await T.AutoModelForCausalLM.from_pretrained(TJS_MODEL, {
    dtype, device,
    progress_callback: (p) => { if (p.status === "progress" && p.total) status(`Downloading ${p.file}: ${(p.progress || 0).toFixed(0)}%`); },
  });
  return {
    runtime: `transformers.js@4.3.0 onnxruntime-web (${device})`,
    model: TJS_MODEL,
    async gen(messages, max_new_tokens) {
      const inputs = tok.apply_chat_template(messages, { add_generation_prompt: true, return_dict: true });
      const n = inputs.input_ids.dims[1];
      const out = await model.generate({ ...inputs, max_new_tokens, do_sample: false });
      const all = Array.from(out.tolist()[0]).map(Number);
      const ids = all.slice(n);
      return { text: tok.decode(ids, { skip_special_tokens: true }), ids, plen: n };
    },
  };
}

async function makeWebllm(build) {
  const webllm = await import(WEBLLM_URL);
  const id = WEBLLM_MODELS[build];
  log(`Loading WebLLM model ${id}... first time downloads the model.`);
  const engine = await webllm.CreateMLCEngine(id, { initProgressCallback: (p) => status(p.text) });
  return {
    runtime: `web-llm@0.2.85`,
    model: id,
    async gen(messages, max_new_tokens) {
      await engine.resetChat();
      const r = await engine.chat.completions.create({ messages, temperature: 0, top_p: 1, max_tokens: max_new_tokens, seed: 0 });
      return { text: r.choices[0].message.content || "", ids: null, plen: r.usage?.prompt_tokens ?? null };
    },
  };
}

self.onmessage = async (ev) => {
  const m = ev.data;
  if (m.type === "stop") { stopFlag = true; return; }
  if (m.type !== "run") return;
  stopFlag = false;
  const { engineKey, dtype, task, limit, label, gpu, userAgent, hardwareConcurrency } = m;
  try {
    const Q = await (await fetch("./questions_v1.json")).json();
    const eng = engineKey === "webllm" ? await makeWebllm(dtype)
      : await makeTjs(engineKey === "tjs-webgpu" ? "webgpu" : "wasm", dtype);
    const jobs = [];
    if (task !== "gsm8k") for (const it of Q.mmlu.slice(0, limit)) jobs.push(["mmlu", it]);
    if (task !== "mmlu") for (const it of Q.gsm8k.slice(0, limit)) jobs.push(["gsm8k", it]);
    const meta = {
      setup: `browser/${engineKey}/${dtype}/${label}`, runtime: eng.runtime, model: eng.model,
      dtype, engine: engineKey, label, gpu, userAgent, hardwareConcurrency,
      harness: H.HARNESS_VERSION, questions: Q.version, started: new Date().toISOString(),
    };
    const results = [];
    const t0 = performance.now();
    for (let i = 0; i < jobs.length && !stopFlag; i++) {
      const [k, it] = jobs[i];
      const t = performance.now();
      const msgs = k === "mmlu" ? H.mmluMessages(it) : H.gsmMessages(it);
      const { text, ids, plen } = await eng.gen(msgs, H.GEN[k].max_new_tokens);
      const pred = k === "mmlu" ? H.extractMmlu(text) : H.extractGsm(text);
      const gold = k === "mmlu" ? it.answer : H.gsmGold(it);
      results.push({ id: it.id, task: k, text, ids, plen, pred, gold, correct: pred === gold, ms: Math.round(performance.now() - t) });
      const acc = (results.filter((r) => r.correct).length / results.length * 100).toFixed(1);
      post("progress", { done: i + 1, total: jobs.length, acc, elapsed: Math.round((performance.now() - t0) / 1000) });
      if ((i + 1) % 10 === 0) post("partial", { data: { meta, results } });
    }
    meta.finished = new Date().toISOString();
    meta.complete = !stopFlag;
    post("done", { data: { meta, results } });
  } catch (e) {
    post("error", { s: String(e && e.stack ? e.stack : e) });
  }
};
