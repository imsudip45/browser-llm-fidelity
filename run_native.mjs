// Native runner: Transformers.js on Node (onnxruntime-node, CPU).
// Usage: node run_native.mjs <model_id> <dtype> <task:mmlu|gsm8k|all> [limit] [outfile]
import fs from "node:fs";
import os from "node:os";
import { AutoTokenizer, AutoModelForCausalLM, env } from "@huggingface/transformers";
import * as H from "./site/harness.js";

const [modelId = "onnx-community/Qwen2.5-0.5B-Instruct", dtype = "fp32", task = "all", limitArg, outArg] =
  process.argv.slice(2);
const limit = limitArg ? Number(limitArg) : Infinity;
const Q = JSON.parse(fs.readFileSync("data/questions_v1.json", "utf8"));
const out = outArg || `results/native_cpu_${modelId.split("/")[1]}_${dtype}_${task}.json`;
fs.mkdirSync("results", { recursive: true });

const tfjsVersion = JSON.parse(
  fs.readFileSync("node_modules/@huggingface/transformers/package.json", "utf8")
).version;

const tok = await AutoTokenizer.from_pretrained(modelId);
const model = await AutoModelForCausalLM.from_pretrained(modelId, { dtype, device: "cpu" });

async function gen(messages, max_new_tokens) {
  const inputs = tok.apply_chat_template(messages, {
    add_generation_prompt: true,
    return_dict: true,
  });
  const n = inputs.input_ids.dims[1];
  const outIds = await model.generate({ ...inputs, max_new_tokens, do_sample: false });
  const all = Array.from(outIds.tolist()[0]).map(Number);
  const newIds = all.slice(n);
  const text = tok.decode(newIds, { skip_special_tokens: true });
  return { text, ids: newIds, plen: n };
}

const meta = {
  setup: `native-node-cpu/${dtype}`,
  runtime: "transformers.js@" + tfjsVersion + " onnxruntime-node (CPU)",
  model: modelId,
  dtype,
  device: "cpu",
  host: `${os.cpus()[0].model} x${os.cpus().length}, ${Math.round(os.totalmem() / 1e9)}GB`,
  harness: H.HARNESS_VERSION,
  questions: Q.version,
  started: new Date().toISOString(),
};
const results = [];
const save = () => fs.writeFileSync(out, JSON.stringify({ meta, results }, null, 0));

const jobs = [];
if (task === "mmlu" || task === "all") for (const it of Q.mmlu.slice(0, limit)) jobs.push(["mmlu", it]);
if (task === "gsm8k" || task === "all") for (const it of Q.gsm8k.slice(0, limit)) jobs.push(["gsm8k", it]);

let t0 = Date.now();
for (const [k, it] of jobs) {
  const t = Date.now();
  const msgs = k === "mmlu" ? H.mmluMessages(it) : H.gsmMessages(it);
  const { text, ids, plen } = await gen(msgs, H.GEN[k].max_new_tokens);
  const pred = k === "mmlu" ? H.extractMmlu(text) : H.extractGsm(text);
  const gold = k === "mmlu" ? it.answer : H.gsmGold(it);
  results.push({ id: it.id, task: k, text, ids, plen, pred, gold, correct: pred === gold, ms: Date.now() - t });
  if (results.length % 10 === 0) {
    save();
    console.log(`${results.length}/${jobs.length} ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  }
}
meta.finished = new Date().toISOString();
save();
const acc = (k) => {
  const r = results.filter((x) => x.task === k);
  return r.length ? (r.filter((x) => x.correct).length / r.length).toFixed(3) + ` (n=${r.length})` : "-";
};
console.log("done", out, "mmlu", acc("mmlu"), "gsm8k", acc("gsm8k"));
