// Shared prompt construction and answer extraction.
// Used identically by the Node (native) runner and the browser runner,
// so that only the inference runtime differs between setups.

export const HARNESS_VERSION = "h2";

// Explicit system prompt (identical to the Qwen2.5 default) so every runtime receives the same prompt.
export const SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

export const GEN = {
  mmlu: { max_new_tokens: 5 },
  gsm8k: { max_new_tokens: 320 },
};

export function mmluMessages(item) {
  const letters = ["A", "B", "C", "D"];
  const opts = item.choices.map((c, i) => `${letters[i]}. ${c}`).join("\n");
  const subject = item.subject.replace(/_/g, " ");
  return [
    { role: "system", content: SYSTEM },
    {
      role: "user",
      content:
        `The following is a multiple choice question about ${subject}. ` +
        `Answer with only the letter (A, B, C, or D).\n\n` +
        `${item.question}\n${opts}\nAnswer:`,
    },
  ];
}

export function gsmMessages(item) {
  return [
    { role: "system", content: SYSTEM },
    {
      role: "user",
      content:
        "Solve the following math problem step by step. At the end, write the final answer " +
        "as 'The answer is N.' where N is a number.\n\n" +
        item.question,
    },
  ];
}

export function extractMmlu(text) {
  const m = text.match(/\b([ABCD])\b/) || text.match(/([ABCD])/);
  return m ? m[1] : null;
}

function normNum(s) {
  if (s == null) return null;
  s = s.replace(/[,$\s]/g, "").replace(/\.$/, "");
  const n = Number(s);
  if (Number.isNaN(n)) return null;
  return String(Number.isInteger(n) ? n : Number(n.toFixed(6)));
}

export function extractGsm(text) {
  const m = text.match(/answer is[:\s]*\$?\s*(-?[\d,]*\.?\d+)/i);
  if (m) return normNum(m[1]);
  const all = text.match(/-?[\d,]*\.?\d+/g);
  return all ? normNum(all[all.length - 1]) : null;
}

export function gsmGold(item) {
  return normNum(item.answer);
}
