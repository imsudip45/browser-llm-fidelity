"""Compare every result file against the reference setup.

Usage: python3 analyze.py [results_dir] [reference_setup_substring]
Writes results/summary.md and results/summary.csv
"""
import glob, json, os, sys, random, csv
from scipy.stats import binomtest

RD = sys.argv[1] if len(sys.argv) > 1 else "results"
REF = sys.argv[2] if len(sys.argv) > 2 else "native-node-cpu/fp32"

# Optional Qwen2.5 tokenizer fallback for runtimes (like WebLLM) that omit token IDs
TOKENIZER = None
try:
    from transformers import AutoTokenizer
    TOKENIZER = AutoTokenizer.from_pretrained("onnx-community/Qwen2.5-0.5B-Instruct")
except Exception:
    TOKENIZER = None


def get_tokens(item):
    """Return token ID list if present; otherwise tokenize text (or fallback to whitespace words)."""
    ids = item.get("ids")
    if ids is not None and len(ids) > 0:
        return ids
    text = item.get("text", "")
    if TOKENIZER is not None:
        return TOKENIZER.encode(text, add_special_tokens=False)
    return text.split()


def load():
    runs = {}
    for f in sorted(glob.glob(os.path.join(RD, "*.json"))):
        try:
            with open(f, encoding="utf-8") as fp:
                d = json.load(fp)
        except Exception:
            continue
        if "meta" not in d:
            continue
        runs[d["meta"]["setup"]] = d
    return runs


def first_div(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def mcnemar(b, c):
    # exact two-sided McNemar on discordant pairs
    if b + c == 0:
        return 1.0
    return binomtest(b, b + c, 0.5).pvalue


def boot_ci(xs, it=2000, seed=0):
    if not xs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(xs)
    ms = sorted(sum(rng.choice(xs) for _ in range(n)) / n for _ in range(it))
    return ms[int(0.025 * it)], ms[int(0.975 * it)]


def main():
    runs = load()
    if REF not in runs:
        print("reference not found:", REF, "available:", list(runs))
        return
    ref = {r["id"]: r for r in runs[REF]["results"]}
    rows = []
    for setup, d in runs.items():
        res = {r["id"]: r for r in d["results"]}
        for task in ["mmlu", "gsm8k"]:
            ids = [i for i, r in res.items() if r["task"] == task and i in ref]
            if not ids:
                continue
            acc = sum(res[i]["correct"] for i in ids) / len(ids)
            racc = sum(ref[i]["correct"] for i in ids) / len(ids)
            agree = [int(res[i]["pred"] == ref[i]["pred"]) for i in ids]
            c2w = sum(1 for i in ids if ref[i]["correct"] and not res[i]["correct"])
            w2c = sum(1 for i in ids if not ref[i]["correct"] and res[i]["correct"])
            exact = sum(1 for i in ids if res[i]["text"] == ref[i]["text"]) / len(ids)
            divs = []
            for i in ids:
                if res[i]["text"] == ref[i]["text"]:
                    continue
                # If both have raw token IDs, compare raw IDs; otherwise compare re-tokenized text
                a_raw, b_raw = res[i].get("ids"), ref[i].get("ids")
                if a_raw is not None and b_raw is not None and len(a_raw) > 0 and len(b_raw) > 0:
                    fd = first_div(a_raw, b_raw)
                else:
                    if TOKENIZER is not None:
                        a_tok = TOKENIZER.encode(res[i].get("text", ""), add_special_tokens=False)
                        b_tok = TOKENIZER.encode(ref[i].get("text", ""), add_special_tokens=False)
                    else:
                        a_tok = res[i].get("text", "").split()
                        b_tok = ref[i].get("text", "").split()
                    fd = first_div(a_tok, b_tok)
                if fd is not None:
                    divs.append(fd)
                else:
                    # Texts differ only in trailing whitespace
                    divs.append(0)
            lo, hi = boot_ci(agree)
            rows.append(dict(
                setup=setup, task=task, n=len(ids), acc=round(acc, 3), ref_acc=round(racc, 3),
                agree=round(sum(agree) / len(ids), 3), agree_ci=f"[{lo:.3f}, {hi:.3f}]",
                flips_c2w=c2w, flips_w2c=w2c, mcnemar_p=round(mcnemar(c2w, w2c), 4),
                exact_text=round(exact, 3),
                median_first_div_token=(sorted(divs)[len(divs) // 2] if divs else ""),
                n_diverged=len(divs),
                gpu=(d["meta"].get("gpu") or {}).get("description") or (d["meta"].get("gpu") or {}).get("vendor") or d["meta"].get("device", ""),
            ))
    rows.sort(key=lambda r: (r["task"], r["setup"]))
    with open(os.path.join(RD, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    cols = ["setup", "task", "n", "acc", "ref_acc", "agree", "agree_ci", "flips_c2w", "flips_w2c", "mcnemar_p", "exact_text", "median_first_div_token", "n_diverged", "gpu"]
    lines = [
        f"Reference: {REF}",
        "Hardware: Laptop with Intel Core i5-13th Gen H processor (i5-13420H), NVIDIA GeForce RTX 2050 (4 GB VRAM), and Intel UHD Graphics",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    with open(os.path.join(RD, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()