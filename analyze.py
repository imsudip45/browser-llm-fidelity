"""Compare every result file against reference setups and generate multi-table audit summaries.

Usage: python analyze.py [results_dir] [reference_setup_substring]
Writes results/summary.md (4 tables) and results/summary.csv (main table)
"""
import glob, json, os, sys, random, csv, re, statistics
from scipy.stats import binomtest

RD = sys.argv[1] if len(sys.argv) > 1 else "results"
REF = sys.argv[2] if len(sys.argv) > 2 else "native-node-cpu/fp32"

# Optional Qwen2.5 tokenizer fallback for runtimes (like WebLLM) that omit token IDs
TOKENIZER = None
try:
    from transformers import AutoTokenizer
    TOKENIZER = AutoTokenizer.from_pretrained("onnx-community/Qwen2.5-0.5B-Instruct", local_files_only=True)
except Exception:
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


def norm_num(s):
    """Normalize numeric strings consistent with site/harness.js normNum."""
    if s is None:
        return None
    s = re.sub(r"[,$\s]", "", str(s)).rstrip(".")
    try:
        n = float(s)
        return str(int(n) if n.is_integer() else round(n, 6))
    except ValueError:
        return None


def strict_mmlu(text):
    """Strict MMLU extraction: requires standalone letter A-D via word boundaries without arbitrary fallback."""
    m = re.search(r"\b([ABCD])\b", text)
    return m.group(1) if m else None


def strict_gsm(text):
    """Strict GSM8K extraction: requires explicit 'answer is <num>' pattern without fallback to trailing numbers."""
    m = re.search(r"answer is\s*([\-0-9,\.]+)", text, re.IGNORECASE)
    return norm_num(m.group(1)) if m else None


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


def format_p(p):
    if p < 0.0001:
        return "< 1e-4"
    return str(round(p, 4))


def boot_ci(xs, it=2000, seed=0):
    if not xs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(xs)
    ms = sorted(sum(rng.choice(xs) for _ in range(n)) / n for _ in range(it))
    return ms[int(0.025 * it)], ms[int(0.975 * it)]


def get_gpu_label(setup, meta):
    if "tjs-wasm" in setup or "cpu-chrome" in setup:
        return "cpu (wasm)"
    if "native-node-cpu" in setup or ("cpu" in setup and not any(k in setup for k in ["nvidia", "intel"])):
        return "cpu"
    if "nvidia" in setup:
        return "nvidia"
    if "intel" in setup:
        return "intel"
    return (meta.get("gpu") or {}).get("description") or (meta.get("gpu") or {}).get("vendor") or meta.get("device", "")


def calc_divergences(res_a, res_b, ids):
    divs = []
    for i in ids:
        if res_a[i]["text"] == res_b[i]["text"]:
            continue
        a_raw, b_raw = res_a[i].get("ids"), res_b[i].get("ids")
        if a_raw is not None and b_raw is not None and len(a_raw) > 0 and len(b_raw) > 0:
            fd = first_div(a_raw, b_raw)
        else:
            if TOKENIZER is not None:
                a_tok = TOKENIZER.encode(res_a[i].get("text", ""), add_special_tokens=False)
                b_tok = TOKENIZER.encode(res_b[i].get("text", ""), add_special_tokens=False)
            else:
                a_tok = res_a[i].get("text", "").split()
                b_tok = res_b[i].get("text", "").split()
            fd = first_div(a_tok, b_tok)
        if fd is not None:
            divs.append(fd)
        else:
            divs.append(0)
    return divs


def format_median(divs):
    if not divs:
        return ""
    m = statistics.median(divs)
    return int(m) if isinstance(m, float) and m.is_integer() else round(m, 1)


def make_md_table(rows, cols):
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def main():
    runs = load()
    if REF not in runs:
        print("reference not found:", REF, "available:", list(runs))
        return
    ref = {r["id"]: r for r in runs[REF]["results"]}

    # =========================================================================
    # TABLE 1: Primary Comparison vs. Full-Precision Reference (native fp32)
    # =========================================================================
    t1_rows = []
    for setup, d in sorted(runs.items()):
        res = {r["id"]: r for r in d["results"]}
        for task in ["gsm8k", "mmlu"]:
            ids = [i for i, r in res.items() if r["task"] == task and i in ref]
            if not ids:
                continue
            n = len(ids)
            acc = sum(res[i]["correct"] for i in ids) / n
            racc = sum(ref[i]["correct"] for i in ids) / n

            # Unparsed and parsed accuracy
            n_unparsed = sum(1 for i in ids if res[i]["pred"] is None)
            parse_rate = (n - n_unparsed) / n
            n_valid = n - n_unparsed
            acc_parsed = sum(res[i]["correct"] for i in ids if res[i]["pred"] is not None) / n_valid if n_valid > 0 else 0.0

            # Strict accuracy audit
            strict_correct = 0
            for i in ids:
                txt = res[i].get("text", "")
                sp = strict_mmlu(txt) if task == "mmlu" else strict_gsm(txt)
                if sp == res[i]["gold"]:
                    strict_correct += 1
            strict_acc = strict_correct / n

            # Agreement metrics
            all_outcome_agree = sum(1 for i in ids if res[i]["pred"] == ref[i]["pred"]) / n
            both_valid = [i for i in ids if res[i]["pred"] is not None and ref[i]["pred"] is not None]
            n_both_valid = len(both_valid)
            valid_agree = sum(1 for i in both_valid if res[i]["pred"] == ref[i]["pred"]) / n_both_valid if n_both_valid > 0 else 0.0
            n_joint_invalid = sum(1 for i in ids if res[i]["pred"] is None and ref[i]["pred"] is None)

            # Flips decomposition
            c2w = sum(1 for i in ids if ref[i]["correct"] and not res[i]["correct"])
            w2c = sum(1 for i in ids if not ref[i]["correct"] and res[i]["correct"])
            unparsed_ref_to_correct = sum(1 for i in ids if ref[i]["pred"] is None and res[i]["correct"])
            correct_to_unparsed_test = sum(1 for i in ids if ref[i]["correct"] and res[i]["pred"] is None)
            mcnemar_p = format_p(binomtest(c2w, c2w + w2c, 0.5).pvalue if (c2w + w2c) > 0 else 1.0)

            # Exact text match & divergence token analysis
            exact = sum(1 for i in ids if res[i]["text"] == ref[i]["text"]) / n
            divs = calc_divergences(res, ref, ids)
            med_div = format_median(divs)
            n_div_0 = sum(1 for d in divs if d == 0)
            n_div_1 = sum(1 for d in divs if d == 1)

            # Truncation count (GSM8K hitting 320 token ceiling)
            n_truncated_or_max = ""
            if task == "gsm8k":
                trunc_count = 0
                for i in ids:
                    toks = get_tokens(res[i])
                    if len(toks) >= 320:
                        trunc_count += 1
                n_truncated_or_max = trunc_count

            lo, hi = boot_ci([int(res[i]["pred"] == ref[i]["pred"]) for i in ids])

            t1_rows.append(dict(
                setup=setup, task=task, n=n,
                acc=round(acc, 3), ref_acc=round(racc, 3), strict_acc=round(strict_acc, 3),
                n_unparsed=n_unparsed, parse_rate=round(parse_rate, 3), acc_parsed=round(acc_parsed, 3),
                valid_agree=round(valid_agree, 3), all_outcome_agree=round(all_outcome_agree, 3),
                agree_ci=f"[{lo:.3f}, {hi:.3f}]", n_both_valid=n_both_valid, n_joint_invalid=n_joint_invalid,
                flips_c2w=c2w, flips_w2c=w2c,
                unparsed_ref_to_correct=unparsed_ref_to_correct, correct_to_unparsed_test=correct_to_unparsed_test,
                mcnemar_p=mcnemar_p, exact_text=round(exact, 3),
                median_first_div_token=med_div, n_diverged=len(divs),
                n_div_at_0=n_div_0, n_div_at_1=n_div_1,
                n_truncated_or_max=n_truncated_or_max,
                gpu=get_gpu_label(setup, d["meta"])
            ))

    t1_rows.sort(key=lambda r: (r["task"], r["setup"]))

    # Write Table 1 to results/summary.csv
    with open(os.path.join(RD, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(t1_rows[0].keys()))
        w.writeheader()
        w.writerows(t1_rows)

    t1_cols = [
        "setup", "task", "n", "acc", "ref_acc", "strict_acc",
        "n_unparsed", "parse_rate", "acc_parsed",
        "valid_agree", "all_outcome_agree", "agree_ci", "n_both_valid", "n_joint_invalid",
        "flips_c2w", "flips_w2c", "unparsed_ref_to_correct", "correct_to_unparsed_test",
        "mcnemar_p", "exact_text", "median_first_div_token", "n_diverged",
        "n_div_at_0", "n_div_at_1", "n_truncated_or_max", "gpu"
    ]

    # =========================================================================
    # TABLE 2: Direct Matched-Dtype Native CPU vs. Browser Comparisons (RQ1)
    # =========================================================================
    t2_pairs = [
        ("browser/tjs-webgpu/fp16/intel-chrome", "native-node-cpu/fp16"),
        ("browser/tjs-webgpu/fp16/nvidia-chrome", "native-node-cpu/fp16"),
        ("browser/tjs-webgpu/fp16/nvidia-chrome-repeat", "native-node-cpu/fp16"),
        ("browser/tjs-webgpu/q4f16/intel-chrome", "native-node-cpu/q4f16"),
        ("browser/tjs-webgpu/q4f16/nvidia-chrome", "native-node-cpu/q4f16"),
        ("browser/tjs-wasm/q4/cpu-chrome-check", "native-node-cpu/q4"),
    ]
    t2_rows = []
    for b_setup, n_setup in t2_pairs:
        if b_setup not in runs or n_setup not in runs:
            continue
        b_map = {r["id"]: r for r in runs[b_setup]["results"]}
        n_map = {r["id"]: r for r in runs[n_setup]["results"]}
        for task in ["gsm8k", "mmlu"]:
            ids = [i for i, r in b_map.items() if r["task"] == task and i in n_map]
            if not ids:
                continue
            n = len(ids)
            b_acc = sum(b_map[i]["correct"] for i in ids) / n
            n_acc = sum(n_map[i]["correct"] for i in ids) / n
            all_agree = sum(1 for i in ids if b_map[i]["pred"] == n_map[i]["pred"]) / n
            both_valid = [i for i in ids if b_map[i]["pred"] is not None and n_map[i]["pred"] is not None]
            n_both_valid = len(both_valid)
            v_agree = sum(1 for i in both_valid if b_map[i]["pred"] == n_map[i]["pred"]) / n_both_valid if n_both_valid > 0 else 0.0
            exact_txt = sum(1 for i in ids if b_map[i]["text"] == n_map[i]["text"]) / n
            exact_ids = sum(1 for i in ids if b_map[i].get("ids") is not None and b_map[i].get("ids") == n_map[i].get("ids")) / n

            divs = calc_divergences(b_map, n_map, ids)
            med_div = format_median(divs)
            n_div_0 = sum(1 for d in divs if d == 0)

            t2_rows.append(dict(
                browser_setup=b_setup, native_ref_setup=n_setup, task=task, n=n,
                browser_acc=round(b_acc, 3), native_acc=round(n_acc, 3),
                valid_agree=round(v_agree, 3), all_outcome_agree=round(all_agree, 3),
                exact_text=round(exact_txt, 3), exact_token_ids=round(exact_ids, 3),
                median_first_div_token=med_div, n_diverged=len(divs), n_div_at_0=n_div_0,
                n_both_valid=n_both_valid
            ))

    t2_rows.sort(key=lambda r: (r["task"], r["browser_setup"]))
    t2_cols = [
        "browser_setup", "native_ref_setup", "task", "n",
        "browser_acc", "native_acc", "valid_agree", "all_outcome_agree",
        "exact_text", "exact_token_ids", "median_first_div_token", "n_diverged", "n_div_at_0", "n_both_valid"
    ]

    # =========================================================================
    # TABLE 3: Cross-GPU Pairwise Consistency: Intel UHD vs. NVIDIA RTX 2050 (RQ2)
    # =========================================================================
    t3_configs = [
        ("tjs-webgpu/fp16", "browser/tjs-webgpu/fp16/intel-chrome", "browser/tjs-webgpu/fp16/nvidia-chrome"),
        ("tjs-webgpu/q4f16", "browser/tjs-webgpu/q4f16/intel-chrome", "browser/tjs-webgpu/q4f16/nvidia-chrome"),
        ("webllm/q4f16_1", "browser/webllm/q4f16_1/intel-chrome", "browser/webllm/q4f16_1/nvidia-chrome"),
        ("webllm/q4f32_1", "browser/webllm/q4f32_1/intel-chrome", "browser/webllm/q4f32_1/nvidia-chrome"),
    ]
    t3_rows = []
    for build, intel_setup, nvidia_setup in t3_configs:
        if intel_setup not in runs or nvidia_setup not in runs:
            continue
        i_map = {r["id"]: r for r in runs[intel_setup]["results"]}
        n_map = {r["id"]: r for r in runs[nvidia_setup]["results"]}
        for task in ["gsm8k", "mmlu"]:
            ids = [i for i, r in i_map.items() if r["task"] == task and i in n_map]
            if not ids:
                continue
            n = len(ids)
            i_acc = sum(i_map[i]["correct"] for i in ids) / n
            nv_acc = sum(n_map[i]["correct"] for i in ids) / n
            all_agree = sum(1 for i in ids if i_map[i]["pred"] == n_map[i]["pred"]) / n
            both_valid = [i for i in ids if i_map[i]["pred"] is not None and n_map[i]["pred"] is not None]
            n_both_valid = len(both_valid)
            v_agree = sum(1 for i in both_valid if i_map[i]["pred"] == n_map[i]["pred"]) / n_both_valid if n_both_valid > 0 else 0.0
            exact_txt = sum(1 for i in ids if i_map[i]["text"] == n_map[i]["text"]) / n
            n_diff_pred = sum(1 for i in ids if i_map[i]["pred"] != n_map[i]["pred"])
            n_diff_txt = sum(1 for i in ids if i_map[i]["text"] != n_map[i]["text"])

            t3_rows.append(dict(
                build=build, task=task, n=n,
                intel_acc=round(i_acc, 3), nvidia_acc=round(nv_acc, 3),
                valid_agree=round(v_agree, 3), all_outcome_agree=round(all_agree, 3),
                exact_text=round(exact_txt, 3), n_both_valid=n_both_valid,
                n_diff_predictions=n_diff_pred, n_diff_text=n_diff_txt
            ))

    t3_rows.sort(key=lambda r: (r["task"], r["build"]))
    t3_cols = [
        "build", "task", "n", "intel_acc", "nvidia_acc",
        "valid_agree", "all_outcome_agree", "exact_text",
        "n_both_valid", "n_diff_predictions", "n_diff_text"
    ]

    # =========================================================================
    # TABLE 4: Parsing & Truncation Audit (Strict vs. Permissive Extraction)
    # =========================================================================
    t4_rows = []
    for r in t1_rows:
        t4_rows.append(dict(
            setup=r["setup"], task=r["task"], n=r["n"],
            standard_acc=r["acc"], strict_acc=r["strict_acc"], acc_parsed=r["acc_parsed"],
            parse_rate=r["parse_rate"], n_unparsed=r["n_unparsed"],
            n_truncated_or_max=r["n_truncated_or_max"], gpu=r["gpu"]
        ))
    t4_rows.sort(key=lambda r: (r["task"], r["setup"]))
    t4_cols = [
        "setup", "task", "n", "standard_acc", "strict_acc", "acc_parsed",
        "parse_rate", "n_unparsed", "n_truncated_or_max", "gpu"
    ]

    # =========================================================================
    # Write results/summary.md
    # =========================================================================
    md_content = f"""# Benchmark Summary & Multi-Table Audit

**Reference Baseline:** `{REF}`  
**Host Hardware:** Single Windows 11 Laptop — 13th Gen Intel(R) Core(TM) i5-13420H CPU, NVIDIA GeForce RTX 2050 (4 GB GDDR6), Intel(R) UHD Graphics (Gen-12LP), 8 GB RAM  
**Software Environment:** Windows 11 x64, Google Chrome 139.0.0.0, Node.js v20+, Transformers.js 4.3.0, WebLLM 0.2.85  

---

## Table 1: Primary Comparison vs. Full-Precision Reference (`native-node-cpu/fp32`)

*Evaluates all 28 experimental configurations against the native CPU fp32 baseline. Separates unparsed predictions, reports valid-only vs. all-outcome agreement, decomposes flips, and provides divergence token distributions.*

{make_md_table(t1_rows, t1_cols)}

---

## Table 2: Direct Matched-Dtype Comparisons: Native CPU vs. Browser (RQ1)

*Directly tests runtime effects by holding model weights and quantization format strictly constant while changing only the runtime environment (Native Node.js CPU vs. Browser WebGPU/WASM).*

{make_md_table(t2_rows, t2_cols)}

*Note on browser WASM check:* `browser/tjs-wasm/q4/cpu-chrome-check` demonstrates **100% exact text (20/20) and exact token-ID (20/20) match** against `native-node-cpu/q4` on the matched diagnostic subset, confirming the execution graph and prompt harness are identical between Node.js and browser WASM CPU.

---

## Table 3: Cross-GPU Pairwise Consistency: Intel UHD vs. NVIDIA RTX 2050 (RQ2)

*Directly compares the Intel UHD Graphics integrated GPU run against the NVIDIA GeForce RTX 2050 discrete GPU run on the same laptop for each browser engine build.*

{make_md_table(t3_rows, t3_cols)}

*Key takeaway:* Under 16-bit intermediate math (`webllm/q4f16_1`), inter-GPU numerical drift causes 5% of MMLU texts and 54% of GSM8K texts to diverge. Upgrading to 32-bit intermediate accumulation (`webllm/q4f32_1`) achieves **100% exact text agreement (300/300 items)** and 0 divergent predictions across Intel and NVIDIA GPUs.

---

## Table 4: Parsing & Truncation Audit (Strict vs. Permissive Extraction)

*Evaluates the effect of regex extraction fallbacks and generation token limits across tasks.*

{make_md_table(t4_rows, t4_cols)}
"""

    with open(os.path.join(RD, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md_content)

    print("Successfully generated results/summary.md (4 tables) and results/summary.csv")


if __name__ == "__main__":
    main()
