"""Compare every result file against reference setups and generate multi-table audit summaries.

Usage: python analyze.py [results_dir] [reference_setup_substring]
Writes:
  results/summary_parsing.csv
  results/summary_matched.csv
  results/summary_cross_adapter.csv
  results/summary_primary.csv
  results/summary.csv (copy of summary_primary.csv)
  results/summary_holm.csv
  results/summary.md (all 5 tables formatted)
"""
import glob, json, os, sys, random, csv, re, statistics, platform, hashlib
import scipy
import transformers
from scipy.stats import binomtest

RD = sys.argv[1] if len(sys.argv) > 1 else "results"
REF = sys.argv[2] if len(sys.argv) > 2 else "native-node-cpu/fp32"

# -----------------------------------------------------------------------------
# Tokenizer loading with strict error handling & explicit revision pinning
# -----------------------------------------------------------------------------
TOKENIZER_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
TOKENIZER = None
TOKENIZER_NAME = None
TOKENIZER_COMMIT = None
TOKENIZER_JSON_SHA256 = None

try:
    from transformers import AutoTokenizer
    repo_id = "Qwen/Qwen2.5-0.5B-Instruct"
    try:
        TOKENIZER = AutoTokenizer.from_pretrained(
            repo_id, revision=TOKENIZER_REVISION, local_files_only=True
        )
    except Exception:
        TOKENIZER = AutoTokenizer.from_pretrained(
            repo_id, revision=TOKENIZER_REVISION
        )
    TOKENIZER_NAME = getattr(TOKENIZER, "name_or_path", repo_id)
    TOKENIZER_COMMIT = TOKENIZER_REVISION

    # Check cached tokenizer.json SHA-256 checksum
    snap_dir = os.path.expanduser(f"~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/{TOKENIZER_REVISION}")
    tok_json_path = os.path.join(snap_dir, "tokenizer.json")
    if os.path.exists(tok_json_path):
        with open(tok_json_path, "rb") as fp:
            TOKENIZER_JSON_SHA256 = hashlib.sha256(fp.read()).hexdigest()
    else:
        TOKENIZER_JSON_SHA256 = "unavailable (local snapshot not found)"
except Exception as e:
    raise RuntimeError(
        f"CRITICAL: Failed to load tokenizer for 'Qwen/Qwen2.5-0.5B-Instruct' with revision '{TOKENIZER_REVISION}'. "
        f"A valid AutoTokenizer is strictly required; whitespace fallback is disabled. Error: {e}"
    )


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
    """Strict MMLU extraction: requires standalone letter A-D via word boundaries."""
    m = re.search(r"\b([ABCD])\b", text)
    return m.group(1) if m else None


def strict_gsm(text):
    """Strict GSM8K extraction: requires explicit 'answer is <num>' pattern."""
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
    """Return index of first differing token ID between sequences a and b."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def format_p(p):
    if p < 0.0001:
        return "< 1e-4"
    return str(round(p, 4))


def format_frac(num, den):
    """Format as 'num/den (pct%)'."""
    pct = (num / den * 100.0) if den > 0 else 0.0
    return f"{num}/{den} ({pct:.2f}%)"


def format_median(divs):
    if not divs:
        return ""
    m = statistics.median(divs)
    return int(m) if isinstance(m, float) and m.is_integer() else round(m, 1)


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


def make_md_table(rows, cols):
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    runs = load()
    if REF not in runs:
        print("reference not found:", REF, "available:", list(runs))
        return
    ref = {r["id"]: r for r in runs[REF]["results"]}

    # =========================================================================
    # TABLE 1 / CSV 1: Parsing & Truncation Audit
    # =========================================================================
    t_parsing_rows = []
    t_parsing_md_rows = []

    for setup, d in sorted(runs.items()):
        res = {r["id"]: r for r in d["results"]}
        for task in ["gsm8k", "mmlu"]:
            ids = [i for i, r in res.items() if r["task"] == task]
            if not ids:
                continue
            n = len(ids)
            n_correct = sum(res[i]["correct"] for i in ids)
            task_acc = round(n_correct / n, 4)

            n_unparsed = sum(1 for i in ids if res[i]["pred"] is None)
            parse_rate = round((n - n_unparsed) / n, 4)
            n_valid = n - n_unparsed
            c_parsed = sum(res[i]["correct"] for i in ids if res[i]["pred"] is not None)
            acc_parsed_val = round(c_parsed / n_valid, 4) if n_valid > 0 else 0.0

            # Strict extraction audit:
            # strict_n_parsed counts responses matching the restrictive regex regardless of correctness
            # strict_n_correct counts responses matching the restrictive regex that match gold
            strict_n_parsed = 0
            strict_n_correct = 0
            for i in ids:
                txt = res[i].get("text", "")
                sp = strict_mmlu(txt) if task == "mmlu" else strict_gsm(txt)
                if sp is not None:
                    strict_n_parsed += 1
                if sp == res[i]["gold"]:
                    strict_n_correct += 1
            strict_parse_rate = round(strict_n_parsed / n, 4)
            strict_acc = round(strict_n_correct / n, 4)

            # Cap tracking
            has_raw = res[ids[0]].get("ids") is not None
            cap_source = "raw_token_ids" if has_raw else "retokenized_text"
            cap_limit = 5 if task == "mmlu" else 320
            n_at_token_cap = 0
            for i in ids:
                if has_raw:
                    tok_len = len(res[i]["ids"])
                else:
                    tok_len = len(TOKENIZER.encode(res[i].get("text", ""), add_special_tokens=False))
                if tok_len >= cap_limit:
                    n_at_token_cap += 1

            row_csv = dict(
                setup=setup, task=task, n=n,
                n_correct=n_correct, task_acc=task_acc,
                n_unparsed=n_unparsed, parse_rate=parse_rate,
                acc_parsed_num_den=f"{c_parsed}/{n_valid}",
                acc_parsed=acc_parsed_val,
                strict_n_parsed=strict_n_parsed,
                strict_parse_rate=strict_parse_rate,
                strict_n_correct=strict_n_correct,
                strict_acc=strict_acc,
                n_at_token_cap=n_at_token_cap, cap_source=cap_source
            )
            t_parsing_rows.append(row_csv)

            row_md = dict(
                setup=setup, task=task, n=n,
                task_acc=task_acc,
                n_unparsed=n_unparsed, parse_rate=parse_rate,
                acc_parsed=format_frac(c_parsed, n_valid),
                strict_parse_rate=format_frac(strict_n_parsed, n),
                strict_acc=format_frac(strict_n_correct, n),
                n_at_token_cap=n_at_token_cap, cap_source=cap_source,
                gpu=get_gpu_label(setup, d["meta"])
            )
            t_parsing_md_rows.append(row_md)

    t_parsing_rows.sort(key=lambda r: (r["task"], r["setup"]))
    t_parsing_md_rows.sort(key=lambda r: (r["task"], r["setup"]))
    write_csv(os.path.join(RD, "summary_parsing.csv"), t_parsing_rows)

    # =========================================================================
    # TABLE 2 / CSV 2: Direct Matched-Dtype Comparisons (RQ1)
    # =========================================================================
    t2_pairs = [
        ("browser/tjs-webgpu/fp16/intel-chrome", "native-node-cpu/fp16"),
        ("browser/tjs-webgpu/fp16/nvidia-chrome", "native-node-cpu/fp16"),
        ("browser/tjs-webgpu/fp16/nvidia-chrome-repeat", "native-node-cpu/fp16"),
        ("browser/tjs-webgpu/q4f16/intel-chrome", "native-node-cpu/q4f16"),
        ("browser/tjs-webgpu/q4f16/nvidia-chrome", "native-node-cpu/q4f16"),
        ("browser/tjs-wasm/q4/cpu-chrome-check", "native-node-cpu/q4"),
    ]
    t_matched_rows = []
    t_matched_md_rows = []

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
            b_correct = sum(b_map[i]["correct"] for i in ids)
            n_correct = sum(n_map[i]["correct"] for i in ids)
            b_acc = round(b_correct / n, 4)
            n_acc = round(n_correct / n, 4)

            both_valid = [i for i in ids if b_map[i]["pred"] is not None and n_map[i]["pred"] is not None]
            n_both_valid = len(both_valid)
            v_agree_count = sum(1 for i in both_valid if b_map[i]["pred"] == n_map[i]["pred"])
            v_agree_pct = round(v_agree_count / n_both_valid, 4) if n_both_valid > 0 else 0.0

            all_agree_count = sum(1 for i in ids if b_map[i]["pred"] == n_map[i]["pred"])
            all_agree_pct = round(all_agree_count / n, 4)

            exact_txt_count = sum(1 for i in ids if b_map[i]["text"] == n_map[i]["text"])
            exact_txt_pct = round(exact_txt_count / n, 4)

            exact_ids_count = sum(1 for i in ids if b_map[i].get("ids") is not None and b_map[i].get("ids") == n_map[i].get("ids"))
            exact_ids_pct = round(exact_ids_count / n, 4)

            # Token divergence on raw tokens across ALL paired items
            divs = []
            for i in ids:
                fd = first_div(b_map[i]["ids"], n_map[i]["ids"])
                if fd is not None:
                    divs.append(fd)
            med_div = format_median(divs)
            n_div_0 = sum(1 for d in divs if d == 0)
            n_div_1 = sum(1 for d in divs if d == 1)

            row_csv = dict(
                browser_setup=b_setup, native_ref_setup=n_setup, task=task, n=n,
                browser_acc=b_acc, native_acc=n_acc,
                valid_agree_num_den=f"{v_agree_count}/{n_both_valid}",
                valid_agree=v_agree_pct,
                all_outcome_agree_num_den=f"{all_agree_count}/{n}",
                all_outcome_agree=all_agree_pct,
                exact_text_num_den=f"{exact_txt_count}/{n}",
                exact_text=exact_txt_pct,
                exact_token_ids_num_den=f"{exact_ids_count}/{n}",
                exact_token_ids=exact_ids_pct,
                median_first_div_token=med_div, n_diverged=len(divs),
                n_div_at_0=n_div_0, n_div_at_1=n_div_1,
                div_source="raw_token_ids", n_both_valid=n_both_valid
            )
            t_matched_rows.append(row_csv)

            row_md = dict(
                browser_setup=b_setup, native_ref_setup=n_setup, task=task, n=n,
                browser_acc=b_acc, native_acc=n_acc,
                valid_agree=format_frac(v_agree_count, n_both_valid),
                all_outcome_agree=format_frac(all_agree_count, n),
                exact_text=format_frac(exact_txt_count, n),
                exact_token_ids=format_frac(exact_ids_count, n),
                median_first_div_token=med_div, n_diverged=len(divs),
                n_div_at_0=n_div_0, n_div_at_1=n_div_1
            )
            t_matched_md_rows.append(row_md)

    t_matched_rows.sort(key=lambda r: (r["task"], r["browser_setup"]))
    t_matched_md_rows.sort(key=lambda r: (r["task"], r["browser_setup"]))
    write_csv(os.path.join(RD, "summary_matched.csv"), t_matched_rows)

    # =========================================================================
    # TABLE 3 / CSV 3: Cross-Adapter Pairwise Consistency (RQ2: Intel vs. NVIDIA)
    # =========================================================================
    t3_configs = [
        ("tjs-webgpu/fp16", "browser/tjs-webgpu/fp16/intel-chrome", "browser/tjs-webgpu/fp16/nvidia-chrome"),
        ("tjs-webgpu/q4f16", "browser/tjs-webgpu/q4f16/intel-chrome", "browser/tjs-webgpu/q4f16/nvidia-chrome"),
        ("webllm/q4f16_1", "browser/webllm/q4f16_1/intel-chrome", "browser/webllm/q4f16_1/nvidia-chrome"),
        ("webllm/q4f32_1", "browser/webllm/q4f32_1/intel-chrome", "browser/webllm/q4f32_1/nvidia-chrome"),
    ]
    t_cross_rows = []
    t_cross_md_rows = []

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
            i_acc = round(sum(i_map[i]["correct"] for i in ids) / n, 4)
            nv_acc = round(sum(n_map[i]["correct"] for i in ids) / n, 4)

            both_valid = [i for i in ids if i_map[i]["pred"] is not None and n_map[i]["pred"] is not None]
            n_both_valid = len(both_valid)
            v_agree_count = sum(1 for i in both_valid if i_map[i]["pred"] == n_map[i]["pred"])
            v_agree_pct = round(v_agree_count / n_both_valid, 4) if n_both_valid > 0 else 0.0

            all_agree_count = sum(1 for i in ids if i_map[i]["pred"] == n_map[i]["pred"])
            all_agree_pct = round(all_agree_count / n, 4)

            exact_txt_count = sum(1 for i in ids if i_map[i]["text"] == n_map[i]["text"])
            exact_txt_pct = round(exact_txt_count / n, 4)

            n_diff_pred = sum(1 for i in ids if i_map[i]["pred"] != n_map[i]["pred"])
            n_diff_txt = sum(1 for i in ids if i_map[i]["text"] != n_map[i]["text"])

            # Token divergence
            has_raw = (i_map[ids[0]].get("ids") is not None) and (n_map[ids[0]].get("ids") is not None)
            div_source = "raw_token_ids" if has_raw else "retokenized_text"
            divs = []
            for i in ids:
                if has_raw:
                    fd = first_div(i_map[i]["ids"], n_map[i]["ids"])
                else:
                    t1 = TOKENIZER.encode(i_map[i].get("text", ""), add_special_tokens=False)
                    t2 = TOKENIZER.encode(n_map[i].get("text", ""), add_special_tokens=False)
                    fd = first_div(t1, t2)
                if fd is not None:
                    divs.append(fd)
            med_div = format_median(divs)
            n_div_0 = sum(1 for d in divs if d == 0)
            n_div_1 = sum(1 for d in divs if d == 1)

            row_csv = dict(
                build=build, task=task, n=n,
                intel_acc=i_acc, nvidia_acc=nv_acc,
                valid_agree_num_den=f"{v_agree_count}/{n_both_valid}",
                valid_agree=v_agree_pct,
                all_outcome_agree_num_den=f"{all_agree_count}/{n}",
                all_outcome_agree=all_agree_pct,
                exact_text_num_den=f"{exact_txt_count}/{n}",
                exact_text=exact_txt_pct,
                n_both_valid=n_both_valid,
                n_diff_predictions=n_diff_pred, n_diff_text=n_diff_txt,
                median_first_div_token=med_div, n_diverged=len(divs),
                n_div_at_0=n_div_0, n_div_at_1=n_div_1,
                div_source=div_source
            )
            t_cross_rows.append(row_csv)

            row_md = dict(
                build=build, task=task, n=n,
                intel_acc=i_acc, nvidia_acc=nv_acc,
                valid_agree=format_frac(v_agree_count, n_both_valid),
                all_outcome_agree=format_frac(all_agree_count, n),
                exact_text=format_frac(exact_txt_count, n),
                n_both_valid=n_both_valid,
                n_diff_predictions=n_diff_pred, n_diff_text=n_diff_txt,
                median_first_div_token=med_div, n_diverged=len(divs),
                div_source=div_source
            )
            t_cross_md_rows.append(row_md)

    t_cross_rows.sort(key=lambda r: (r["task"], r["build"]))
    t_cross_md_rows.sort(key=lambda r: (r["task"], r["build"]))
    write_csv(os.path.join(RD, "summary_cross_adapter.csv"), t_cross_rows)

    # =========================================================================
    # TABLE 4 / CSV 4: Primary Comparison vs. Full-Precision Reference (fp32)
    # =========================================================================
    t_primary_rows = []
    t_primary_md_rows = []

    for setup, d in sorted(runs.items()):
        res = {r["id"]: r for r in d["results"]}
        for task in ["gsm8k", "mmlu"]:
            ids = [i for i, r in res.items() if r["task"] == task and i in ref]
            if not ids:
                continue
            n = len(ids)
            acc = round(sum(res[i]["correct"] for i in ids) / n, 4)
            racc = round(sum(ref[i]["correct"] for i in ids) / n, 4)

            # Unparsed and valid
            n_unparsed = sum(1 for i in ids if res[i]["pred"] is None)
            parse_rate = round((n - n_unparsed) / n, 4)
            n_valid = n - n_unparsed
            c_parsed = sum(res[i]["correct"] for i in ids if res[i]["pred"] is not None)
            acc_parsed_val = round(c_parsed / n_valid, 4) if n_valid > 0 else 0.0

            # Strict extraction accuracy
            strict_correct = 0
            for i in ids:
                txt = res[i].get("text", "")
                sp = strict_mmlu(txt) if task == "mmlu" else strict_gsm(txt)
                if sp == res[i]["gold"]:
                    strict_correct += 1
            strict_acc = round(strict_correct / n, 4)

            # Agreement
            both_valid = [i for i in ids if res[i]["pred"] is not None and ref[i]["pred"] is not None]
            n_both_valid = len(both_valid)
            v_agree_count = sum(1 for i in both_valid if res[i]["pred"] == ref[i]["pred"])
            v_agree_pct = round(v_agree_count / n_both_valid, 4) if n_both_valid > 0 else 0.0

            all_agree_count = sum(1 for i in ids if res[i]["pred"] == ref[i]["pred"])
            all_agree_pct = round(all_agree_count / n, 4)
            n_joint_invalid = sum(1 for i in ids if res[i]["pred"] is None and ref[i]["pred"] is None)

            # Flips
            c2w = sum(1 for i in ids if ref[i]["correct"] and not res[i]["correct"])
            w2c = sum(1 for i in ids if not ref[i]["correct"] and res[i]["correct"])
            unparsed_ref_to_correct = sum(1 for i in ids if ref[i]["pred"] is None and res[i]["correct"])
            correct_to_unparsed_test = sum(1 for i in ids if ref[i]["correct"] and res[i]["pred"] is None)
            mcnemar_p_val = binomtest(c2w, c2w + w2c, 0.5).pvalue if (c2w + w2c) > 0 else 1.0
            mcnemar_p = format_p(mcnemar_p_val)

            # Exact text
            exact_txt_count = sum(1 for i in ids if res[i]["text"] == ref[i]["text"])
            exact_txt_pct = round(exact_txt_count / n, 4)

            # Divergence tokens across ALL paired items
            has_raw = (res[ids[0]].get("ids") is not None) and (ref[ids[0]].get("ids") is not None)
            div_source = "raw_token_ids" if has_raw else "retokenized_text"
            divs = []
            for i in ids:
                if has_raw:
                    fd = first_div(res[i]["ids"], ref[i]["ids"])
                else:
                    t1 = TOKENIZER.encode(res[i].get("text", ""), add_special_tokens=False)
                    t2 = TOKENIZER.encode(ref[i].get("text", ""), add_special_tokens=False)
                    fd = first_div(t1, t2)
                if fd is not None:
                    divs.append(fd)
            med_div = format_median(divs)
            n_div_0 = sum(1 for d in divs if d == 0)
            n_div_1 = sum(1 for d in divs if d == 1)

            # Truncation cap count
            cap_limit = 5 if task == "mmlu" else 320
            n_at_token_cap = 0
            for i in ids:
                if res[i].get("ids") is not None:
                    tok_len = len(res[i]["ids"])
                else:
                    tok_len = len(TOKENIZER.encode(res[i].get("text", ""), add_special_tokens=False))
                if tok_len >= cap_limit:
                    n_at_token_cap += 1

            lo, hi = boot_ci([int(res[i]["pred"] == ref[i]["pred"]) for i in ids])

            row_csv = dict(
                setup=setup, task=task, n=n,
                acc=acc, ref_acc=racc, strict_acc=strict_acc,
                n_unparsed=n_unparsed, parse_rate=parse_rate,
                acc_parsed_num_den=f"{c_parsed}/{n_valid}", acc_parsed=acc_parsed_val,
                valid_agree_num_den=f"{v_agree_count}/{n_both_valid}", valid_agree=v_agree_pct,
                all_outcome_agree_num_den=f"{all_agree_count}/{n}", all_outcome_agree=all_agree_pct,
                agree_ci=f"[{lo:.3f}, {hi:.3f}]", n_both_valid=n_both_valid, n_joint_invalid=n_joint_invalid,
                flips_c2w=c2w, flips_w2c=w2c,
                unparsed_ref_to_correct=unparsed_ref_to_correct, correct_to_unparsed_test=correct_to_unparsed_test,
                mcnemar_p_unadjusted=mcnemar_p_val, mcnemar_p=mcnemar_p,
                exact_text_num_den=f"{exact_txt_count}/{n}", exact_text=exact_txt_pct,
                median_first_div_token=med_div, n_diverged=len(divs),
                n_div_at_0=n_div_0, n_div_at_1=n_div_1,
                div_source=div_source, n_at_token_cap=n_at_token_cap,
                gpu=get_gpu_label(setup, d["meta"])
            )
            t_primary_rows.append(row_csv)

            row_md = dict(
                setup=setup, task=task, n=n,
                acc=acc, ref_acc=racc, strict_acc=strict_acc,
                n_unparsed=n_unparsed, parse_rate=parse_rate,
                acc_parsed=format_frac(c_parsed, n_valid),
                valid_agree=format_frac(v_agree_count, n_both_valid),
                all_outcome_agree=format_frac(all_agree_count, n),
                agree_ci=f"[{lo:.3f}, {hi:.3f}]", n_both_valid=n_both_valid, n_joint_invalid=n_joint_invalid,
                flips_c2w=c2w, flips_w2c=w2c,
                unparsed_ref_to_correct=unparsed_ref_to_correct, correct_to_unparsed_test=correct_to_unparsed_test,
                mcnemar_p=mcnemar_p, exact_text=format_frac(exact_txt_count, n),
                median_first_div_token=med_div, n_diverged=len(divs),
                n_div_at_0=n_div_0, n_div_at_1=n_div_1,
                div_source=div_source, n_at_token_cap=n_at_token_cap,
                gpu=get_gpu_label(setup, d["meta"])
            )
            t_primary_md_rows.append(row_md)

    t_primary_rows.sort(key=lambda r: (r["task"], r["setup"]))
    t_primary_md_rows.sort(key=lambda r: (r["task"], r["setup"]))

    write_csv(os.path.join(RD, "summary_primary.csv"), t_primary_rows)
    write_csv(os.path.join(RD, "summary.csv"), t_primary_rows)

    # =========================================================================
    # TABLE 5 / CSV 5: 22-Comparison Holm Step-Down Multiple-Testing Table
    # =========================================================================
    t_holm_candidates = [
        r for r in t_primary_rows
        if r["setup"] != REF and "repeat" not in r["setup"] and "check" not in r["setup"]
    ]
    t_holm_candidates.sort(key=lambda r: r["mcnemar_p_unadjusted"])
    m_holm = len(t_holm_candidates)  # 22

    t_holm_rows = []
    t_holm_md_rows = []
    cum_max = 0.0

    for i, r in enumerate(t_holm_candidates):
        rank = i + 1
        multiplier = m_holm - i
        adj_p = min(1.0, r["mcnemar_p_unadjusted"] * multiplier)
        cum_max = max(cum_max, adj_p)
        sig = cum_max < 0.05

        row_csv = dict(
            setup=r["setup"], task=r["task"],
            flips_c2w=r["flips_c2w"], flips_w2c=r["flips_w2c"],
            mcnemar_p_unadjusted=r["mcnemar_p_unadjusted"],
            holm_rank=rank,
            mcnemar_p_holm=cum_max,
            significant_at_0_05=sig
        )
        t_holm_rows.append(row_csv)

        row_md = dict(
            setup=r["setup"], task=r["task"],
            flips=f"c2w={r['flips_c2w']}, w2c={r['flips_w2c']}",
            mcnemar_p_unadjusted=f"{r['mcnemar_p_unadjusted']:.6e}" if r['mcnemar_p_unadjusted'] < 0.001 else f"{r['mcnemar_p_unadjusted']:.4f}",
            holm_rank=rank,
            mcnemar_p_holm=f"{cum_max:.6e}" if cum_max < 0.001 else f"{cum_max:.4f}",
            significant_at_0_05="Yes" if sig else "No"
        )
        t_holm_md_rows.append(row_md)

    write_csv(os.path.join(RD, "summary_holm.csv"), t_holm_rows)

    # =========================================================================
    # Write results/summary.md containing all 5 tables
    # =========================================================================
    p_cols = [
        "setup", "task", "n", "task_acc", "n_unparsed", "parse_rate",
        "acc_parsed", "strict_parse_rate", "strict_acc", "n_at_token_cap", "cap_source", "gpu"
    ]
    m_cols = [
        "browser_setup", "native_ref_setup", "task", "n",
        "browser_acc", "native_acc", "valid_agree", "all_outcome_agree",
        "exact_text", "exact_token_ids", "median_first_div_token", "n_diverged",
        "n_div_at_0", "n_div_at_1"
    ]
    c_cols = [
        "build", "task", "n", "intel_acc", "nvidia_acc",
        "valid_agree", "all_outcome_agree", "exact_text",
        "n_both_valid", "n_diff_predictions", "n_diff_text",
        "median_first_div_token", "n_diverged", "div_source"
    ]
    t1_cols = [
        "setup", "task", "n", "acc", "ref_acc", "strict_acc",
        "n_unparsed", "parse_rate", "acc_parsed",
        "valid_agree", "all_outcome_agree", "agree_ci", "n_both_valid", "n_joint_invalid",
        "flips_c2w", "flips_w2c", "unparsed_ref_to_correct", "correct_to_unparsed_test",
        "mcnemar_p", "exact_text", "median_first_div_token", "n_diverged",
        "n_div_at_0", "n_div_at_1", "div_source", "n_at_token_cap", "gpu"
    ]
    h_cols = [
        "setup", "task", "flips", "mcnemar_p_unadjusted", "holm_rank", "mcnemar_p_holm", "significant_at_0_05"
    ]

    py_ver = platform.python_version()
    scipy_ver = scipy.__version__
    trans_ver = transformers.__version__

    md_content = f"""# Benchmark Summary & Multi-Table Static Audit

**Reference Baseline:** `{REF}`  
**Host Hardware:** Single Windows 11 Laptop — 13th Gen Intel(R) Core(TM) i5-13420H CPU, NVIDIA GeForce RTX 2050 (4 GB GDDR6), Intel(R) UHD Graphics (Gen-12LP), 8 GB RAM  
**Software Environment:** Windows 11 x64, Google Chrome 139.0.0.0, Node.js v20+, Transformers.js 4.3.0, WebLLM 0.2.85  
**Analysis Execution Environment:** Python `{py_ver}`, scipy `{scipy_ver}`, transformers `{trans_ver}`  
**Tokenizer Audit:** `{TOKENIZER_NAME}` (Pinned Revision: `{TOKENIZER_REVISION}`, `tokenizer.json` SHA-256: `{TOKENIZER_JSON_SHA256}`)  

---

## Table 1: Primary Comparison vs. Full-Precision Reference (`native-node-cpu/fp32`)

*Evaluates all 28 experimental configurations against the native CPU fp32 baseline. Separates unparsed predictions, reports valid-only and all-outcome agreement with explicit fractions, decomposes flips, and tracks token sequence divergence across all paired items.*

*Statistical Note on Confidence Intervals:* Percentile bootstrap confidence intervals (`agree_ci`) reflect item-resampling variability across the fixed 300-question set (where an observed 0% sample yields a degenerate `[0.000, 0.000]` percentile interval) and do not capture variance across devices, drivers, or sessions.

{make_md_table(t_primary_md_rows, t1_cols)}

---

## Table 2: Direct Matched-Dtype Comparisons: Native CPU vs. Browser (RQ1)

*Evaluates runtime effects by matching model repository identifiers and dtypes between Native Node.js CPU and Browser WebGPU/WASM across all paired item evaluations. (Note: Repository IDs and requested dtypes were strictly matched; file SHA-256 hashes were not pinned across the Oct 8–9 and Oct 28 run dates).*

{make_md_table(t_matched_md_rows, m_cols)}

*Browser WASM CPU Verification:* `browser/tjs-wasm/q4/cpu-chrome-check` achieves **10/10 (100.00%) exact text and 10/10 (100.00%) exact token-ID match** on GSM8K, and **10/10 (100.00%) exact text and 10/10 (100.00%) exact token-ID match** on MMLU (with 8/8 valid parsed agreement) against `native-node-cpu/q4` on the matched diagnostic subset.

---

## Table 3: Cross-Adapter Pairwise Consistency: Intel-Labelled vs. NVIDIA-Labelled Runs (RQ2)

*Directly compares the Intel-labelled integrated GPU run against the NVIDIA-labelled discrete GPU run on the same laptop for each browser engine build. Adapter labels reflect a main-page `navigator.gpu.requestAdapter()` check after switching Windows Graphics Settings rather than internal worker attestation.*

{make_md_table(t_cross_md_rows, c_cols)}

*Key takeaway:* Under `webllm/q4f16_1` (where MLC `model_dtype` is `float16`), cross-adapter numerical drift produces divergent text on 10/200 (5.0%) MMLU items and 54/100 (54.0%) GSM8K items. Under `webllm/q4f32_1` (where MLC `model_dtype` is `float32`), cross-adapter consistency reaches **100.00% exact text agreement (200/200 MMLU and 100/100 GSM8K)** with 0 divergent predictions.

---

## Table 4: Parsing & Truncation Audit (Strict vs. Permissive Extraction)

*Evaluates regex extraction parse rates, accuracy restricted to valid parsed responses, strict-extraction parse rates (`strict_parse_rate`) and accuracy (`strict_acc` requiring both a restrictive regex match—standalone `r'\\b([ABCD])\\b'` on MMLU or `r'answer is\\s*([\\-0-9,\\.]+)'` on GSM8K—and a correct answer), and token limit saturation (5 tokens for MMLU, 320 tokens for GSM8K).*

{make_md_table(t_parsing_md_rows, p_cols)}

*Unparsed Response Findings:* In `native-node-cpu/fp32` on MMLU, 39/200 items failed letter-only extraction: 38 saturated the 5-token limit without emitting a standalone letter, and 1 terminated early at 3 tokens (`"WEP"` + EOS, ID `mmlu-1554`).

---

## Table 5: Holm Step-Down Multiple-Testing Correction (22 Hypotheses)

*Controls family-wise error rate across all 22 primary experimental comparisons against `native-node-cpu/fp32` (11 full runs $\\times$ 2 tasks, excluding the reference run, repeatability check, and diagnostic 20-item subset).*

{make_md_table(t_holm_md_rows, h_cols)}
"""

    with open(os.path.join(RD, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md_content)

    print("Successfully generated all 5 CSV summaries and results/summary.md")


if __name__ == "__main__":
    main()
