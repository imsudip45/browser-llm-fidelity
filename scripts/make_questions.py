"""Rebuild data/questions_v1.json exactly (seed 2026).
Needs: pip install pandas pyarrow
"""
import json, random, urllib.request, os
import pandas as pd

os.makedirs("data", exist_ok=True)
SRC = {
    "mmlu": "https://huggingface.co/api/datasets/cais/mmlu/parquet/all/test/0.parquet",
    "gsm8k": "https://huggingface.co/api/datasets/openai/gsm8k/parquet/main/test/0.parquet",
}
for k, u in SRC.items():
    p = f"data/{k}.parquet"
    if not os.path.exists(p):
        urllib.request.urlretrieve(u, p)
m = pd.read_parquet("data/mmlu.parquet"); g = pd.read_parquet("data/gsm8k.parquet")
rng = random.Random(2026)
items = []
for s in sorted(m.subject.unique()):
    idx = list(m[m.subject == s].index); rng.shuffle(idx)
    for i in idx[:4]:
        r = m.loc[i]
        items.append(dict(id=f"mmlu-{i}", subject=s, question=r.question, choices=list(r.choices), answer="ABCD"[int(r.answer)]))
rng.shuffle(items); items = items[:200]
gi = list(range(len(g))); rng.shuffle(gi)
gs = []
for i in gi[:100]:
    r = g.loc[i]
    gs.append(dict(id=f"gsm8k-{i}", question=r.question, answer=r.answer.split("####")[-1].strip().replace(",", "")))
json.dump(dict(version="v1-seed2026", mmlu=items, gsm8k=gs), open("data/questions_v1.json", "w"), ensure_ascii=False)
print(len(items), len(gs))
