"""라벨 재현성 비교(같은 프롬프트, 독립 세션 두 번). 지표와 기준은 계산 전에 Notion에 기록했다(2026-10-07).

주 지표: 이미지 식별 정확도. pass0 라벨 텍스트로 pass9의 300개 중 같은 이미지를 TF-IDF 코사인으로 찾는 비율(top-1, top-5).
  기준: top-1 ≥ 0.50. 우연 1/300.
보조: 차원별 식별(notes 제외, 우연의 10배 이상), 같은 이미지 쌍 대 다른 이미지 쌍의 평균 코사인, 단어 Jaccard.
TF-IDF는 표면 어휘 기준이라 동의어를 다르게 센다. 같은 모델·프롬프트의 일관성이지 타당성이 아니다.

사용: python3 src/labeling/repro_check.py --bundle-dir data/colab_bundle
"""
import argparse
import gzip
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

import text_variants as tv

DIMS = ("place_type", "environment", "activity", "implied_event", "scale")
STOP = set("a an the of in on at to and or with for from by is are was be as it its this that their there into near next than very some one two more most while being has have had".split())


def toks(s):
    return [w for w in re.findall(r"[a-z]+", s.lower()) if len(w) >= 3 and w not in STOP]


def tfidf(docs):
    df = Counter(w for d in docs for w in set(d))
    n = len(docs)
    vocab = {w: i for i, w in enumerate(sorted(df))}
    X = np.zeros((n, len(vocab)), np.float32)
    for i, d in enumerate(docs):
        for w, c in Counter(d).items():
            X[i, vocab[w]] = (1 + math.log(c)) * math.log((1 + n) / (1 + df[w])) + 0.0
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X


def ident(A, B):
    S = A @ B.T
    rank = (S > np.diag(S)[:, None]).sum(1)  # 정답보다 높은 후보 수
    return float((rank == 0).mean()), float((rank < 5).mean()), S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", default="data/colab_bundle")
    ap.add_argument("--out", default="data/results/repro_check.json")
    a = ap.parse_args()
    b = Path(a.bundle_dir)
    p0 = {r["cocoId"]: r for r in map(json.loads, gzip.open(b / "labels_pass0.jsonl.gz", "rt"))}
    p9 = {r["cocoId"]: r for r in map(json.loads, gzip.open(b / "labels_pass9.jsonl.gz", "rt"))}
    ids = sorted(set(p0) & set(p9))
    print(f"비교 이미지 {len(ids)}장 (pass0 {len(p0)}, pass9 {len(p9)}), 프롬프트 pass0: {sorted({p0[i]['prompt_version'] for i in ids})}, pass9: {sorted({p9[i]['prompt_version'] for i in ids})}")
    by0 = [tv.context_by_dim(p0[i]["labels"]) for i in ids]
    by9 = [tv.context_by_dim(p9[i]["labels"]) for i in ids]
    out = {"n": len(ids), "chance_top1": 1 / len(ids)}

    def text(by, dims):
        return [w for d in dims for ph in by[d] for w in toks(ph)]
    # 전체
    docs = [text(x, DIMS + ("notes",)) for x in by0] + [text(x, DIMS + ("notes",)) for x in by9]
    X = tfidf(docs)
    t1, t5, S = ident(X[:len(ids)], X[len(ids):])
    diag = np.diag(S)
    off = S[~np.eye(len(ids), dtype=bool)]
    out["all"] = {"top1": t1, "top5": t5, "mean_cos_same": float(diag.mean()), "mean_cos_diff": float(off.mean())}
    print(f"전체: top-1 {t1:.3f}, top-5 {t5:.3f}, 같은 이미지 코사인 {diag.mean():.3f}, 다른 이미지 {off.mean():.3f}  (기준 top-1 ≥ 0.50: {'충족' if t1 >= 0.5 else '미충족'})")
    # 차원별
    out["by_dim"] = {}
    for d in DIMS:
        docs = [text(x, (d,)) for x in by0] + [text(x, (d,)) for x in by9]
        X = tfidf(docs)
        t1, t5, S = ident(X[:len(ids)], X[len(ids):])
        j = np.mean([len(set(text(x, (d,))) & set(text(y, (d,)))) / max(1, len(set(text(x, (d,))) | set(text(y, (d,))))) for x, y in zip(by0, by9)])
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(ids))
        jn = np.mean([len(set(text(x, (d,))) & set(text(by9[k], (d,)))) / max(1, len(set(text(x, (d,))) | set(text(by9[k], (d,))))) for x, k in zip(by0, perm)])
        out["by_dim"][d] = {"top1": t1, "top5": t5, "jaccard_same": float(j), "jaccard_random_pair": float(jn), "x_chance": t1 / (1 / len(ids))}
        print(f"  {d:14s} top-1 {t1:.3f} (우연의 {t1 * len(ids):.0f}배), top-5 {t5:.3f}, Jaccard 같은 이미지 {j:.3f} / 무작위 쌍 {jn:.3f}  (기준 ≥ 우연의 10배: {'충족' if t1 * len(ids) >= 10 else '미충족'})")
    ne = np.mean([len(x["notes"]) == 0 for x in by0])
    out["notes_empty_frac_pass0"] = float(ne)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
