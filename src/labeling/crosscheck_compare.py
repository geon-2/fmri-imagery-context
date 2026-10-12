"""라벨 교차 검증 세 가지(R1, R2, R3)의 일관성 계산. 기준은 실행 전에 docs/label_crosscheck_criteria.md에 기록했다.

지표는 repro_check.py와 같다(TF-IDF 코사인 식별 정확도, 차원별 식별, Jaccard). 후보 수에 따라 달라지므로 무작위 블록으로 나눠 계산한다.
  R1  pass 0(v9) 대 pass 1(v10), 큐 앞 1,000장, 블록 약 300장
  R2  pass 0 대 pass 2, 무작위 2,000장, 블록 약 300장
  R3  pass 0 대 Claude, R2 표본 앞 200장(후보 200장), 그리고 pass 0·pass 2·Claude 세 쌍 비교
사용: python3 src/labeling/crosscheck_compare.py [--parts R1,R2,R3]
"""
import argparse
import csv
import glob
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

import repro_check as rc

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "data/labels/classify_scale"
XM = REPO / "data/labels/external_labeling/xmodel"
DIMS = ("place_type", "environment", "activity", "implied_event", "scale")
ALL = DIMS + ("notes",)


def load_pass(p):
    out = {}
    for f in sorted(glob.glob(str(BASE / f"accepted/u*_p{p}_*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        for it in d["items"]:
            out[str(it["image_id"])] = {k: [x["value"].strip() for x in v if x.get("value", "").strip()] for k, v in it["labels"].items()}
    return out


def load_claude():
    out = {}
    for f in sorted(glob.glob(str(XM / "x*_out_claude.json"))):
        d = json.load(open(f, encoding="utf-8"))
        for it in d["items"]:
            out[str(it["image_id"])] = {k: [x["value"].strip() for x in v if x.get("value", "").strip()] for k, v in it["labels"].items()}
    return out


def text(by, dims):
    return [w for d in dims for ph in by.get(d, []) for w in rc.toks(ph)]


def measure(A, B, ids):
    """A, B: {id: by_dim}. 같은 이미지 찾기 정확도(전체와 차원별), Jaccard."""
    res = {}
    for name, dims in [("all", ALL)] + [(d, (d,)) for d in DIMS]:
        docs = [text(A[i], dims) for i in ids] + [text(B[i], dims) for i in ids]
        X = rc.tfidf(docs)
        t1, t5, S = rc.ident(X[:len(ids)], X[len(ids):])
        r = {"top1": t1, "top5": t5}
        if name == "all":
            r["cos_same"] = float(np.diag(S).mean())
            r["cos_diff"] = float(S[~np.eye(len(ids), dtype=bool)].mean())
        else:
            r["jaccard"] = float(np.mean([len(set(text(A[i], dims)) & set(text(B[i], dims))) / max(1, len(set(text(A[i], dims)) | set(text(B[i], dims)))) for i in ids]))
        res[name] = r
    return res


def blocks(A, B, ids, target=300, seed=0):
    nb = max(1, round(len(ids) / target))
    perm = np.random.default_rng(seed).permutation(len(ids))
    parts = np.array_split(perm, nb)
    ms = [measure(A, B, [ids[k] for k in p]) for p in parts]
    agg = {}
    for key in ms[0]:
        agg[key] = {m: {"mean": float(np.mean([x[key][m] for x in ms])), "sd": float(np.std([x[key][m] for x in ms]))} for m in ms[0][key]}
    return {"n": len(ids), "n_blocks": nb, "block_size": [len(p) for p in parts], "res": agg}


def print_res(title, r):
    print(f"\n[{title}] n={r['n']} 블록 {r['n_blocks']}개 크기 {r['block_size']}")
    for k, v in r["res"].items():
        print(f"  {k:14s} top-1 {v['top1']['mean']:.3f}±{v['top1']['sd']:.3f}  top-5 {v['top5']['mean']:.3f}" + (f"  Jaccard {v['jaccard']['mean']:.3f}" if "jaccard" in v else f"  cos같은 {v['cos_same']['mean']:.3f} 다른 {v['cos_diff']['mean']:.3f}"))


def min_rate(P, ids):
    ok = sum(len(P[i].get("place_type", [])) >= 2 and len(P[i].get("environment", [])) >= 3 and len(P[i].get("activity", [])) >= 2 for i in ids)
    return ok / len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="R1,R2,R3")
    ap.add_argument("--out", default="data/results/crosscheck.json")
    a = ap.parse_args()
    parts = a.parts.split(",")
    p0 = load_pass(0)
    out = {}
    if "R1" in parts:
        p1 = load_pass(1)
        ids = sorted(set(p0) & set(p1), key=int)
        core = [r["cocoId"] for r in list(csv.DictReader(open(BASE / "queue_order.csv", encoding="utf-8")))[:1000]]
        ids = [i for i in core if i in p1]
        r = blocks(p0, p1, ids)
        r["min_rate_pass1"], r["min_rate_pass0"] = min_rate(p1, ids), min_rate(p0, ids)
        r["labels_per_image"] = {"pass0": float(np.mean([sum(len(v) for v in p0[i].values()) for i in ids])), "pass1": float(np.mean([sum(len(v) for v in p1[i].values()) for i in ids]))}
        out["R1"] = r
        print_res("R1 pass0(v9) 대 pass1(v10)", r)
        print(f"  최소 개수 충족: pass0 {r['min_rate_pass0']:.3f}, pass1 {r['min_rate_pass1']:.3f}; 이미지당 라벨 {r['labels_per_image']}; 이미지 {len(ids)}/1000; 기준 top-1 ≥ 0.50: {'충족' if r['res']['all']['top1']['mean'] >= 0.5 else '미충족'}")
    p2 = load_pass(2) if ("R2" in parts or "R3" in parts) else {}
    if "R2" in parts:
        samp = [r["cocoId"] for r in csv.DictReader(open(BASE / "queue_repro2000.csv", encoding="utf-8"))]
        ids = [i for i in samp if i in p2 and i in p0]
        r = blocks(p0, p2, ids)
        r["min_rate_pass2"] = min_rate(p2, ids)
        prev = {"place_type": 0.763, "environment": 0.690, "activity": 0.740, "implied_event": 0.577, "scale": 0.393}
        r["vs_prev300_within_0.10"] = {d: abs(r["res"][d]["top1"]["mean"] - v) <= 0.10 for d, v in prev.items()}
        out["R2"] = r
        print_res("R2 pass0 대 pass2", r)
        print(f"  최소 개수 충족 pass2 {r['min_rate_pass2']:.3f}; 이미지 {len(ids)}/2000; 기존 300장 대비 ±0.10: {r['vs_prev300_within_0.10']}; 기준 top-1 ≥ 0.50: {'충족' if r['res']['all']['top1']['mean'] >= 0.5 else '미충족'}")
    if "R3" in parts:
        cl = load_claude()
        samp = [r["cocoId"] for r in csv.DictReader(open(BASE / "queue_repro2000.csv", encoding="utf-8"))][:200]
        ids = [i for i in samp if i in cl and i in p0]
        r = {"n": len(ids), "n_blocks": 1, "block_size": [len(ids)], "res": {k: {m: {"mean": v, "sd": 0.0} for m, v in d.items()} for k, d in measure(p0, cl, ids).items()}}
        r["min_rate_claude"] = min_rate(cl, ids)
        out["R3"] = r
        print_res("R3 pass0(Codex) 대 Claude", r)
        print(f"  최소 개수 충족 Claude {r['min_rate_claude']:.3f}; 이미지 {len(ids)}/200; 기준 top-1 ≥ 0.50: {'충족' if r['res']['all']['top1']['mean'] >= 0.5 else '미충족'}")
        ids2 = [i for i in ids if i in p2]
        if ids2:
            tri = {"p0_p2": measure(p0, p2, ids2), "p0_claude": measure(p0, cl, ids2), "p2_claude": measure(p2, cl, ids2)}
            out["R3"]["triplet"] = {"n": len(ids2), **tri}
            print(f"\n[R3 세 쌍 비교] n={len(ids2)}")
            for k, v in tri.items():
                print(f"  {k:10s} 전체 top-1 {v['all']['top1']:.3f} | " + ", ".join(f"{d} {v[d]['top1']:.2f}" for d in DIMS))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
