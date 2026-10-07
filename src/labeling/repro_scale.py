"""공간 규모(scale)가 재현성에서 가장 낮은 원인 구분(기술용). 규칙과 기준은 계산 전에 정했다(2026-10-08).

배경: 차원별 식별(TF-IDF)에서 scale 0.393이 가장 낮다. 후보 원인 두 가지.
  (a) 표면 어휘 변동: 같은 틀(close/medium/wide)을 다른 말로 쓴다 → 틀 수준으로 거칠게 묶으면 일치가 높다.
  (b) 실제 불안정: 거칠게 묶어도 두 pass의 틀이 다르다.
거친 틀(framing) 규칙(키워드, 구 단위, 먼저 맞는 것): 
  close  : close, tight, extreme, macro, detail, tabletop, portrait, headshot
  wide   : wide, distant, far, panoramic, expansive, landscape, aerial, establishing
  medium : medium, mid, room-scale, waist, half
  그 외  : other
이미지의 틀 = 첫 구(가장 확신 높은 구)의 틀. 보조: 구 전체에서 나온 틀 집합의 Jaccard.
판정(결과 보기 전): 틀 일치 Cohen's kappa ≥ 0.60 → (a) 어휘 변동 쪽, < 0.40 → (b) 불안정 쪽, 사이 → 불확정. 
other 비율이 30% 넘으면 규칙이 어휘를 못 덮은 것이라 판정 보류.
비교로 같은 방식의 kappa를 다른 차원에는 못 낸다(틀 정의가 scale 전용). 불일치 이미지는 직접 보도록 목록을 남긴다.
"""
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np

import repro_check as rc
import text_variants as tv

RULES = [("close", "close tight extreme macro detail tabletop portrait headshot"),
         ("wide", "wide distant far panoramic expansive landscape aerial establishing"),
         ("medium", "medium mid room-scale waist half")]


def framing(phrase):
    p = phrase.lower()
    for name, kws in RULES:
        if any(k in p for k in kws.split()):
            return name
    return "other"


def kappa(a, b):
    labs = sorted(set(a) | set(b))
    po = np.mean([x == y for x, y in zip(a, b)])
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[l] / len(a) * cb[l] / len(b) for l in labs)
    return float((po - pe) / (1 - pe + 1e-12)), float(po)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", default="data/colab_bundle")
    ap.add_argument("--out", default="data/results/repro_scale.json")
    a = ap.parse_args()
    b = Path(a.bundle_dir)
    p0 = {r["cocoId"]: r for r in map(json.loads, gzip.open(b / "labels_pass0.jsonl.gz", "rt"))}
    p9 = {r["cocoId"]: r for r in map(json.loads, gzip.open(b / "labels_pass9.jsonl.gz", "rt"))}
    ids = sorted(set(p0) & set(p9))
    def first(r):
        v = sorted(r["labels"].get("scale", []), key=lambda x: -x[1])
        return v[0][0] if v else ""
    def sets(r):
        return {framing(v[0]) for v in r["labels"].get("scale", [])}
    f0 = [framing(first(p0[i])) for i in ids]
    f9 = [framing(first(p9[i])) for i in ids]
    k, po = kappa(f0, f9)
    js = float(np.mean([len(sets(p0[i]) & sets(p9[i])) / max(1, len(sets(p0[i]) | sets(p9[i]))) for i in ids]))
    oth = float(np.mean([x == "other" for x in f0 + f9]))
    dist0, dist9 = Counter(f0), Counter(f9)
    conf = Counter(zip(f0, f9))
    print(f"이미지 {len(ids)}장, 첫 구 틀 일치율 {po:.3f}, kappa {k:.3f}, 틀 집합 Jaccard {js:.3f}, other 비율 {oth:.3f}")
    print("분포 pass0", dict(dist0), "pass9", dict(dist9))
    print("혼동(pass0→pass9)", {f"{x}->{y}": c for (x, y), c in sorted(conf.items())})
    verdict = "판정 보류(other>30%)" if oth > 0.3 else ("(a) 어휘 변동 쪽" if k >= 0.6 else "(b) 불안정 쪽" if k < 0.4 else "불확정")
    print("판정:", verdict)
    dis = [i for i, x, y in zip(ids, f0, f9) if x != y]
    ex = [{"cocoId": i, "pass0": first(p0[i]), "pass9": first(p9[i])} for i in dis]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n": len(ids), "kappa": k, "agree": po, "jaccard_sets": js, "other_frac": oth, "dist0": dist0, "dist9": dist9,
               "confusion": {f"{x}->{y}": c for (x, y), c in conf.items()}, "verdict": verdict, "disagree": ex}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"불일치 {len(dis)}장 → {a.out} (직접 확인용)")


if __name__ == "__main__":
    main()
