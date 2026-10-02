"""직접 확인용 표본 100장: 무작위 80장 + COCO-Stuff와 어긋난 곳에서 뽑은 표적 20장.

표적은 결과를 보기 전에 아래 규칙으로 고정했다(stuff_environment_check.py의 대조 쌍을 그대로 쓴다).
  A 불일치: A1 라벨은 outdoor인데 stuff는 실내 쪽(점수 ≤ -0.2) 또는 그 반대(점수 ≥ +0.2)
            A2 라벨에 키워드(snow/water/grass/sky/tree)가 있는데 해당 stuff 면적이 1% 미만
  B 누락:  해당 stuff 면적이 20% 이상인데 라벨에 키워드가 없음
무작위 80장(v10 64장, v9 16장)만 통과 기준 계산에 쓴다(표적 20장은 표본이 편향되어 있다).
출력: review_sample_100.md (표시 없음, 판정이 영향받지 않게), review_sample_100_key.md (층과 이유)
사용: python3 src/labeling/make_review_sample_targeted.py --seed 3
"""
import argparse
import glob
import json
import random
import re
from pathlib import Path

import numpy as np

from stuff_environment_check import D, INDOOR, OUTDOOR, PAIRS, load_all

DIMS = ("place_type", "environment", "activity", "implied_event", "scale", "notes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    out_md, key_md = D / "review_sample_100.md", D / "review_sample_100_key.md"
    wave, env, area = load_all()
    ids = sorted(i for i in env if i in area)

    labels, caps = {}, {}
    import csv
    for r in csv.DictReader(open(D / "queue_order.csv", encoding="utf-8")):
        caps[r["cocoId"]] = json.loads(r["captions"])
    for f in sorted(glob.glob(str(D / "accepted" / "u*_p0_*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        for it in d["items"]:
            labels[str(it["image_id"])] = (it["labels"], d["meta"]["prompt_version"])

    seen = set()
    for f in D.glob("review_sample*.md"):  # 이미 본 표본은 제외(이 스크립트가 덮어쓸 100장 파일과 키 파일은 제외 대상 아님)
        if f not in (out_md, key_md):
            seen |= set(re.findall(r"^## (\d+)", open(f, encoding="utf-8").read(), re.M))

    frac = lambda i, cats: sum(area[i].get(c, 0.0) for c in cats)
    tags = {}
    for i in ids:
        t = []
        only_in = bool(re.search(r"\bindoor|\binterior", env[i])) and not re.search(r"\boutdoor|\bexterior", env[i])
        only_out = bool(re.search(r"\boutdoor|\bexterior", env[i])) and not re.search(r"\bindoor|\binterior", env[i])
        sc = frac(i, OUTDOOR) - frac(i, INDOOR)
        if (only_out and sc <= -0.2) or (only_in and sc >= 0.2):
            t.append("A1 실내외 불일치")
        for nm, pat, cats in PAIRS:
            has, fr = bool(re.search(pat, env[i])), frac(i, cats)
            if has and fr < 0.01:
                t.append(f"A2 {nm}: 라벨에 있으나 stuff 면적 {fr:.1%}")
            if not has and fr >= 0.2:
                t.append(f"B {nm}: stuff 면적 {fr:.0%}인데 라벨에 없음")
        tags[i] = t

    rng = random.Random(a.seed)
    avail = [i for i in ids if i not in seen]
    pickA, pickB = [], []
    candA1 = [i for i in avail if any(x.startswith("A1") for x in tags[i])]
    candA2 = [i for i in avail if any(x.startswith("A2") for x in tags[i]) and i not in candA1]
    pickA = rng.sample(candA1, min(5, len(candA1)))
    pickA += rng.sample([i for i in candA2 if i not in pickA], 10 - len(pickA))
    used = set(pickA)
    for nm, _, _ in PAIRS:  # 키워드별 2장
        cand = [i for i in avail if i not in used and any(x.startswith(f"B {nm}:") for x in tags[i])]
        s = rng.sample(cand, 2)
        pickB += s
        used |= set(s)
    rest = [i for i in avail if i not in used]
    core = [i for i in rest if wave.get(i) == "0"]
    other = [i for i in rest if wave.get(i) != "0"]
    rand = rng.sample(other, 64) + rng.sample(core, 16)
    order = pickA + pickB + rand
    rng.shuffle(order)

    out = [f"# 라벨 직접 확인용 표본 100장 (시드 {a.seed})", "", "이미지는 data/raw/images/{id}.jpg. 이전에 본 표본은 제외했다. 층과 이유는 review_sample_100_key.md에 따로 있다(판정이 영향받지 않게 여기에는 표시하지 않음).", ""]
    for i in order:
        lab, ver = labels[i]
        out += [f"## {i}  (data/raw/images/{i}.jpg, {ver})", "- 캡션: " + " / ".join(caps.get(i, []))]
        for dim in DIMS:
            vals = lab[dim]
            out.append(f"- **{dim}**: " + (" | ".join(f"{v['value']} ({v['confidence']})" for v in vals) if vals else "(없음)"))
        out.append("")
    out_md.write_text("\n".join(out), encoding="utf-8")

    kind = {**{i: "표적 A(불일치)" for i in pickA}, **{i: "표적 B(누락)" for i in pickB}}
    key = ["# review_sample_100 층과 이유", "",
           "무작위 80장(v10 64, v9 16)만 통과 기준 계산에 쓴다. 표적 20장은 어긋난 곳에서 뽑아 틀림이 더 많이 나오므로 전체 비율에 섞지 않고 오류 유형 확인에만 쓴다.", "",
           "| 이미지 | 층 | 버전 | 이유(COCO-Stuff 대조) |", "|---|---|---|---|"]
    for i in sorted(order, key=lambda x: (kind.get(x, "z"), x)):
        key.append(f"| {i} | {kind.get(i, '무작위')} | {labels[i][1]} | {'; '.join(tags[i]) if tags[i] else '-'} |")
    key_md.write_text("\n".join(key), encoding="utf-8")
    print(f"저장 완료. 후보: A1 {len(candA1)}, A2 {len(candA2)}, 무작위 후보 {len(rest)}; 표적 {len(pickA) + len(pickB)}장, 무작위 {len(rand)}장")
    print("무작위 80장 중 우연히 불일치/누락 조건에 해당:", sum(1 for i in rand if tags[i]))


if __name__ == "__main__":
    main()
