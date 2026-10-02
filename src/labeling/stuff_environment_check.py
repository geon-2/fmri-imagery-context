"""environment 라벨과 COCO-Stuff(사람이 만든 픽셀 단위 stuff 주석)를 대조한다. 진단용이며 합격선은 없다.

대조 쌍은 결과를 보기 전에 아래에 고정했다(키워드 → stuff 범주). 일치율이 낮아도 라벨이 틀렸다는 뜻은 아니다
(COCO-Stuff 범주가 우리 개념과 1:1이 아니고, 주석은 NSD가 보여 준 크롭이 아니라 원본 전체 이미지 기준이다).
결과: results/tables/environment_vs_cocostuff.md
사용: python3 src/labeling/stuff_environment_check.py
"""
import csv
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

D = Path("data/labels/classify_scale")
A = Path("data/raw/annotations")

OUTDOOR = ["sky-other", "clouds", "tree", "grass", "mountain", "hill", "sea", "river", "sand", "snow", "road", "pavement",
           "bush", "dirt", "ground-other", "rock", "fog", "playingfield", "water-other", "gravel"]
INDOOR = ["ceiling-other", "ceiling-tile", "wall-panel", "wall-tile", "wall-wood", "floor-other", "floor-wood", "floor-tile",
          "floor-marble", "carpet", "rug", "cabinet", "cupboard", "curtain", "counter", "shelf", "door-stuff", "window-blind",
          "furniture-other", "desk-stuff", "mirror-stuff"]
# (표시 이름, 라벨 키워드 정규식, stuff 범주)
PAIRS = [
    ("snow", r"\bsnow", ["snow"]),
    ("water", r"\b(water|shore|ocean|sea|beach|lake|river)", ["sea", "river", "water-other"]),
    ("grass", r"\b(grass|lawn|pasture|meadow)", ["grass"]),
    ("sky", r"\bsky|cloud", ["sky-other", "clouds"]),
    ("tree", r"\b(tree|forest|wood(?!en)|foliage)", ["tree", "leaves", "branch"]),
]


def auc(pos, neg):
    """Mann-Whitney 기반 AUC. pos가 neg보다 큰 확률."""
    x = np.concatenate([pos, neg])
    r = np.empty(len(x))
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x))
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    r[order] = ranks
    return (r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def load_all():
    """라벨(environment 텍스트)과 이미지별 stuff 면적 비율을 읽는다. 면적은 캐시한다(주석 파일이 1GB를 넘어 느리다)."""
    wave = {}
    for r in csv.DictReader(open(D / "queue_order.csv", encoding="utf-8")):
        wave[r["cocoId"]] = r["wave"]
    env = {}
    for f in sorted(glob.glob(str(D / "accepted" / "u*_p0_*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        for it in d["items"]:
            env[str(it["image_id"])] = " | ".join(v["value"].casefold() for v in it["labels"]["environment"])
    ids = set(env)
    cache = D / "stuff_area_cache.json"
    if cache.exists():
        area = json.load(open(cache, encoding="utf-8"))
        return wave, env, {k: defaultdict(float, v) for k, v in area.items() if k in ids}
    area = {}  # cocoId -> {stuff 이름: 이미지 면적 대비 비율}
    for sp in ("train", "val"):
        d = json.load(open(A / f"stuff_{sp}2017.json", encoding="utf-8"))
        name = {c["id"]: c["name"] for c in d["categories"]}
        size = {str(im["id"]): im["width"] * im["height"] for im in d["images"] if str(im["id"]) in ids}
        for a in d["annotations"]:
            k = str(a["image_id"])
            if k in size:
                area.setdefault(k, defaultdict(float))[name[a["category_id"]]] += a["area"] / size[k]
        del d
    json.dump({k: dict(v) for k, v in area.items()}, open(cache, "w", encoding="utf-8"))
    return wave, env, area


def main():
    wave, env, area = load_all()
    ids = set(env)
    ids = sorted(i for i in ids if i in area)

    def frac(i, cats):
        return sum(area[i].get(c, 0.0) for c in cats)

    out = [f"# environment 라벨 × COCO-Stuff 대조 (이미지 {len(ids):,}장)", "",
           "environment 라벨의 키워드가 COCO-Stuff 면적 비율과 맞는지 본다. 진단용이며 합격선은 없다. 일치율이 낮아도 라벨이 틀렸다는 뜻은 아니다.",
           "COCO-Stuff는 NSD 크롭이 아니라 원본 이미지 기준이고, 범주 체계가 우리 라벨과 1:1이 아니다. 대조 쌍은 결과를 보기 전에 코드에 고정했다.", ""]

    ind = np.array([bool(re.search(r"\bindoor|\binterior", env[i])) for i in ids])
    outd = np.array([bool(re.search(r"\boutdoor|\bexterior", env[i])) for i in ids])
    only_in, only_out = ind & ~outd, outd & ~ind
    score = np.array([frac(i, OUTDOOR) - frac(i, INDOOR) for i in ids])
    out += ["## 실내/실외", "",
            f"- 라벨에 indoor만 있는 이미지 {only_in.sum():,}장, outdoor만 있는 이미지 {only_out.sum():,}장 (둘 다 있거나 둘 다 없는 {len(ids) - only_in.sum() - only_out.sum():,}장 제외)",
            f"- stuff 점수(실외 범주 면적 − 실내 범주 면적)로 outdoor 라벨 이미지를 indoor 라벨 이미지와 가르는 AUC: **{auc(score[only_out], score[only_in]):.3f}** (0.5는 무작위)",
            f"- 평균 점수: outdoor 라벨 {score[only_out].mean():+.3f}, indoor 라벨 {score[only_in].mean():+.3f}", ""]

    out += ["## 키워드별 (라벨에 키워드가 있는 이미지 vs 없는 이미지)", "",
            "| 키워드 | stuff 범주 | 키워드 있음 | stuff 면적 평균(있음) | (없음) | AUC | stuff 면적 20% 이상인 이미지 중 라벨에 키워드 있는 비율 |",
            "|---|---|---|---|---|---|---|"]
    for nm, pat, cats in PAIRS:
        flag = np.array([bool(re.search(pat, env[i])) for i in ids])
        a = np.array([frac(i, cats) for i in ids])
        big = a >= 0.2
        rec = f"{flag[big].mean():.1%} ({big.sum():,}장 중)" if big.sum() else "-"
        out.append(f"| {nm} | {', '.join(cats)} | {flag.sum():,}장 ({flag.mean():.1%}) | {a[flag].mean():.3f} | {a[~flag].mean():.3f} | {auc(a[flag], a[~flag]):.3f} | {rec} |")
    out.append("")
    Path("results/tables/environment_vs_cocostuff.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
