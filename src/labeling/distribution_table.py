"""공식 라벨(accepted, pass 0)의 6항목 값 분포와 이미지당 라벨 수 분포를 표(md)로 만든다. 외부 패키지 없이 동작.

코어 1,000장(프롬프트 v9)과 그 밖(v10)을 나눠서 보여 준다. 판정 기준은 두지 않는다(진단용).
사용: python3 src/labeling/distribution_table.py
"""
import csv
import glob
import json
from collections import Counter
from pathlib import Path

D = Path("data/labels/classify_scale")
DIMS = ("place_type", "environment", "activity", "implied_event", "scale", "notes")
MINS = {"place_type": 2, "environment": 3, "activity": 2}


def load():
    wave = {r["cocoId"]: r["wave"] for r in csv.DictReader(open(D / "queue_order.csv", encoding="utf-8"))}
    items = {}
    for f in sorted(glob.glob(str(D / "accepted" / "u*_p0_*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        for it in d["items"]:
            items[str(it["image_id"])] = (it["labels"], d["meta"]["prompt_version"])
    return wave, items


def value_table(group, title):
    n_img = len(group)
    rows = []
    for dim in DIMS:
        per_img = [{v["value"].casefold().strip() for v in lab[dim]} for lab, _ in group]
        c = Counter(x for s in per_img for x in s)
        total = sum(len(s) for s in per_img)
        top = c.most_common(1)[0] if c else ("-", 0)
        rows.append(f"| {dim} | {total} | {len(c)} | {top[1] / n_img:.1%} ({top[0]}) |")
    return [f"### {title} (이미지 {n_img:,}장)", "", "| 항목 | 라벨 수 | 고유 값 | 가장 흔한 값의 이미지 비율 |", "|---|---|---|---|", *rows, ""]


def count_table(group, title):
    n_img = len(group)
    out = [f"### {title}: 이미지당 라벨 수 분포 (이미지 {n_img:,}장)", "",
           "| 항목 | 0개 | 1개 | 2개 | 3개 | 4개 | 5개 이상 | 평균 | 최소 개수 충족 |", "|---|---|---|---|---|---|---|---|---|"]
    for dim in DIMS:
        cnt = Counter(min(len(lab[dim]), 5) for lab, _ in group)
        mean = sum(len(lab[dim]) for lab, _ in group) / n_img
        ok = f"{sum(1 for lab, _ in group if len(lab[dim]) >= MINS[dim]) / n_img:.1%}" if dim in MINS else "-"
        out.append(f"| {dim} | " + " | ".join(f"{cnt.get(k, 0) / n_img:.1%}" for k in range(6)) + f" | {mean:.2f} | {ok} |")
    out.append("")
    return out


def main():
    wave, items = load()
    core = [v for i, v in items.items() if wave.get(i) == "0"]
    rest = [v for i, v in items.items() if wave.get(i) != "0"]
    out = [f"# 6항목 값 분포와 이미지당 라벨 수 (pass 0, 이미지 {len(items):,}장, casefold 기준)", "",
           "값 하나가 얼마나 많은 이미지에 나오는지(이미지 비율)와, 이미지당 라벨 수가 어떻게 분포하는지를 본다. 진단용이며 판정 기준은 두지 않았다.",
           "코어 1,000장은 프롬프트 v9, 그 밖은 v10이다.", ""]
    out += value_table(core, "코어 (v9)") + value_table(rest, "그 밖 (v10)")
    out += count_table(core, "코어 (v9)") + count_table(rest, "그 밖 (v10)")
    Path("results/tables/six_dim_distribution.md").write_text("\n".join(out), encoding="utf-8")
    print(f"저장: results/tables/six_dim_distribution.md (코어 {len(core)}, 그 밖 {len(rest)})")


if __name__ == "__main__":
    main()
