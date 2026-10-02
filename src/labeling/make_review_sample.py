"""공식 라벨(accepted, pass 0)에서 직접 확인용 표본을 뽑아 md로 만든다. 형식은 review_sample_v9.md와 같다.

코어 1,000장(프롬프트 v9)과 그 밖(v10)에서 각각 정해진 수를 무작위로 뽑는다. 이전에 확인한 표본(review_sample_v9.md)은 제외한다.
사용: python3 src/labeling/make_review_sample.py --core 10 --rest 20 --seed 1 --out data/labels/classify_scale/review_sample_v10.md
"""
import argparse
import csv
import glob
import json
import random
import re
from pathlib import Path

D = Path("data/labels/classify_scale")
DIMS = ("place_type", "environment", "activity", "implied_event", "scale", "notes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", type=int, default=10)
    ap.add_argument("--rest", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(D / "review_sample_v10.md"))
    a = ap.parse_args()

    wave = {}
    caps = {}
    for r in csv.DictReader(open(D / "queue_order.csv", encoding="utf-8")):
        wave[r["cocoId"]] = r["wave"]
        caps[r["cocoId"]] = json.loads(r["captions"])
    prev = set()
    for f in D.glob("review_sample*.md"):  # 이미 본 표본은 모두 제외(출력 파일 자신은 덮어쓰므로 제외 대상에서 뺀다)
        if f.resolve() != Path(a.out).resolve():
            prev |= set(re.findall(r"^## (\d+)", open(f, encoding="utf-8").read(), re.M))

    items = {}
    for f in sorted(glob.glob(str(D / "accepted" / "u*_p0_*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        for it in d["items"]:
            items[str(it["image_id"])] = (it, d["meta"]["prompt_version"])

    core = sorted(i for i in items if wave.get(i) == "0" and i not in prev)
    rest = sorted(i for i in items if wave.get(i) != "0" and i not in prev)
    rng = random.Random(a.seed)
    pick = rng.sample(core, a.core) + rng.sample(rest, a.rest)

    out = [f"# 라벨 직접 확인용 표본 {len(pick)}장 (코어 v9 {a.core}장, 그 밖 v10 {a.rest}장, 시드 {a.seed})", "",
           "이미지는 data/raw/images/{id}.jpg. 이전에 확인한 표본(review_sample*.md)은 제외했다.", ""]
    for i in pick:
        it, ver = items[i]
        out += [f"## {i}  (data/raw/images/{i}.jpg, {ver})", "- 캡션: " + " / ".join(caps.get(i, []))]
        for dim in DIMS:
            vals = it["labels"][dim]
            out.append(f"- **{dim}**: " + (" | ".join(f"{v['value']} ({v['confidence']})" for v in vals) if vals else "(없음)"))
        out.append("")
    Path(a.out).write_text("\n".join(out), encoding="utf-8")
    print(f"저장: {a.out} ({len(pick)}장, 후보: 코어 {len(core)}, 그 밖 {len(rest)})")


if __name__ == "__main__":
    main()
