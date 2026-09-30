"""기존 pilot_300.csv(시드 0)의 300장을 그대로 유지하면서, 후보 풀에서 나머지를 시드
고정으로 추가로 뽑아 pilot_1000.csv를 만든다. 이미 라벨링해둔 300장을 재추출로 잃지
않기 위해서다(같은 시드라도 n이 다르면 random.sample 결과가 300장을 그대로 포함한다는
보장이 없다).

출력:
  data/processed/pilot_1000.csv   nsdId, cocoId, cocoSplit, url, captions(JSON 리스트)
                                   (git 추적 안 함)
"""
import argparse
import csv
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
POOL = REPO / "data" / "processed" / "pilot_pool.csv"
EXISTING = REPO / "data" / "processed" / "pilot_300.csv"
ANN = REPO / "data" / "raw" / "annotations"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    existing_rows = list(csv.DictReader(open(EXISTING, encoding="utf-8")))
    existing_ids = {r["cocoId"] for r in existing_rows}
    print(f"기존 pilot_300: {len(existing_rows)}장 유지")

    pool = list(csv.DictReader(open(POOL, encoding="utf-8")))
    remaining_pool = [r for r in pool if r["cocoId"] not in existing_ids]

    n_extra = args.total - len(existing_rows)
    rng = random.Random(args.seed)
    picked = rng.sample(remaining_pool, n_extra)
    print(f"새로 뽑은 이미지: {len(picked)}장 (seed={args.seed}, 후보 풀 {len(remaining_pool)}장 중)")

    caps = {}
    for split in ("train", "val"):
        for a in json.load(open(ANN / f"captions_{split}2017.json", encoding="utf-8"))["annotations"]:
            caps.setdefault(a["image_id"], []).append(a["caption"].strip())

    out = REPO / "data" / "processed" / f"pilot_{args.total}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["nsdId", "cocoId", "cocoSplit", "url", "captions"])
        for r in existing_rows:
            w.writerow([r["nsdId"], r["cocoId"], r["cocoSplit"], r["url"], r["captions"]])
        for r in picked:
            cid = int(r["cocoId"])
            url = f"http://images.cocodataset.org/{r['cocoSplit']}/{cid:012d}.jpg"
            w.writerow([r["nsdId"], cid, r["cocoSplit"], url, json.dumps(caps[cid], ensure_ascii=False)])
    print(f"wrote {out} (총 {len(existing_rows) + len(picked)}장)")


if __name__ == "__main__":
    main()
