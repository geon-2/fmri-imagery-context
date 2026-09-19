"""후보 풀(pilot_pool.csv)에서 파일럿 이미지를 시드 고정으로 뽑고 캡션과 함께 저장한다.

출력:
  data/processed/pilot_300.csv   nsdId, cocoId, cocoSplit, url, captions(JSON 리스트)  (git 추적 안 함)
이미지 다운로드는 별도 단계(URL은 http:// — images.cocodataset.org 는 https 인증서가 맞지 않는다).
"""
import argparse
import csv
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
POOL = REPO / "data" / "processed" / "pilot_pool.csv"
ANN = REPO / "data" / "raw" / "annotations"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = list(csv.DictReader(open(POOL, encoding="utf-8")))
    rng = random.Random(args.seed)
    picked = rng.sample(pool, args.n)

    caps = {}
    for split in ("train", "val"):
        for a in json.load(open(ANN / f"captions_{split}2017.json", encoding="utf-8"))["annotations"]:
            caps.setdefault(a["image_id"], []).append(a["caption"].strip())

    out = REPO / "data" / "processed" / f"pilot_{args.n}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["nsdId", "cocoId", "cocoSplit", "url", "captions"])
        for r in picked:
            cid = int(r["cocoId"])
            url = f"http://images.cocodataset.org/{r['cocoSplit']}/{cid:012d}.jpg"
            w.writerow([r["nsdId"], cid, r["cocoSplit"], url, json.dumps(caps[cid], ensure_ascii=False)])
    print(f"wrote {out} ({len(picked)} images, seed={args.seed})")


if __name__ == "__main__":
    main()
