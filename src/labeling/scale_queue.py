"""전체 풀의 라벨링 순서를 고정한다(재현 가능, 어느 지점에서 멈춰도 무작위 표본).

순서: pilot_1000(이미 고른 1000장, wave 0) → 나머지 풀을 시드 셔플(wave 1~).
어떤 접두 구간도 풀에서의 무작위 표본이라 중간에 멈춰도 분석에 쓸 수 있다.
단위(unit)는 순서대로 UNIT_SIZE장씩 자른 것이고, 파일 저장소 레이아웃은 아래와 같다.

  data/labels/classify_scale/
    queue_order.csv          순서 (order, nsdId, cocoId, cocoSplit, wave, captions)
    config.json              unit_size, seed, wave_size
    units/u{K:05d}_in.json   단위 입력(요청 시 생성)
    staging/                 워커가 쓰는 임시 출력(검증 전)
    accepted/                검증 통과한 이미지만 담은 단위 출력 (u{K}_p{P}[r{n}].json)
    quarantine/              반려된 원본 출력(삭제 안 함, 재검증·분석용)
    ledger.jsonl             시도별 기록(append-only)

사용: python3 src/labeling/scale_queue.py --unit-size 40 --seed 1
"""
import argparse
import csv
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "data" / "labels" / "classify_scale"
WAVE_SIZE = 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit-size", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    pilot = list(csv.DictReader(open(REPO / "data/processed/pilot_1000.csv", encoding="utf-8")))
    pool = list(csv.DictReader(open(REPO / "data/processed/pilot_pool.csv", encoding="utf-8")))
    used = {r["cocoId"] for r in pilot}
    rest = [r for r in pool if r["cocoId"] not in used]
    random.Random(a.seed).shuffle(rest)

    rows = pilot + rest
    caps = {}
    for split in ("train", "val"):
        for an in json.load(open(REPO / "data/raw/annotations" / f"captions_{split}2017.json", encoding="utf-8"))["annotations"]:
            caps.setdefault(an["image_id"], []).append(an["caption"].strip())
    for r in rows:
        if not r.get("captions"):
            r["captions"] = json.dumps(caps.get(int(r["cocoId"]), []), ensure_ascii=False)
    for d in ("units", "staging", "accepted", "quarantine"):
        (BASE / d).mkdir(parents=True, exist_ok=True)
    cols = ["order", "nsdId", "cocoId", "cocoSplit", "wave", "captions"]
    with open(BASE / "queue_order.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"order": i, "nsdId": r["nsdId"], "cocoId": r["cocoId"], "cocoSplit": r["cocoSplit"],
                        "wave": i // WAVE_SIZE, "captions": r.get("captions", "")})
    json.dump({"unit_size": a.unit_size, "seed": a.seed, "wave_size": WAVE_SIZE, "n_images": len(rows)},
              open(BASE / "config.json", "w"), indent=1)
    print(f"큐 {len(rows)}장 (pilot {len(pilot)} + 나머지 {len(rest)}), unit_size={a.unit_size}, "
          f"단위 {-(-len(rows) // a.unit_size)}개")


if __name__ == "__main__":
    main()
