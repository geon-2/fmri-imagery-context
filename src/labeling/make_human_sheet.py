"""파일럿 300장 중 사람 라벨링용 200장을 뽑아 빈 시트(CSV)를 만든다.

이미지 파일만 보고 판단하도록 캡션은 넣지 않는다(LLM이 캡션에서 읽은 정보와 독립적으로 비교하기 위해).
파이프라인 출력을 보기 전에 작성해야 한다.

출력: data/labels/human_labels_200.csv (git 추적 안 함)
"""
import argparse
import csv
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PILOT = REPO / "data" / "processed" / "pilot_300.csv"
OUT = REPO / "data" / "labels" / "human_labels_200.csv"
COLUMNS = ["image_id", "image_path", "place_type", "environment", "activity", "scale", "notes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(PILOT, encoding="utf-8")))
    picked = random.Random(args.seed).sample(rows, args.n)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise SystemExit(f"{OUT} 이미 있다. 작성 중인 라벨을 덮어쓰지 않도록 중단한다.")
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: 엑셀 한글 호환
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for r in picked:
            w.writerow([r["cocoId"], f"data/raw/images/{r['cocoId']}.jpg", "", "", "", "", ""])
    print(f"wrote {OUT} ({args.n} images, seed={args.seed})")


if __name__ == "__main__":
    main()
