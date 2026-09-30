"""pilot CSV(nsdId,cocoId,cocoSplit,url,captions)를 읽어서 data/raw/images/{cocoId}.jpg가
없는 것만 새로 내려받는다. 이미 있는 파일은 건너뛴다(재다운로드 안 함).

사용:
  python3 src/labeling/download_images.py data/processed/pilot_1000.csv
"""
import csv
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IMG_DIR = REPO / "data" / "raw" / "images"


def main():
    if len(sys.argv) != 2:
        print("사용: python3 download_images.py <pilot_csv>")
        sys.exit(1)
    csv_path = Path(sys.argv[1])
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows if not (IMG_DIR / f"{r['cocoId']}.jpg").exists()]
    print(f"전체 {len(rows)}장 중 이미 있는 것 {len(rows) - len(todo)}장, 새로 받을 것 {len(todo)}장")

    ok, fail = 0, []
    for i, r in enumerate(todo, 1):
        dest = IMG_DIR / f"{r['cocoId']}.jpg"
        tmp = dest.with_suffix(".jpg.part")
        try:
            req = urllib.request.Request(r["url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp, open(tmp, "wb") as f:
                f.write(resp.read())
            tmp.rename(dest)
            ok += 1
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            fail.append((r["cocoId"], str(e)))
            if tmp.exists():
                tmp.unlink()
        if i % 50 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} 처리 (성공 {ok}, 실패 {len(fail)})")

    print(f"완료: 성공 {ok}, 실패 {len(fail)}")
    if fail:
        print("실패 목록 (cocoId, 이유) 최대 20개:")
        for cid, err in fail[:20]:
            print(f"  {cid}: {err}")


if __name__ == "__main__":
    main()
