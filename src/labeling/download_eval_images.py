"""평가용 이미지(shared1000, NSD-Imagery 자연 장면 5장 포함)를 COCO 공개 서버에서 받는다.

download_pool.py의 fetch를 그대로 쓴다(이미 있는 파일은 건너뜀, JPEG 시그니처 확인, 원자적 교체).
imagery 자연 장면 5장은 0부터 센 nsdId 28751, 30856, 53881, 61177, 65872이며 모두 shared1000 안에 있다.
사용: python3 src/labeling/download_eval_images.py --workers 8
"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor

from download_pool import IMG, REPO, fetch

IMAGERY_NSD0 = {"28751", "30856", "53881", "61177", "65872"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(REPO / "data/raw/nsd_stim_info_merged.csv", encoding="utf-8")) if r["shared1000"] == "True"]
    inside = sum(1 for r in rows if r["nsdId"] in IMAGERY_NSD0)
    todo = [r for r in rows if not (IMG / f"{r['cocoId']}.jpg").exists()]
    print(f"shared1000 {len(rows)}장 (imagery 5장 중 {inside}장 포함), 새로 받을 것 {len(todo)}장", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(fetch, todo))
    fails = [(r["cocoId"], x) for r, x in zip(todo, res) if x.startswith("fail")]
    print(f"받음 {sum(1 for x in res if x == 'ok')}, 이미 있음 {sum(1 for x in res if x == 'skip')}, 실패 {len(fails)}")
    for c, e in fails[:10]:
        print("  실패", c, e)
    have = sum(1 for r in rows if (IMG / f"{r['cocoId']}.jpg").exists())
    print(f"shared1000 중 로컬에 있는 이미지: {have}/{len(rows)}")


if __name__ == "__main__":
    main()
