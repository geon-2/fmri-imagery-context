"""queue_order.csv의 전체 이미지를 병렬로 내려받는다(이미 있는 파일은 건너뜀, 큐 순서대로).

run_scale.py가 같은 파일을 동시에 받아도 깨지지 않도록 임시 파일명을 프로세스·스레드별로 분리하고
os.replace로 원자적으로 교체한다. 받은 파일은 JPEG 시그니처와 크기를 확인한다.
실패 목록은 data/processed/download_failures.txt에 남긴다.

사용: python3 src/labeling/download_pool.py --workers 8
"""
import argparse
import csv
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IMG = REPO / "data" / "raw" / "images"
QUEUE = REPO / "data" / "labels" / "classify_scale" / "queue_order.csv"
FAIL = REPO / "data" / "processed" / "download_failures.txt"


def fetch(row, retries=3):
    path = IMG / f"{row['cocoId']}.jpg"
    if path.exists():
        return "skip"
    url = f"http://images.cocodataset.org/{row['cocoSplit']}/{int(row['cocoId']):012d}.jpg"
    tmp = path.with_name(f"{path.name}.dl{os.getpid()}_{threading.get_ident()}")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            if len(data) < 1000 or data[:2] != b"\xff\xd8":
                raise ValueError(f"not a jpeg ({len(data)} bytes)")
            tmp.write_bytes(data)
            os.replace(tmp, path)
            return "ok"
        except Exception as e:
            err = str(e)
            time.sleep(2 * (attempt + 1))
    if tmp.exists():
        tmp.unlink()
    return f"fail:{err}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(QUEUE, encoding="utf-8")))
    todo = [r for r in rows if not (IMG / f"{r['cocoId']}.jpg").exists()]
    print(f"전체 {len(rows)}장 중 이미 있음 {len(rows) - len(todo)}장, 새로 받을 것 {len(todo)}장", flush=True)
    ok = fail = 0
    failures = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(fetch, r): r for r in todo}
        for i, f in enumerate(as_completed(futs), 1):
            res = f.result()
            if res.startswith("fail"):
                fail += 1
                failures.append(f"{futs[f]['cocoId']}\t{res}")
            else:
                ok += 1
            if i % 1000 == 0:
                print(f"  {i}/{len(todo)} 처리 (성공 {ok}, 실패 {fail}, {time.time() - t0:.0f}초)", flush=True)
    FAIL.write_text("\n".join(failures), encoding="utf-8")
    print(f"완료: 성공 {ok}, 실패 {fail} ({time.time() - t0:.0f}초). 실패 목록: {FAIL}", flush=True)


if __name__ == "__main__":
    main()
