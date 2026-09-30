"""구버전(4차원, v2) 파이프라인과 신버전(5차원, v3) 파이프라인의 adjudicate 결과를
이미지 단위로 나란히 비교하는 표를 만든다. 재라벨링 없이 이미 있는 로그만 읽는다.

사용:
  python3 src/labeling/compare_pipelines.py > results/tables/pipeline_comparison.csv
"""
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOG_DIR = REPO / "data" / "labels" / "agent_logs"
PILOT = REPO / "data" / "processed" / "pilot_300.csv"

SHARED_DIMS = ("place_type", "environment", "activity", "scale")


def load_adjudicate(version):
    out = {}
    for line in open(LOG_DIR / "adjudicate.jsonl", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("prompt_version") == version:
            out[str(rec["image_id"])] = rec["parsed_label"]
    return out


def top_values(parsed, dim, n=3):
    """support 내림차순으로 상위 n개 값을 'value(support)' 형태로 합친다."""
    rows = parsed.get("labels", {}).get(dim, [])
    rows = sorted(rows, key=lambda r: -r.get("support", 0))[:n]
    return "; ".join(f"{r['value']}({r.get('support', 0):.2f})" for r in rows)


def leak_status(parsed):
    flags = parsed.get("flags", [])
    return "leak" if "leakage_suspected" in flags else "ok"


def order_from_pilot():
    return [r["cocoId"] for r in csv.DictReader(open(PILOT, encoding="utf-8"))]


def main():
    old = load_adjudicate("v2")
    new = load_adjudicate("v3")
    ids = [i for i in order_from_pilot() if i in old and i in new]

    writer = csv.writer(sys.stdout)
    header = ["image_id", "old_leak", "new_leak", "status_change"]
    for dim in SHARED_DIMS:
        header += [f"old_{dim}", f"new_{dim}"]
    header.append("new_implied_event")
    writer.writerow(header)

    for iid in ids:
        o, n = old[iid], new[iid]
        old_leak, new_leak = leak_status(o), leak_status(n)
        if old_leak == new_leak:
            change = "same_" + old_leak
        elif old_leak == "ok" and new_leak == "leak":
            change = "ok_to_leak"
        else:
            change = "leak_to_ok"
        row = [iid, old_leak, new_leak, change]
        for dim in SHARED_DIMS:
            row += [top_values(o, dim), top_values(n, dim)]
        row.append(top_values(n, "implied_event"))
        writer.writerow(row)


if __name__ == "__main__":
    main()
