"""단계별 로그를 data/labels/agent_logs/{stage}.jsonl 에 한 줄씩 append 한다 (git 추적 안 함).

사용: 로그 레코드 JSON(단일 객체 또는 {"items": [레코드, ...]} 배치)을 표준입력으로 넘긴다.
  python3 src/labeling/log_stage.py --stage classify < record.json
  python3 src/labeling/log_stage.py --stage classify < batch.json   # {"items":[...]}

레코드 필드는 CLAUDE.md의 로그 스키마를 따른다(image_id, agent_name, agent_model,
prompt_version, input, raw_output, parsed_label, flags). stage와 timestamp는 여기서 채운다.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOG_DIR = REPO / "data" / "labels" / "agent_logs"
STAGES = ("extract", "classify", "critique_a", "critique_b", "adjudicate")
REQUIRED = ("image_id", "agent_name", "agent_model", "prompt_version", "raw_output")


def append_log(stage, record, log_dir=LOG_DIR):
    if stage not in STAGES:
        raise ValueError(f"알 수 없는 stage: {stage}")
    missing = [k for k in REQUIRED if k not in record]
    if missing:
        raise ValueError(f"필수 필드 누락: {missing}")
    rec = {"stage": stage, **record,
           "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    rec.setdefault("flags", [])
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"{stage}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--log-dir", default=str(LOG_DIR))
    args = ap.parse_args()
    data = json.load(sys.stdin)
    records = data["items"] if isinstance(data, dict) and "items" in data else [data]
    for rec in records:
        append_log(args.stage, rec, args.log_dir)
    print(f"{len(records)}건 기록", file=sys.stderr)
