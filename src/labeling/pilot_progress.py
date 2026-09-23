"""파일럿 300장 중 각 이미지가 어느 단계까지 끝났는지 로그에서 읽어 보고한다.

서브에이전트 호출을 자동화하지 않는다(CLAUDE.md: 파일럿 단계는 서브에이전트 호출을 직접 쓴다).
이 스크립트는 다음에 무엇을 해야 하는지 알려주는 읽기 전용 도우미다. 여러 세션에 걸쳐
파일럿을 이어갈 때, 로그(data/labels/agent_logs/*.jsonl)를 보고 남은 이미지를 계산한다.

사용:
  python3 src/labeling/pilot_progress.py            # 요약만
  python3 src/labeling/pilot_progress.py --next 5    # 다음에 돌릴 이미지 id 5개 출력
"""
import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PILOT = REPO / "data" / "processed" / "pilot_300.csv"
LOG_DIR = REPO / "data" / "labels" / "agent_logs"
STAGES = ("extract", "classify", "critique_a", "critique_b", "adjudicate")
N_CLASSIFY_SAMPLES = 3  # CLAUDE.md 파일럿 기본값


def image_ids():
    return [r["cocoId"] for r in csv.DictReader(open(PILOT, encoding="utf-8"))]


def stage_counts(stage):
    """image_id -> 그 이미지에 대해 이 단계 로그가 몇 줄 있는지."""
    path = LOG_DIR / f"{stage}.jsonl"
    counts = {}
    if not path.exists():
        return counts
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        iid = str(rec.get("image_id"))
        counts[iid] = counts.get(iid, 0) + 1
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--next", type=int, default=0, help="다음에 돌릴 이미지 id를 N개 출력")
    args = ap.parse_args()

    ids = image_ids()
    counts = {s: stage_counts(s) for s in STAGES}

    done = {s: 0 for s in STAGES}
    remaining = []
    for iid in ids:
        needed = {
            "extract": 1,
            "classify": N_CLASSIFY_SAMPLES,
            "critique_a": 1,
            "critique_b": 1,
            "adjudicate": 1,
        }
        finished_all = True
        for s in STAGES:
            if counts[s].get(iid, 0) >= needed[s]:
                done[s] += 1
            else:
                finished_all = False
        if not finished_all:
            remaining.append(iid)

    print(f"파일럿 전체: {len(ids)}장")
    for s in STAGES:
        print(f"  {s}: {done[s]}/{len(ids)} 완료")
    print(f"adjudicate까지 완전히 끝난 이미지: {len(ids) - len(remaining)}/{len(ids)}")
    print(f"남은 이미지: {len(remaining)}")

    if args.next > 0:
        print(f"\n다음 {min(args.next, len(remaining))}개:")
        for iid in remaining[: args.next]:
            per_stage = {s: counts[s].get(iid, 0) for s in STAGES}
            print(f"  {iid}: {per_stage}")


if __name__ == "__main__":
    main()
