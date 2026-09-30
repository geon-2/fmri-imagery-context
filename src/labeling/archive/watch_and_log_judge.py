"""judge_v7/ 디렉터리에서 새로 나온 critique-a/adjudicate 출력을 찾아 검증하고
agent_logs에 기록한다. 이미 처리한 파일은 .processed.json에 기록해 중복 로그를 막는다.

사용:
  python3 src/labeling/watch_and_log_judge.py [--version v8]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def get_jdir(version):
    return REPO / "data" / "labels" / "external_labeling" / f"judge_{version}"


def load_state(jdir):
    state = jdir / ".processed.json"
    if state.exists():
        return set(json.load(open(state, encoding="utf-8")))
    return set()


def save_state(jdir, s):
    state = jdir / ".processed.json"
    json.dump(sorted(s), open(state, "w", encoding="utf-8"), ensure_ascii=False)


def log_items(jdir, stage, version, model, items_by_image):
    log = []
    for iid, parsed in items_by_image.items():
        log.append({
            "image_id": iid, "agent_name": f"context-{stage.replace('_', '-')}",
            "agent_model": model, "prompt_version": version, "input": {},
            "raw_output": json.dumps(parsed, ensure_ascii=False),
            "parsed_label": parsed, "flags": [],
        })
    tmp = jdir / f"_tmp_{stage}_log.json"
    json.dump({"items": log}, open(tmp, "w"), ensure_ascii=False)
    r = subprocess.run(
        [sys.executable, str(REPO / "src" / "labeling" / "log_stage.py"), "--stage", stage],
        stdin=open(tmp, encoding="utf-8"), cwd=REPO, capture_output=True, text=True,
    )
    tmp.unlink()
    return r.stdout.strip(), r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v8")
    args = ap.parse_args()
    jdir = get_jdir(args.version)

    processed = load_state(jdir)
    new_processed = set(processed)
    report = []

    for f in sorted(jdir.glob("*_crita_out*.json")):
        if f.name in processed:
            continue
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception as e:
            report.append(f"{f.name}: JSON 파싱 실패 ({e}) - 건너뜀, 재확인 필요")
            continue
        items = {str(it["image_id"]): it for it in d.get("items", [])}
        out, code = log_items(jdir, "critique_a", "v6", "claude-sonnet-5-high", items)
        report.append(f"{f.name}: critique_a 로그 {out} (이미지 {len(items)}개)")
        new_processed.add(f.name)

    for f in sorted(jdir.glob("*_adj_out*.json")):
        if f.name in processed:
            continue
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception as e:
            report.append(f"{f.name}: JSON 파싱 실패 ({e}) - 건너뜀, 재확인 필요")
            continue
        items = {str(it["image_id"]): it for it in d.get("items", [])}
        out, code = log_items(jdir, "adjudicate", "v4", "claude-sonnet-5-high", items)
        report.append(f"{f.name}: adjudicate 로그 {out} (이미지 {len(items)}개)")
        new_processed.add(f.name)

    save_state(jdir, new_processed)

    if report:
        print("\n".join(report))
    else:
        print("새로 처리할 파일 없음")

    crita_done = len(list(jdir.glob("*_crita_out*.json")))
    adj_done = len(list(jdir.glob("*_adj_out*.json")))
    print(f"\n진행 상황({args.version}): critique-a 출력 파일 {crita_done}개, adjudicate 출력 파일 {adj_done}개")


if __name__ == "__main__":
    main()
