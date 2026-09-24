"""critique-b: Ollama 로컬 모델로 누수 판정을 배치로 수행하는 래퍼.

프롬프트 본문은 .claude/agents/context-critique-a.md 에서 읽어온다(두 벌로 관리하지 않음).
입력은 critique-a와 같은 배치 JSON {"items": [{"image_id": ..., "labels": {...}}, ...]}를
표준입력(또는 --input 파일)으로 받아, 한 번의 Ollama 호출로 배치 전체를 처리한다.
출력은 {"items": [<로그 레코드>, ...]} — 각 레코드는 log_stage.py가 그대로 append할 수 있는 형태.

**동시에 여러 프로세스로 이 스크립트를 띄우지 않는다** — Ollama 모델 하나(9GB급)를 여러
요청이 동시에 두고 경쟁하면 16GB RAM에서 대부분 타임아웃한다(실측). 항상 한 번에 하나씩,
배치 안에서 여러 이미지를 한 번의 호출로 묶어 처리한다.

표준 라이브러리만 사용한다.
"""
import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROMPT_FILE = REPO / ".claude" / "agents" / "context-critique-a.md"
DEFAULT_MODEL = "qwen2.5:14b"
OLLAMA_URL = "http://localhost:11434/api/chat"


def load_prompt(path: Path = PROMPT_FILE):
    """frontmatter와 HTML 주석을 제거한 프롬프트 본문과 prompt_version을 반환한다."""
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"\A---\n.*?\n---\n", "", text, count=1, flags=re.S)
    m = re.search(r"<!--\s*prompt_version:\s*(\S+)\s*-->", text)
    version = m.group(1) if m else "unknown"
    body = re.sub(r"<!--.*?-->", "", text, flags=re.S).strip()
    return body, version


def call_ollama(model: str, system: str, user: str, timeout: int = 600) -> str:
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)["message"]["content"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--input", help="입력 JSON 파일 (기본: 표준입력)")
    ap.add_argument("--timeout", type=int, default=600, help="배치 크기에 맞춰 늘려도 된다")
    args = ap.parse_args()

    raw_in = Path(args.input).read_text(encoding="utf-8") if args.input else sys.stdin.read()
    try:
        batch = json.loads(raw_in)
    except json.JSONDecodeError as e:
        print(f"입력이 JSON이 아님: {e}", file=sys.stderr)
        return 2

    items = batch.get("items", [])
    image_ids = [it.get("image_id") for it in items]
    system, version = load_prompt()

    try:
        raw_out = call_ollama(args.model, system, json.dumps(batch, ensure_ascii=False), timeout=args.timeout)
    except Exception as e:  # 연결 실패 등은 호출자가 error로 처리하도록 비정상 종료
        print(f"Ollama 호출 실패: {e}", file=sys.stderr)
        return 1

    try:
        parsed_batch = json.loads(raw_out)
        parsed_by_id = {p.get("image_id"): p for p in parsed_batch.get("items", [])}
    except (json.JSONDecodeError, AttributeError):
        parsed_by_id = {}

    records = []
    for iid in image_ids:
        parsed = parsed_by_id.get(iid)
        records.append({
            "image_id": iid,
            "stage": "critique_b",
            "agent_name": "context-critique-b",
            "agent_model": args.model,
            "prompt_version": version,
            "raw_output": raw_out,
            "parsed_label": parsed,
            "flags": [] if parsed is not None else ["parse_failed"],
        })

    print(json.dumps({"items": records}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
