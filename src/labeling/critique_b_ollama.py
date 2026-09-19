"""critique-b: Ollama 로컬 모델로 누수 판정을 수행하는 래퍼.

프롬프트 본문은 .claude/agents/context-critique-a.md 에서 읽어온다(두 벌로 관리하지 않음).
입력 JSON은 표준입력(또는 --input 파일), 출력 JSON은 표준출력.
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


def call_ollama(model: str, system: str, user: str, timeout: int = 300) -> str:
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
    args = ap.parse_args()

    raw_in = Path(args.input).read_text(encoding="utf-8") if args.input else sys.stdin.read()
    try:
        item = json.loads(raw_in)
    except json.JSONDecodeError as e:
        print(f"입력이 JSON이 아님: {e}", file=sys.stderr)
        return 2

    system, version = load_prompt()
    try:
        raw_out = call_ollama(args.model, system, json.dumps(item, ensure_ascii=False))
    except Exception as e:  # 연결 실패 등은 호출자가 error로 처리하도록 비정상 종료
        print(f"Ollama 호출 실패: {e}", file=sys.stderr)
        return 1

    try:
        parsed = json.loads(raw_out)
    except json.JSONDecodeError:
        parsed = None

    result = {
        "image_id": item.get("image_id"),
        "stage": "critique_b",
        "agent_name": "context-critique-b",
        "agent_model": args.model,
        "prompt_version": version,
        "raw_output": raw_out,
        "parsed_label": parsed,
        "flags": [] if parsed else ["parse_failed"],
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
