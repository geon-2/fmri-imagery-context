"""Codex/Claude 세션 로그에서 마커 문자열이 등장하는 세션의 토큰 사용량을 뽑는다.
사용량을 사람이 따로 기록하지 않아도 사후에 측정할 수 있다.

사용:
  python3 src/labeling/session_usage.py --marker test5 --since 2026-09-29T13:15
Claude 단가(API 환산, $/MTok): Sonnet 5.5 입력2 출력10, 캐시읽기 0.1x, 캐시쓰기 1.25x.
Codex는 구독제라 토큰만 보고(단가 환산 안 함). billable-like = 비캐시 입력 + 0.1*캐시 입력 + 출력.
"""
import argparse
import datetime
import glob
import json
import os
import re

CLAUDE_DIR = os.path.expanduser("~/.claude/projects/-Users-gunlee-Development-fmri-imagery-context")
CODEX_DIR = os.path.expanduser("~/.codex/sessions")


def codex(marker, since):
    rows = []
    for f in glob.glob(os.path.join(CODEX_DIR, "**", "*.jsonl"), recursive=True):
        if os.path.getmtime(f) < since:
            continue
        model = eff = last = first = end = None
        hit = False
        for line in open(f, encoding="utf-8", errors="ignore"):
            if marker in line:
                hit = True
            try:
                e = json.loads(line)
            except Exception:
                continue
            p = e.get("payload") if isinstance(e.get("payload"), dict) else {}
            if e.get("type") == "turn_context":
                model = p.get("model")
                eff = p.get("effort") or p.get("reasoning_effort")
            if p.get("type") == "token_count" and p.get("info"):
                last = p["info"].get("total_token_usage")
            first = first or e.get("timestamp")
            end = e.get("timestamp") or end
        if hit and last:
            inp, cached, out = last["input_tokens"], last["cached_input_tokens"], last["output_tokens"]
            secs = (datetime.datetime.fromisoformat(end.replace("Z", "+00:00")) - datetime.datetime.fromisoformat(first.replace("Z", "+00:00"))).total_seconds()
            rows.append((f"codex {model}/{eff}", inp - cached, cached, out, inp - cached + 0.1 * cached + out, secs, None))
    return rows


def claude(marker, since):
    rows = []
    for f in glob.glob(os.path.join(CLAUDE_DIR, "*.jsonl")):
        if os.path.getmtime(f) < since:
            continue
        txt = open(f, encoding="utf-8", errors="ignore").read()
        if marker not in txt:
            continue
        i = o = cr = cw = 0
        models, ts = set(), []
        for line in txt.splitlines():
            try:
                e = json.loads(line)
            except Exception:
                continue
            m = e.get("message")
            if e.get("type") == "assistant" and isinstance(m, dict) and m.get("usage"):
                u = m["usage"]
                i += u.get("input_tokens", 0)
                o += u.get("output_tokens", 0)
                cr += u.get("cache_read_input_tokens", 0)
                cw += u.get("cache_creation_input_tokens", 0)
                models.add(m.get("model"))
            if e.get("timestamp"):
                ts.append(e["timestamp"])
        if len(models - {"<synthetic>"}) != 1 or "test5_out" not in txt[:200000] and marker == "test5":
            pass
        secs = (datetime.datetime.fromisoformat(ts[-1].replace("Z", "+00:00")) - datetime.datetime.fromisoformat(ts[0].replace("Z", "+00:00"))).total_seconds()
        usd = i * 2e-6 + o * 10e-6 + cr * 0.2e-6 + cw * 2.5e-6
        label = re.search(r"test5_out_(claude-[a-z0-9-]+)", txt)
        rows.append((f"claude {label.group(1) if label else sorted(models)}", i + cw, cr, o, i + cw + 0.1 * cr + o, secs, usd))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--marker", default="test5")
    ap.add_argument("--since", default="2026-09-29T13:15")
    a = ap.parse_args()
    since = datetime.datetime.fromisoformat(a.since).timestamp()
    print(f"{'세션':<40}{'비캐시입력':>11}{'캐시입력':>11}{'출력':>8}{'billable-like':>14}{'초':>6}{'API$(claude)':>13}")
    for r in codex(a.marker, since) + claude(a.marker, since):
        usd = f"{r[6]:.3f}" if r[6] is not None else "-"
        print(f"{r[0]:<40}{r[1]:>11,}{r[2]:>11,}{r[3]:>8,}{r[4]:>14,.0f}{r[5]:>6.0f}{usd:>13}")


if __name__ == "__main__":
    main()
