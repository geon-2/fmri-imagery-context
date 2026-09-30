"""단위(unit)마다 새 Codex 세션(`codex exec`)을 띄워 classify를 돌리고, 검증하고, 실패한 이미지만 재시도한다.

- 컨텍스트 격리: 단위마다 독립 프로세스라 앞 단위의 출력이 다음 단위에 새어 들어갈 수 없다.
- 검증: validate_unit.py (워커 세션 로그의 view_image 호출 확인 포함). LLM 추가 호출 없음.
- 살리기: 통과한 이미지는 accepted/에 즉시 저장, 실패한 이미지만 다음 시도 대상. 반려 원본은 삭제하지 않고
  quarantine/에 보관. ledger.jsonl에 시도별 모델·effort·프롬프트 해시·토큰·소요시간을 남긴다.
- 재개 가능: 실행할 때마다 accepted/를 읽어 남은 이미지만 처리한다.

사용 예:
  python3 src/labeling/run_scale.py --pass 0 --units 0:3 --dry-run
  python3 src/labeling/run_scale.py --pass 0 --units 0:3 --workers 2 --model gpt-5.5 --effort medium
"""
import argparse
import csv
import datetime
import glob
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_unit import norm, validate  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "data" / "labels" / "classify_scale"
PROMPT_FILE = REPO / ".claude" / "agents" / "context-classify.md"
CODEX_SESSIONS = Path(os.path.expanduser("~/.codex/sessions"))
MAX_ATTEMPTS = 3
UNIT_SIZE = json.load(open(BASE / "config.json"))["unit_size"]
PROMPT_VERSION = __import__("re").search(r"prompt_version: (v\d+)", PROMPT_FILE.read_text(encoding="utf-8")).group(1)
DIMS = ("implied_event", "notes", "activity")

lock = threading.Lock()


QUOTA_RE = __import__("re").compile(r"usage limit|rate.?limit|quota|\b429\b|too many requests|try again (in|at|later)", __import__("re").I)


def load_queue():
    return list(csv.DictReader(open(BASE / "queue_order.csv", encoding="utf-8")))


def accepted_ids(unit, pas):
    """통과한 이미지는 단위 번호가 아니라 pass 전체에서 찾는다(단위 크기를 바꿔도 이미 통과한 이미지를 잃지 않음)."""
    ids = set()
    for f in glob.glob(str(BASE / "accepted" / f"u*_p{pas}_*.json")):
        ids |= {str(it["image_id"]) for it in json.load(open(f, encoding="utf-8"))["items"]}
    return ids


def attempts(unit, pas):
    n = 0
    p = BASE / "ledger.jsonl"
    if p.exists():
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if "note" in r or r.get("counts_attempt") is False:
                continue
            if r["unit"] == unit and r["pass"] == pas and r.get("unit_size") == UNIT_SIZE:
                n += 1
    return n


def download(row):
    path = REPO / "data" / "raw" / "images" / f"{row['cocoId']}.jpg"
    if path.exists():
        return
    url = f"http://images.cocodataset.org/{row['cocoSplit']}/{int(row['cocoId']):012d}.jpg"
    tmp = path.with_name(f"{path.name}.dl{os.getpid()}_{threading.get_ident()}")
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read()
    tmp.write_bytes(data)
    os.replace(tmp, path)


def find_session_log(marker, since):
    for f in sorted(CODEX_SESSIONS.rglob("*.jsonl"), key=os.path.getmtime, reverse=True):
        if os.path.getmtime(f) < since:
            break
        with open(f, encoding="utf-8", errors="ignore") as fh:
            if marker in fh.read():
                return f
    return None


def tokens_of(log):
    last = None
    for line in open(log, encoding="utf-8", errors="ignore"):
        if '"token_count"' in line:
            try:
                p = json.loads(line).get("payload", {})
            except Exception:
                continue
            if p.get("type") == "token_count" and p.get("info"):
                last = p["info"]["total_token_usage"]
    if not last:
        return None
    return {"input": last["input_tokens"], "cached": last["cached_input_tokens"], "output": last["output_tokens"],
            "billable_like": last["input_tokens"] - last["cached_input_tokens"] + 0.1 * last["cached_input_tokens"] + last["output_tokens"]}


def phrase_counts():
    p = BASE / "phrase_counts.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else {}


def add_phrases(items):
    counts = phrase_counts()
    for it in items:
        seen = set()
        for d in DIMS:
            for v in it["labels"].get(d, []):
                k = norm(v["value"])
                if len(k.split()) >= 5 and k not in seen:
                    counts[k] = counts.get(k, 0) + 1
                    seen.add(k)
    json.dump(counts, open(BASE / "phrase_counts.json", "w", encoding="utf-8"), ensure_ascii=False)


def build_prompt(work, rel_in, rel_out, sample_idx, n, retry=False):
    return (
        f"작업 저장소: {REPO}\n작업 ID: {work} (추적용 문자열)\n\n"
        f".claude/agents/context-classify.md 파일을 읽고, 프롬프트 본문(YAML frontmatter 아래 전체)을 그대로 따라라.\n"
        f"입력은 {rel_in} ({n}장)이다. 각 이미지를 이미지 열람 도구(view_image)로 반드시 직접 열어 실제로 보고 판단해라. "
        f"캡션 문장을 재조합해서 채우는 것은 금지다. 이미지에서만 보이는, 캡션에 없는 구체적 정보를 포함시켜라. "
        f"label의 value 문자열은 반드시 영어로 쓴다. 각 이미지의 라벨은 그 이미지만 보고 새로 쓰고, 다른 이미지에 쓴 문장을 그대로 복사하지 마라.\n"
        f"sample_idx는 {sample_idx}이다. 출력은 {rel_out} 로 저장한다(스키마: 프롬프트의 출력 형식 그대로, 입력의 모든 image_id 포함, "
        f"6차원 키 전부). "
        + ("이 이미지들은 이전 시도가 자동 검사를 통과하지 못해 다시 하는 것이다. 특히 항목별 최소 개수(place_type 2, environment 3, activity 2 이상)를 반드시 채워라. " if retry else "")
        + f"이 작업 외에 다른 파일은 만들거나 수정하지 마라. 끝나면 '완료'라고만 답해라.\n"
    )


def run_work(unit, pas, rows_in, model, effort, prompt_sha, dry, timeout):
    att = attempts(unit, pas)
    work = f"u{unit:05d}_p{pas}_a{att}"
    rel_in = f"data/labels/classify_scale/units/{work}_in.json"
    rel_out = f"data/labels/classify_scale/staging/{work}_out.json"
    items = [{"image_id": r["cocoId"], "image_path": f"data/raw/images/{r['cocoId']}.jpg",
              "captions": json.loads(r["captions"])} for r in rows_in]
    unit_in = {"vocab": None, "items": items}
    if dry:
        print(f"[dry] {work}: {len(items)}장 model={model} effort={effort}")
        return None
    for r in rows_in:
        download(r)
    (REPO / rel_in).write_text(json.dumps(unit_in, ensure_ascii=False, indent=1), encoding="utf-8")
    prompt = build_prompt(work, rel_in, rel_out, pas, len(items), retry=att > 0)
    t0 = time.time()
    cmd = ["codex", "exec", "-m", model, "-c", f'model_reasoning_effort="{effort}"', "--ignore-user-config",
           "-s", "workspace-write", "-C", str(REPO), "-"]
    proc = None
    try:
        proc = subprocess.run(cmd, input=prompt, text=True, capture_output=True, timeout=timeout, cwd=REPO)
        status = "ok" if proc.returncode == 0 else f"exit{proc.returncode}"
    except subprocess.TimeoutExpired:
        status = "timeout"
    secs = time.time() - t0
    if status != "ok" and proc is not None and QUOTA_RE.search((proc.stderr or "") + (proc.stdout or "")):
        rec = {"ts": datetime.datetime.now().isoformat(timespec="seconds"), "work": work, "unit": unit, "pass": pas,
               "unit_size": UNIT_SIZE, "verdict": "quota_stop", "counts_attempt": False, "n_in": len(items),
               "n_accepted": 0, "seconds": round(secs), "tokens": None, "exec_status": status,
               "detail": ((proc.stderr or "") + (proc.stdout or ""))[-300:]}
        with lock:
            (BASE / "ledger.jsonl").open("a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec
    log = find_session_log(work, t0 - 5)
    tok = tokens_of(log) if log else None

    rec = {"ts": datetime.datetime.now().isoformat(timespec="seconds"), "work": work, "unit": unit, "pass": pas,
           "attempt": att, "unit_size": UNIT_SIZE, "model": model, "effort": effort, "prompt_sha": prompt_sha, "n_in": len(items),
           "seconds": round(secs), "tokens": tok, "session_log": str(log) if log else None, "exec_status": status}
    out_path = REPO / rel_out
    if not out_path.exists():
        rec.update(verdict="no_output", n_accepted=0)
        with lock:
            (BASE / "ledger.jsonl").open("a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec
    try:
        unit_out = json.loads(out_path.read_text(encoding="utf-8"))
    except Exception:
        rec.update(verdict="bad_json", n_accepted=0)
        (BASE / "quarantine" / f"{work}_raw.txt").write_text(out_path.read_text(encoding="utf-8", errors="ignore"))
        with lock:
            (BASE / "ledger.jsonl").open("a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    with lock:
        counts = phrase_counts()
    try:
        res = validate(unit_in, unit_out, log, counts, secs, enforce_min=(att < MAX_ATTEMPTS - 1))
    except Exception as e:  # 한 단위의 이상한 산출물이 전체 실행을 멈추지 않게
        rec.update(verdict="validator_error", n_accepted=0, error=repr(e)[:200])
        with lock:
            (BASE / "quarantine" / f"{work}_raw.json").write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")
            with (BASE / "ledger.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_path.unlink()
        return rec
    if log is None:
        res.update(verdict="rejected", accepted_ids=[], redo_ids=list(res["redo_ids"]) or [str(i["image_id"]) for i in items])
        res["hard"] = {str(i["image_id"]): ["no_session_log"] for i in items}
    out_items = {str(it["image_id"]): it for it in unit_out.get("items", [])}
    acc = []
    for iid in res["accepted_ids"]:
        it = dict(out_items[iid])
        it["flags"] = res["soft"].get(iid, []) + res["unit_flags"]
        acc.append(it)
    with lock:
        if acc:
            meta = {"work": work, "unit": unit, "pass": pas, "model": model, "effort": effort, "prompt_sha": prompt_sha,
                    "prompt_version": PROMPT_VERSION}
            (BASE / "accepted" / f"{work}.json").write_text(
                json.dumps({"meta": meta, "sample_idx": pas, "items": acc}, ensure_ascii=False, indent=1), encoding="utf-8")
            add_phrases(acc)
        if res["hard"]:
            (BASE / "quarantine" / f"{work}_raw.json").write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")
        rec.update(verdict=res["verdict"], n_accepted=len(acc), hard=res["hard"], soft_n=len(res["soft"]),
                   unit_flags=res["unit_flags"])
        with (BASE / "ledger.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out_path.unlink()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pas", type=int, required=True, help="sample_idx (독립 샘플 번호)")
    ap.add_argument("--units", required=True, help="START:END 단위 인덱스 범위(END 미포함)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--max-billable-tokens", type=float, default=float("inf"))
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg = json.load(open(BASE / "config.json"))
    U = cfg["unit_size"]
    queue = load_queue()
    s, e = map(int, a.units.split(":"))
    prompt_sha = hashlib.sha256(PROMPT_FILE.read_bytes()).hexdigest()[:12]

    todo = []
    for k in range(s, e):
        rows = queue[k * U:(k + 1) * U]
        if not rows:
            break
        done = accepted_ids(k, a.pas)
        pending = [r for r in rows if r["cocoId"] not in done]
        if not pending:
            continue
        if attempts(k, a.pas) >= MAX_ATTEMPTS:
            print(f"unit {k}: 시도 {MAX_ATTEMPTS}회 초과, 건너뜀(수동 확인 필요, 남은 {len(pending)}장)")
            continue
        todo.append((k, pending))
    print(f"실행 대상 단위 {len(todo)}개 (pass={a.pas}, model={a.model}/{a.effort}, prompt_sha={prompt_sha})")

    spent = 0.0
    fails = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i in range(0, len(todo), a.workers):
            if spent > a.max_billable_tokens:
                print(f"토큰 예산 초과({spent:,.0f} > {a.max_billable_tokens:,.0f}) — 남은 {len(todo) - i}개 단위는 실행하지 않음")
                break
            futs = [(k, ex.submit(run_work, k, a.pas, pending, a.model, a.effort, prompt_sha, a.dry_run, a.timeout))
                    for k, pending in todo[i:i + a.workers]]
            stop = False
            for k, f in futs:
                r = f.result()
                if r and r["verdict"] == "quota_stop":
                    print(f"unit {k}: 사용량 한도/속도제한 감지 — 실행 중단. 상세: {r.get('detail')}")
                    stop = True
                    continue
                if r and r["verdict"] in ("no_output", "bad_json"):
                    fails += 1
                elif r:
                    fails = 0
                if r:
                    spent += (r["tokens"] or {}).get("billable_like", 0)
                    print(f"unit {k}: {r['verdict']} 통과 {r['n_accepted']}/{r['n_in']} {r['seconds']}초 "
                          f"tokens={round((r['tokens'] or {}).get('billable_like', 0)):,} 재작업={len(r.get('hard', {}))}장 "
                          f"unit_flags={r.get('unit_flags')}")
            if stop:
                break
            if fails >= 2 * a.workers:
                print(f"연속 {fails}개 단위가 출력 없음/JSON 손상 — 체계적 문제로 보고 중단")
                break

if __name__ == "__main__":
    main()
