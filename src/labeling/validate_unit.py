"""classify 산출물 단위(unit) 검증기 — 추가 LLM 호출 없이 산출물과 워커 세션 로그만으로 판정.

이미지 단위로 hard 실패(재작업 대상)와 soft 플래그(보존, 표시만)를 나눈다.

hard 실패(그 이미지만 다시):
  schema      6차원 키/타입/confidence 범위 위반, 이미지 누락, 라벨이 전부 비어 있음
  not_viewed  워커 세션 로그에 그 이미지 경로의 view_image 호출이 없음(세션 로그를 준 경우)
  copy        같은 단위 안에서 여러 이미지가 5단어 이상의 implied_event/notes/activity 문장을
              글자 그대로 공유하는데, 그중 어떤 이미지는 그 문장의 핵심 단어가 자기 캡션·라벨에
              하나도 없고 다른 이미지에는 있음(복사 오염, 예: 화장실 사진의 야구 문구)
  below_min   항목별 최소 개수(place_type 2, environment 3, activity 2) 미달 — 마지막 시도에서는 soft로 내림
  language    label value가 영어가 아님(한글 비율 20% 초과) — 영어 기준의 다른 검사가 무력화되므로
  recombined  implied_event/notes 값이 전부 캡션 단어의 85% 이상과 겹침(이미지 미확인 의심)
soft 플래그(라벨 유지):
  ungrounded_generic  전역 색인에서 같은 5단어 이상 문장이 10개 이상 이미지에 등장하는데 이
                      이미지 자신의 캡션·라벨과 겹치는 단어가 없음(v7 실측: 알려진 오염 49/49를
                      잡지만 정상 샘플도 9~16% 건드려서 hard가 아닌 soft로 둠)
  below_min       (마지막 시도에서만) 최소 개수 미달을 못 채운 이미지 표시
  low_label_rate  단위 이미지의 30% 초과가 below_min
  conf_uniform    단위 안 confidence가 한 값에 50% 초과 편중
  too_fast        단위 처리 시간이 이미지당 3초 미만

단위 판정: hard 실패 이미지가 전체의 25% 이하면 나머지를 accepted, 실패 이미지는 재작업.
25% 초과면 단위 전체 rejected.

사용(단독): python3 src/labeling/validate_unit.py --in IN.json --out OUT.json [--log SESSION.jsonl]
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

DIMS = ("place_type", "environment", "activity", "implied_event", "scale", "notes")
STOP = {"a", "an", "the", "is", "are", "of", "in", "on", "at", "to", "with", "and", "that", "this",
        "it", "its", "his", "her", "next", "before", "intent"}
LONG_PHRASE_WORDS = 5
RECOMBINE_OVERLAP = 0.85
UNIT_FAIL_FRACTION = 0.25
GENERIC_COUNT = 10
CONF_UNIFORM = 0.5
MIN_SEC_PER_IMAGE = 3.0
MIN_COUNTS = {"place_type": 2, "environment": 3, "activity": 2}
LOW_LABEL_UNIT_FRACTION = 0.3


def sval(v):
    """value가 문자열이 아니면(숫자 등) 빈 문자열로 취급한다 — 검증기가 이상한 산출물 때문에 죽지 않게."""
    x = v.get("value") if isinstance(v, dict) else None
    return x if isinstance(x, str) else ""


def norm(s):
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


def words(s):
    return set(re.findall(r"[a-z]+", s.lower())) - STOP


def viewed_paths(session_log):
    paths = set()
    for line in open(session_log, encoding="utf-8", errors="ignore"):
        if "view_image" not in line:
            continue
        try:
            p = json.loads(line).get("payload", {})
        except Exception:
            continue
        if p.get("name") == "view_image":
            try:
                paths.add(json.loads(p["arguments"])["path"])
            except Exception:
                pass
    return paths


def validate(unit_in, unit_out, session_log=None, global_phrase_counts=None, seconds=None, enforce_min=True):
    """enforce_min=True면 항목별 최소 개수 미달을 hard 실패로 본다(그 이미지만 재작업).
    마지막 시도에서는 False로 불러서 못 채운 이미지도 표시(soft)만 붙여 받아들인다 — 채우려고 지어내는 걸 막고,
    이미지가 영영 라벨 없이 남지 않게 하려는 것이다.
    반환: dict(hard={image_id:[reasons]}, soft={image_id:[flags]}, unit_flags=[...], verdict, accepted_ids, redo_ids)"""
    in_items = {str(it["image_id"]): it for it in unit_in["items"]}
    out_items = {str(it["image_id"]): it for it in unit_out.get("items", [])}
    hard, soft, unit_flags = defaultdict(list), defaultdict(list), []

    for iid in in_items:
        if iid not in out_items:
            hard[iid].append("schema:missing")

    for iid, it in out_items.items():
        if iid not in in_items:
            continue
        labels = it.get("labels", {})
        if any(d not in labels for d in DIMS):
            hard[iid].append("schema:dims")
            continue
        n = 0
        for d in DIMS:
            for v in labels[d]:
                n += 1
                c = v.get("confidence")
                if not isinstance(v.get("value"), str) or not isinstance(c, (int, float)) or not 0 <= c <= 1:
                    hard[iid].append("schema:value")
                    break
        if n == 0:
            hard[iid].append("schema:empty")
        hangul = sum(1 for d in DIMS for v in labels[d] if re.search(r"[가-힣]", str(v.get("value", ""))))
        if n and hangul / n > 0.2:
            hard[iid].append("language")

    if not session_log:
        unit_flags.append("no_session_log")
    if session_log:
        seen = {str(Path(p).resolve()) for p in viewed_paths(session_log)}
        for iid, it in in_items.items():
            if str(Path(it["image_path"]).resolve()) not in seen:
                # image_path는 저장소 상대경로일 수 있어 절대경로로 비교
                if not any(s.endswith(it["image_path"]) or s.endswith(f"/{iid}.jpg") for s in seen):
                    hard[iid].append("not_viewed")

    def stems(text):
        return {w[:4] for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 3} - {s[:4] for s in STOP}

    own = {}
    for iid, it in out_items.items():
        txt = " ".join(in_items.get(iid, {}).get("captions", []))
        txt += " " + " ".join(sval(v) for d in ("place_type", "environment", "activity", "scale")
                               for v in it.get("labels", {}).get(d, []))
        own[iid] = stems(txt)

    phrase_imgs = defaultdict(set)
    for iid, it in out_items.items():
        for d in ("implied_event", "notes", "activity"):
            for v in it.get("labels", {}).get(d, []):
                k = norm(sval(v))
                if len(k.split()) >= LONG_PHRASE_WORDS:
                    phrase_imgs[k].add(iid)
    for k, imgs in phrase_imgs.items():
        if len(imgs) < 2:
            continue
        ps = stems(k)
        overlap = {i: len(ps & own.get(i, set())) for i in imgs}
        ref = max(overlap.values())
        for i, o in overlap.items():
            # 여러 이미지가 공유한 문장인데, 이 이미지 자신의 캡션·라벨과 겹치는 단어가 전혀 없고
            # 다른 이미지에는 근거가 있다 → 그 이미지에 복사돼 들어온 것
            if o == 0 and ref > 0:
                hard[i].append("copy")

    for iid, it in out_items.items():
        if iid not in in_items:
            continue
        caps = set().union(*[words(c) for c in in_items[iid].get("captions", [])] or [set()])
        vals = [sval(v) for d in ("implied_event", "notes") for v in it.get("labels", {}).get(d, [])]
        if vals and caps and all(len(words(v) & caps) / max(len(words(v)), 1) >= RECOMBINE_OVERLAP for v in vals):
            hard[iid].append("recombined")

    if global_phrase_counts is not None:
        for iid, it in out_items.items():
            for d in ("implied_event", "notes", "activity"):
                for v in it.get("labels", {}).get(d, []):
                    k = norm(sval(v))
                    if (len(k.split()) >= LONG_PHRASE_WORDS and global_phrase_counts.get(k, 0) >= GENERIC_COUNT
                            and not (stems(k) & own.get(iid, set()))):
                        soft[iid].append("ungrounded_generic")
                        break

    n_low = 0
    for iid, it in out_items.items():
        if iid in in_items:
            low = [d for d, m in MIN_COUNTS.items() if len(it.get("labels", {}).get(d, [])) < m]
            if low:
                (hard if enforce_min else soft)[iid].append("below_min:" + ",".join(low))
                n_low += 1
    if in_items and n_low / len(in_items) > LOW_LABEL_UNIT_FRACTION:
        unit_flags.append("low_label_rate")

    confs = Counter(round(v["confidence"], 2) for it in out_items.values()
                    for d in DIMS for v in it.get("labels", {}).get(d, []) if isinstance(v.get("confidence"), (int, float)))
    total = sum(confs.values())
    if total and confs.most_common(1)[0][1] / total > CONF_UNIFORM:
        unit_flags.append("conf_uniform")
    if seconds is not None and seconds < MIN_SEC_PER_IMAGE * len(in_items):
        unit_flags.append("too_fast")

    hard = {k: sorted(set(v)) for k, v in hard.items()}
    n = len(in_items)
    verdict = "rejected" if len(hard) / n > UNIT_FAIL_FRACTION else "accepted"
    accepted = [i for i in in_items if i not in hard] if verdict == "accepted" else []
    redo = [i for i in in_items if i not in accepted]
    return {"verdict": verdict, "hard": hard, "soft": {k: sorted(set(v)) for k, v in soft.items()},
            "unit_flags": unit_flags, "accepted_ids": accepted, "redo_ids": redo}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log")
    a = ap.parse_args()
    r = validate(json.load(open(a.inp)), json.load(open(a.out)), a.log)
    print(f"verdict={r['verdict']} accepted={len(r['accepted_ids'])} redo={len(r['redo_ids'])} unit_flags={r['unit_flags']}")
    for iid, why in r["hard"].items():
        print(" hard", iid, why)
    for iid, why in r["soft"].items():
        print(" soft", iid, why)


if __name__ == "__main__":
    main()
