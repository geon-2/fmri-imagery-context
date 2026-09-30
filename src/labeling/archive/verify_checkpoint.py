"""classify_v8 체크포인트 산출물을 검증한다.

체크포인트 상태 파일(checkpoints_{sample_idx}.json)에서 output_written=true,
verified=null인 체크포인트를 찾아 아래를 확인하고 verified 필드를 true/false로 채운다:

1. 파일 존재: 체크포인트에 속한 모든 group의 g{NN}_classify_out_{s}.json이 있는가.
2. 타임스탬프 간격: 그룹 파일들의 mtime 범위가 이미지 수 대비 너무 짧지 않은가
   (실제로 이미지를 하나씩 봤다면 최소 몇 분은 걸려야 정상 — 100장 기준 60초 미만이면
   기계적으로 찍어낸 것으로 의심).
3. 캡션 재조합(템플릿) 의심: implied_event/notes 값이 그 이미지의 캡션 단어를 그대로
   가져다 쓴 비율이 너무 높으면(캡션에 없는 내용이 거의 없으면) 이미지를 직접 보지 않고
   캡션만으로 찍어낸 것으로 의심.
4. confidence 획일성: 같은 체크포인트 안에서 너무 많은 값이 정확히 같은 confidence를
   공유하면(예: 전부 0.55) 의심 신호로 취급.

통과하지 못하면 verified=false로 표시하고, 문제가 확인된 그룹(20장 단위)만 redo_groups에
담는다 — 체크포인트 전체(100장)를 다시 하지 않고 그 그룹들만 재작업하면 된다(자동 삭제는
하지 않음 — 사람이 최종 판단).

사용:
  python3 src/labeling/verify_checkpoint.py            # 0,1,2 전부 검사
  python3 src/labeling/verify_checkpoint.py --sample 1 # 특정 sample_idx만
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CDIR = REPO / "data" / "labels" / "external_labeling" / "classify_v8"

MIN_SECONDS_PER_100_IMAGES = 90  # 이보다 빠르면 의심
CAPTION_OVERLAP_THRESHOLD = 0.85  # 캡션 단어와 겹치는 비율이 이 이상이면 의심
STOPWORDS = {"a", "an", "the", "is", "are", "of", "in", "on", "at", "to", "with", "and",
             "next", "before", "intent", "the", "that", "this", "it", "its", "his", "her"}


def words(s):
    return set(re.findall(r"[a-z]+", s.lower())) - STOPWORDS


def caption_overlap(value, captions):
    vw = words(value)
    if not vw:
        return 0.0
    cw = set()
    for c in captions:
        cw |= words(c)
    if not cw:
        return 0.0
    return len(vw & cw) / len(vw)


MIN_SECONDS_PER_GROUP = MIN_SECONDS_PER_100_IMAGES * 20 / 100  # 그룹당(20장) 최소 기대 시간


def verify(sample_idx, cp):
    """그룹(20장) 단위로 진단해서, 문제 있는 그룹만 redo_groups에 담는다.
    반환: (ok: bool, notes: str, redo_groups: list[str])
    """
    groups = cp["groups"]
    missing = [gg for gg in groups if not (CDIR / f"{gg}_classify_out_{sample_idx}.json").exists()]
    if missing:
        return False, f"출력 파일 없음: {missing}", missing

    files = {gg: CDIR / f"{gg}_classify_out_{sample_idx}.json" for gg in groups}
    mtimes = {gg: f.stat().st_mtime for gg, f in files.items()}
    in_files = {gg: json.load(open(CDIR / f"{gg}_classify_in.json", encoding="utf-8")) for gg in groups}

    group_notes = {}
    bad_groups = set()

    # 그룹 사이 처리 간격(순서대로 처리했다고 가정, 앞 그룹 mtime 대비) — 너무 빠르면 의심.
    ordered = list(groups)
    for i, gg in enumerate(ordered):
        if i == 0:
            continue  # 첫 그룹은 체크포인트 시작 시각을 모르니 타이밍으로 판단 불가(내용 검사로만 판단)
        prev_gg = ordered[i - 1]
        duration = mtimes[gg] - mtimes[prev_gg]
        if duration < MIN_SECONDS_PER_GROUP:
            group_notes.setdefault(gg, []).append(f"처리 간격 {duration:.0f}초(그룹당 최소 {MIN_SECONDS_PER_GROUP:.0f}초 기대) — 너무 빠름")
            bad_groups.add(gg)

    for gg, f in files.items():
        d = json.load(open(f, encoding="utf-8"))
        caption_by_image = {it["image_id"]: it.get("captions", []) for it in in_files[gg]["items"]}
        overlap_hits = overlap_total = 0
        confidences = Counter()
        for it in d["items"]:
            caps = caption_by_image.get(it["image_id"], [])
            for dim in ("implied_event", "notes"):
                for v in it["labels"].get(dim, []):
                    overlap_total += 1
                    if caption_overlap(v["value"], caps) >= CAPTION_OVERLAP_THRESHOLD:
                        overlap_hits += 1
                    confidences[round(v.get("confidence", -1), 2)] += 1
        if overlap_total > 0:
            rate = overlap_hits / overlap_total
            if rate > 0.3:
                group_notes.setdefault(gg, []).append(f"캡션 재조합 의심 {rate:.0%}({overlap_hits}/{overlap_total})")
                bad_groups.add(gg)
            top_n = confidences.most_common(1)[0][1] if confidences else 0
            if top_n / overlap_total > 0.5:
                group_notes.setdefault(gg, []).append(f"confidence 편중 {top_n}/{overlap_total}")
                bad_groups.add(gg)

    if bad_groups:
        parts = [f"{gg}({'; '.join(group_notes[gg])})" for gg in sorted(bad_groups)]
        return False, "문제 그룹: " + ", ".join(parts), sorted(bad_groups)

    n_images = sum(len(json.load(open(f, encoding="utf-8"))["items"]) for f in files.values())
    return True, f"통과 ({n_images}장, 그룹 {len(groups)}개 전부 이상 없음)", []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, choices=[0, 1, 2])
    args = ap.parse_args()
    samples = [args.sample] if args.sample is not None else [0, 1, 2]

    any_change = False
    for s in samples:
        path = CDIR / f"checkpoints_{s}.json"
        doc = json.load(open(path, encoding="utf-8"))
        for cp in doc["checkpoints"]:
            # done_unverified만 검증 대상. Codex는 최초 제출이든 반려 후 재작업 제출이든
            # 항상 status를 done_unverified로 맞춰서 저장해야 한다(그래야 재검증 대상이 됨).
            if cp["status"] != "done_unverified":
                continue
            if not cp["output_written"]:
                groups = cp["groups"]
                if all((CDIR / f"{gg}_classify_out_{s}.json").exists() for gg in groups):
                    cp["output_written"] = True
                else:
                    continue
            ok, msg, redo_groups = verify(s, cp)
            cp["verified"] = ok
            cp["status"] = "verified" if ok else "rejected"
            cp["notes"] = msg
            cp["redo_groups"] = redo_groups
            if not ok:
                cp["redo_count"] = cp.get("redo_count", 0) + 1
            any_change = True
            label = "통과" if ok else f"반려(재작업: {redo_groups})"
            print(f"sample={s} checkpoint={cp['checkpoint']} ({','.join(cp['groups'])}): {label} — {msg}")
        json.dump(doc, open(path, "w"), ensure_ascii=False, indent=1)

    if not any_change:
        print("검증할 새 체크포인트 없음")


if __name__ == "__main__":
    main()
