"""대규모 classify 실행의 최종 수치를 진행 보고서용 마크다운 표로 출력한다.

사용: python3 src/labeling/report_stats.py
읽는 곳: data/labels/classify_scale/{ledger.jsonl, accepted/}
"""
import collections
import glob
import json
import statistics as st
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / "data" / "labels" / "classify_scale"
DIMS = ("place_type", "environment", "activity", "implied_event", "scale", "notes")
CORE = 1000  # queue_order의 앞 1,000장 = 코어(파일럿 300 + 신규 700)


def main():
    ledger = [json.loads(l) for l in open(BASE / "ledger.jsonl", encoding="utf-8") if l.strip()]
    ledger = [r for r in ledger if "note" not in r and r.get("verdict")]

    items = {}  # (pass, image_id) -> (item, prompt_version)
    for f in sorted(glob.glob(str(BASE / "accepted" / "u*_p*_a*.json"))):
        a = json.load(open(f, encoding="utf-8"))
        for it in a["items"]:
            items[(a["sample_idx"], str(it["image_id"]))] = (it, a["meta"]["prompt_version"])

    print("### 라벨링 완료 현황\n")
    print("| pass | 프롬프트 | 이미지 수 |")
    print("|---|---|---|")
    cnt = collections.Counter((p, v) for (p, _), (_, v) in items.items())
    for (p, v), n in sorted(cnt.items()):
        print(f"| {p} | {v} | {n:,} |")
    uniq = {i for (_, i) in items}
    print(f"\n라벨링된 고유 이미지: **{len(uniq):,}장**\n")

    print("### 실행 통계 (원장 기준)\n")
    scale = [r for r in ledger if r.get("unit_size") == 50]
    n_in = sum(r["n_in"] for r in scale)
    n_acc = sum(r["n_accepted"] for r in scale)
    tok = [(r.get("tokens") or {}).get("billable_like", 0) for r in scale]
    secs = [r["seconds"] for r in scale if r["seconds"] < 3000]
    hard = sum(len(r.get("hard") or {}) for r in scale)
    print("| 항목 | 값 |")
    print("|---|---|")
    print(f"| 실행 단위(50장) | {len(scale)}개 |")
    print(f"| 처리 이미지 / 1차 통과 | {n_in:,} / {n_acc:,} ({100 * n_acc / max(n_in, 1):.1f}%) |")
    print(f"| 재작업 대상 이미지 | {hard}장 |")
    print(f"| 토큰(billable-like) 합계 | {sum(tok) / 1e6:.1f}M (장당 {sum(tok) / max(n_in, 1):,.0f}) |")
    print(f"| 단위당 소요 중앙값 | {st.median(secs):.0f}초 |" if secs else "| 단위당 소요 | - |")
    print(f"| 모델 / effort | {', '.join(sorted({r['model'] + '/' + r['effort'] for r in scale}))} |")

    print("\n### 이미지당 라벨 수 (프롬프트 버전별, 차원별 평균)\n")
    print("| 프롬프트 | 이미지 | 전체 | " + " | ".join(DIMS) + " |")
    print("|---|---|---|" + "---|" * len(DIMS))
    by = collections.defaultdict(list)
    for (_, _), (it, v) in items.items():
        by[v].append(it)
    for v, its in sorted(by.items()):
        n = len(its)
        tot = sum(len(x) for it in its for x in it["labels"].values()) / n
        per = [sum(len(it["labels"][d]) for it in its) / n for d in DIMS]
        print(f"| {v} | {n:,} | {tot:.1f} | " + " | ".join(f"{x:.2f}" for x in per) + " |")

    flags = collections.Counter(f.split(":")[0] for (it, _) in items.values() for f in it.get("flags", []))
    print(f"\n검증 soft 플래그: {dict(flags) if flags else '없음'}")


if __name__ == "__main__":
    main()
