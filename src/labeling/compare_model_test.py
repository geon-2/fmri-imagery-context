"""model_test/test5_out_{모델명}.json 들을 비교해 표로 출력한다.

- 형식·완결성: 5장 전부, 6차원 키 존재
- 이미지 근거: 캡션과 겹치지 않는 값의 비율(높을수록 캡션 재조합이 아니라 이미지에서 뽑은 값)
- 분량: 차원별 평균 라벨 수
- 모델 간 일치도: 이미지·차원별 단어집합 Jaccard의 평균
- 사용량: model_test/usage.json이 있으면 장당 소진 %와 규모별 외삽

usage.json 형식: {"모델명": {"before_pct": 12.0, "after_pct": 13.5, "seconds": 95}}
사용:
  python3 src/labeling/compare_model_test.py
"""
import itertools
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TDIR = REPO / "data" / "labels" / "external_labeling" / "model_test"
DIMS = ("place_type", "environment", "activity", "implied_event", "scale", "notes")
STOP = {"a", "an", "the", "is", "are", "of", "in", "on", "at", "to", "with", "and", "that", "this", "it"}


def words(s):
    return set(re.findall(r"[a-z]+", s.lower())) - STOP


def load_inputs():
    d = json.load(open(TDIR / "test5_in.json", encoding="utf-8"))
    return {it["image_id"]: it for it in d["items"]}


def main():
    inputs = load_inputs()
    outs = {}
    for f in sorted(TDIR.glob("test5_out_*.json")):
        name = f.stem.replace("test5_out_", "")
        try:
            outs[name] = {str(it["image_id"]): it for it in json.load(open(f, encoding="utf-8"))["items"]}
        except Exception as e:
            print(f"{name}: 파싱 실패 ({e})")
    if not outs:
        print("test5_out_*.json 없음")
        return

    print(f"{'모델':<28}{'장수':>4}{'스키마':>7}{'캡션독립%':>10}{'평균라벨/이미지':>15}")
    for name, items in outs.items():
        ok = all(all(d in it.get("labels", {}) for d in DIMS) for it in items.values())
        novel = total = n_labels = 0
        for iid, it in items.items():
            caps = set().union(*[words(c) for c in inputs.get(iid, {}).get("captions", [])] or [set()])
            for d in DIMS:
                for v in it.get("labels", {}).get(d, []):
                    n_labels += 1
                    vw = words(v["value"])
                    if d in ("activity", "implied_event", "notes") and vw:
                        total += 1
                        if len(vw & caps) / len(vw) < 0.5:
                            novel += 1
        pct = f"{100 * novel / total:.0f}" if total else "-"
        print(f"{name:<28}{len(items):>4}{'OK' if ok else 'X':>7}{pct:>10}{n_labels / max(len(items), 1):>15.1f}")

    print("\n모델 간 일치도(단어 Jaccard 평균, 차원별)")
    names = list(outs)
    for a, b in itertools.combinations(names, 2):
        row = []
        for d in DIMS:
            js = []
            for iid in inputs:
                if iid in outs[a] and iid in outs[b]:
                    wa = set().union(*[words(v["value"]) for v in outs[a][iid]["labels"].get(d, [])] or [set()])
                    wb = set().union(*[words(v["value"]) for v in outs[b][iid]["labels"].get(d, [])] or [set()])
                    if wa or wb:
                        js.append(len(wa & wb) / len(wa | wb))
            row.append(f"{d[:5]}={sum(js) / len(js):.2f}" if js else f"{d[:5]}=-")
        print(f"{a} vs {b}: " + " ".join(row))

    up = TDIR / "usage.json"
    if up.exists():
        usage = json.load(open(up, encoding="utf-8"))
        print("\n사용량 (5장 기준 → 외삽, 단위: 해당 구독 풀의 한도 %)")
        print(f"{'모델':<28}{'장당%':>8}{'1000장x1':>10}{'1000장x3':>10}{'10000장x1':>11}{'초/장':>7}")
        for name, u in usage.items():
            per = (u["after_pct"] - u["before_pct"]) / 5
            print(f"{name:<28}{per:>8.3f}{per * 1000:>10.0f}{per * 3000:>10.0f}{per * 10000:>11.0f}{u.get('seconds', 0) / 5:>7.1f}")


if __name__ == "__main__":
    main()
