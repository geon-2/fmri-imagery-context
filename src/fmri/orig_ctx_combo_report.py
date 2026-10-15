"""원본+맥락 결합 정리(기준 docs/orig_ctx_combo_criteria.md). 사용: python3 src/fmri/orig_ctx_combo_report.py"""
from pathlib import Path

import numpy as np

D = Path("data/results/orig_ctx_combo")
SUBJ = ["01", "02", "05", "07"]
Z = {s: np.load(D / f"orig_ctx_combo_{s}.npz") for s in SUBJ}
rng = np.random.default_rng(3)
g = lambda s, r, k="c1": Z[s][f"{r}__{k}"]


def bs(v):
    m = v[rng.integers(0, len(v), (1000, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def vd(d, lo, hi, t):
    return "높음" if lo > 0 and d >= t else ("낮음" if hi < 0 and d <= -t else "동등")


L = ["# 원본 캡션 + 맥락 라벨 결합 타깃 대 원본 캡션 단독 (피험자 1·2·5·7, CLIP, 전처리 a, 전체 복셀)", "", "기준 `docs/orig_ctx_combo_criteria.md`. 판정: 구간이 0 제외 + top-1 ≥ 0.02(top-5 ≥ 0.03), 「일관」은 4명 중 3명 이상.", "",
     "## 피험자별 top-1 / top-5", "", "| 타깃 | " + " | ".join(SUBJ) + " | 평균 top-1 | 평균 top-5 |", "|---|" + "---|" * (len(SUBJ) + 2)]
RS = ["orig_P1", "ctx_P1", "obj_P1", "orig+ctx_concat", "orig+ctx_mean", "orig+ctx_concat_shuf"]
for r in RS:
    t1 = [g(s, r).mean() for s in SUBJ]
    t5 = [g(s, r, "c5").mean() for s in SUBJ]
    L.append(f"| {r} | " + " | ".join(f"{a:.3f} / {b:.3f}" for a, b in zip(t1, t5)) + f" | {np.mean(t1):.4f} | {np.mean(t5):.4f} |")
L += ["", "## 쌍 비교 (top-1, top-5, 판정 인원 높음-동등-낮음)", "", "| 비교 | 피험자별 top-1 차이 | top-1 판정 | top-5 평균 차이 | top-5 판정 |", "|---|---|---|---|---|"]
for a_, b_ in (("orig+ctx_concat", "orig_P1"), ("orig+ctx_concat", "ctx_P1"), ("orig+ctx_mean", "orig_P1"), ("orig+ctx_mean", "ctx_P1")):
    row = []
    for k, t in (("c1", 0.02), ("c5", 0.03)):
        ds, vs = [], []
        for s in SUBJ:
            d, lo, hi = bs(g(s, a_, k) - g(s, b_, k))
            ds.append(d)
            vs.append(vd(d, lo, hi, t))
        c = {x: vs.count(x) for x in ("높음", "동등", "낮음")}
        rep = [x for x, v in c.items() if v >= 3]
        row.append((ds, f"{c['높음']}-{c['동등']}-{c['낮음']} / {rep[0] if rep else '일관 안 됨'}"))
    L.append(f"| {a_} − {b_} | {' / '.join(f'{x:+.3f}' for x in row[0][0])} | {row[0][1]} | {np.mean(row[1][0]):+.3f} | {row[1][1]} |")
sh = [g(s, "orig+ctx_concat_shuf").mean() for s in SUBJ]
L += ["", f"## 셔플 대조 top-1(결합 타깃, 피험자별): {', '.join(f'{x:.4f}' for x in sh)} (기준 ≤ 0.005 {'충족' if max(sh) <= 0.005 else '미충족'})"]
Path("results/tables/orig_ctx_combo.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
