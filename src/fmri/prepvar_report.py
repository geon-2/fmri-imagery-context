"""전처리 a 대 d의 1단계 결과 비교 정리(기준 docs/stage1_prepvar_criteria.md). 사용: python3 src/fmri/prepvar_report.py"""
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import kendalltau

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

KFONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
if Path(KFONT).exists():
    font_manager.fontManager.addfont(KFONT)
    plt.rcParams["font.family"] = ["Apple SD Gothic Neo", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
R = Path("data/results")
A = dict(np.load(R / "text_decode_compare/text_decode_compare_scores.npz"))
A.update(dict(np.load(R / "text_decode_compare/text_decode_t10_scores.npz")))
Dd = dict(np.load(R / "text_decode_prepvar_d/text_decode_prepvar_scores.npz"))
REPS = ["orig_P1", "orig_V2", "obj_P1", "obj_V2", "ctx_P1", "ctx_V2", "obj_P1+ctx_P1", "ctx_T10", "ctx_P1_shuf"]
CON = {"K1 맥락P1−사물P1": ("ctx_P1", "obj_P1"), "K2 맥락P1−원본P1": ("ctx_P1", "orig_P1"), "K3 V2−P1": ("ctx_V2", "ctx_P1"),
       "K4 (사물+맥락)−원본P1": ("obj_P1+ctx_P1", "orig_P1"), "K5 T10−맥락P1": ("ctx_T10", "ctx_P1"), "K6 T10−V2": ("ctx_T10", "ctx_V2")}
n = len(A["CLIP__orig_P1__c1"])
rng = np.random.default_rng(11)
idx = np.stack([rng.integers(0, n, n) for _ in range(1000)])
bs = lambda v: (float(v.mean()), float(np.percentile(v[idx].mean(1), 2.5)), float(np.percentile(v[idx].mean(1), 97.5)))
vd = lambda d, lo, hi, t: "변함" if (lo > 0 or hi < 0) and abs(d) >= t else "같음"
g = lambda S, e, r, k: S[f"{e}__{r}__{k}"]
L = ["# 1단계를 전처리 a와 d로 반복한 결과 (피험자 1)", "", "기준 `docs/stage1_prepvar_criteria.md`. a = 6,728 복셀, d = 4,087 복셀. d − a는 같은 이미지 쌍 부트스트랩.", ""]
for e in ("CLIP", "MPNet"):
    L += [f"## {e}: 표현별 점수 (a / d, d−a와 판정)", "", "| 표현 | top-1 a | top-1 d | d−a | 판정 | top-5 a | top-5 d | d−a | 판정 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in REPS:
        if f"{e}__{r}__c1" not in Dd:
            continue
        row = []
        for k, t in (("c1", 0.02), ("c5", 0.03)):
            a, d = g(A, e, r, k), g(Dd, e, r, k)
            df, lo, hi = bs(d - a)
            row += [f"{a.mean():.4f}", f"{d.mean():.4f}", f"{df:+.4f}", vd(df, lo, hi, t)]
        L.append(f"| {r} | " + " | ".join(row) + " |")
    L.append("")
L += ["## 핵심 대비 (top-1 차이, 판정 = 차이 있음/동등)", "", "| 대비 | CLIP a | CLIP d | MPNet a | MPNet d |", "|---|---|---|---|---|"]
same = True
for nm, (x, y) in CON.items():
    row = []
    for e in ("CLIP", "MPNet"):
        if f"{e}__{x}__c1" not in Dd or f"{e}__{y}__c1" not in Dd:
            row += ["-", "-"]
            continue
        vs = []
        for S in (A, Dd):
            d, lo, hi = bs(g(S, e, x, "c1") - g(S, e, y, "c1"))
            vs.append((d, "차이 있음" if (lo > 0 or hi < 0) and abs(d) >= 0.02 else "동등"))
        same &= vs[0][1] == vs[1][1]
        row += [f"{v[0]:+.4f} {v[1]}" for v in vs]
    L.append(f"| {nm} | " + " | ".join(row) + " |")
L += ["", f"핵심 대비 판정이 a와 d에서 모두 같은가: **{'같다' if same else '다르다(위 표 참조)'}**", "", "## 표현 순위 Kendall τ (top-1, a vs d)", ""]
tau = {}
for e in ("CLIP", "MPNet"):
    rs = [r for r in REPS if r != "ctx_P1_shuf" and f"{e}__{r}__c1" in Dd]
    tau[e] = float(kendalltau([g(A, e, r, "c1").mean() for r in rs], [g(Dd, e, r, "c1").mean() for r in rs])[0])
    L.append(f"- {e}: τ = {tau[e]:.2f} ({len(rs)}개 표현) → {'순위 유지' if tau[e] >= 0.9 else '순위 달라짐'}")
er = json.load(open(R / "text_decode_prepvar_d/text_decode_prepvar.json", encoding="utf-8"))["encoder_rule"]
ea = json.load(open(R / "text_decode_compare/text_decode_compare.json", encoding="utf-8"))["encoder_rule"]
L += ["", f"## 인코더 선정 규칙: a = {ea['verdict']}, d = {er['verdict']}", ""]
for r in ("orig_P1", "ctx_P1", "obj_P1+ctx_P1"):
    L.append(f"- {r}: a {ea[r]['mpnet_minus_clip']:+.4f}, d {er[r]['mpnet_minus_clip']:+.4f} (MPNet − CLIP)")
sh = {e: g(Dd, e, "ctx_P1_shuf", "c1").mean() for e in ("CLIP", "MPNet")}
L += ["", f"## 셔플 대조 top-1 (d): CLIP {sh['CLIP']:.4f}, MPNet {sh['MPNet']:.4f} (기준 ≤ 0.005 {'충족' if max(sh.values()) <= 0.005 else '미충족'})"]
Path("results/tables/text_decode_prepvar.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
nm = ["orig_P1", "obj_P1", "ctx_P1", "obj_P1+ctx_P1", "ctx_V2", "ctx_T10"]
for ax, (k, t) in zip(axs, (("c1", "top-1"), ("c5", "top-5"))):
    for i, (S, lab, col) in enumerate(((A, "전처리 a (6,728)", "#2B6CB0"), (Dd, "전처리 d (4,087)", "#D9822B"))):
        ax.bar(np.arange(len(nm)) + (i - 0.5) * 0.38, [g(S, "CLIP", r, k).mean() for r in nm], 0.36, color=col, label=lab)
    ax.set_xticks(range(len(nm)))
    ax.set_xticklabels(["원본\nP1", "사물\nP1", "맥락\nP1", "사물+맥락", "맥락\nV2", "맥락\nT10"], fontsize=9)
    ax.set_ylabel(f"decoding {t} (CLIP)")
    ax.set_title(f"전처리 a와 d의 {t}")
    ax.spines[["top", "right"]].set_visible(False)
axs[0].legend(frameon=False)
fig.tight_layout()
fig.savefig("docs/report_figures/s1_prepvar_a_vs_d.png", dpi=200)
