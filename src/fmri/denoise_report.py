"""E3 결과 정리: top-1·top-5 표, D0 대비 판정(docs/denoise_stage1_criteria.md), 핵심 대비 유지, 셔플 확인, 그림.
사용: python3 src/fmri/denoise_report.py
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

KFONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
if Path(KFONT).exists():
    font_manager.fontManager.addfont(KFONT)
    plt.rcParams["font.family"] = ["Apple SD Gothic Neo", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

D = Path("data/results/denoise_decode")
FIG = Path("docs/report_figures")
CONDS = ["D0", "D1_k100", "D1_k300", "D1_k1000", "D2_fwhm1", "D2_fwhm2", "D3"]
REPS = ["orig_P1", "obj_P1", "ctx_P1", "obj_P1+ctx_P1", "ctx_V2", "ctx_T10", "ctx_P1_shuf"]
CONTR = {"K1 맥락P1−사물P1": ("ctx_P1", "obj_P1"), "K2 맥락P1−원본P1": ("ctx_P1", "orig_P1"), "K5 T10−맥락P1": ("ctx_T10", "ctx_P1")}
B = 1000
data = {c: np.load(D / f"denoise_{c}.npz") for c in CONDS}
n = len(data["D0"][f"{REPS[0]}__c1"])
rng = np.random.default_rng(7)
idx = np.stack([rng.integers(0, n, n) for _ in range(B)])


def bs(v):
    m = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def verdict(lo, hi, d, thr):
    if lo > 0 and d >= thr:
        return "개선"
    if hi < 0 and d <= -thr:
        return "악화"
    return "동등"


g = lambda c, r, k: data[c][f"{r}__{k}"]
L = ["# E3 잡음 제거 적용 후 decoding 결과 (피험자 1, CLIP, 복셀 6,728 고정)", "", f"기준 `docs/denoise_stage1_criteria.md`. 이미지 {n}장 5겹. 탐색 단계. 판정: D−D0 구간이 0 제외 + 크기 기준(top-1 0.02, top-5 0.03).", ""]
res = {}
for k, thr, nm in (("c1", 0.02, "top-1"), ("c5", 0.03, "top-5")):
    L += [f"## 표현별 {nm} (D0 대비 차이와 판정)", "", "| 표현 | " + " | ".join(CONDS) + " |", "|---|" + "---|" * len(CONDS)]
    for r in REPS:
        row = []
        for c in CONDS:
            m, _, _ = bs(g(c, r, k))
            if c == "D0":
                row.append(f"{m:.4f}")
            else:
                d, lo, hi = bs(g(c, r, k) - g("D0", r, k))
                v = verdict(lo, hi, d, thr)
                res[(k, r, c)] = (m, d, v)
                row.append(f"{m:.4f} ({d:+.4f}, {v})")
        L.append(f"| {r} | " + " | ".join(row) + " |")
    L.append("")
L += ["## 핵심 대비 (top-1 차이, 판정 = 구간이 0 제외 + |차이| ≥ 0.02)", "", "| 대비 | " + " | ".join(CONDS) + " |", "|---|" + "---|" * len(CONDS)]
for nm, (a, b) in CONTR.items():
    row = []
    for c in CONDS:
        d, lo, hi = bs(g(c, a, "c1") - g(c, b, "c1"))
        row.append(f"{d:+.4f} {'차이 있음' if (lo > 0 or hi < 0) and abs(d) >= 0.02 else '동등'}")
    L.append(f"| {nm} | " + " | ".join(row) + " |")
sh = {c: bs(g(c, "ctx_P1_shuf", "c1"))[0] for c in CONDS}
L += ["", "## 셔플 대조 top-1 (기준 ≤ 0.005)", "", " ".join(f"{c}: {v:.4f}{'' if v <= 0.005 else ' (초과)'}" for c, v in sh.items())]
Path("results/tables/denoise_decode.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))

# 그림 1: 조건별 top-1/top-5 (핵심 표현)
show = ["orig_P1", "obj_P1", "ctx_P1", "ctx_T10"]
nm = {"orig_P1": "원본 캡션", "obj_P1": "사물 이름", "ctx_P1": "맥락 라벨", "ctx_T10": "맥락 T10"}
cols = ["#555", "#2B6CB0", "#D9822B", "#2F855A"]
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, (k, t) in zip(axs, (("c1", "top-1"), ("c5", "top-5"))):
    w = 0.2
    for i, r in enumerate(show):
        ys = [g(c, r, k).mean() for c in CONDS]
        ax.bar(np.arange(len(CONDS)) + (i - 1.5) * w, ys, w * 0.92, color=cols[i], label=nm[r])
    ax.set_xticks(range(len(CONDS)))
    ax.set_xticklabels(["D0\n기준", "PCA\nk100", "PCA\nk300", "PCA\nk1000", "스무딩\nFWHM1", "스무딩\nFWHM2", "백색화"], fontsize=9)
    ax.set_ylabel(f"decoding {t}")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"잡음 제거 조건별 {t}")
axs[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "e3_denoise_top1_top5.png", dpi=200)
plt.close(fig)

# 그림 2: D0 대비 차이 (맥락 P1)
fig, ax = plt.subplots(figsize=(8, 4))
for j, (k, t, col) in enumerate((("c1", "top-1", "#2B6CB0"), ("c5", "top-5", "#D9822B"))):
    for i, c in enumerate(CONDS[1:]):
        d, lo, hi = bs(g(c, "ctx_P1", k) - g("D0", "ctx_P1", k))
        ax.errorbar(d, i + (j - 0.5) * 0.25, xerr=[[d - lo], [hi - d]], fmt="o", color=col, capsize=3, label=t if i == 0 else None)
ax.axvline(0, color="#999", lw=0.8)
ax.set_yticks(range(6))
ax.set_yticklabels(["PCA k100", "PCA k300", "PCA k1000", "스무딩 FWHM1", "스무딩 FWHM2", "백색화"])
ax.invert_yaxis()
ax.set_xlabel("D0 대비 차이 (맥락 라벨 P1, 막대 = 95% 구간)")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "e3_denoise_diff_ctx.png", dpi=200)
plt.close(fig)
