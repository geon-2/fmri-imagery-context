"""타깃 종류 비교 정리(기준 docs/stageA_target_type_criteria.md). 사용: python3 src/fmri/target_type_report.py"""
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
z = np.load("data/results/target_type_compare/target_type_scores.npz")
g = lambda r, k: z[f"{r}__{k}"]
n = len(g("orig_P1", "c1"))
rng = np.random.default_rng(4)
idx = np.stack([rng.integers(0, n, n) for _ in range(1000)])


def bs(v):
    m = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def vd(d, lo, hi, t):
    return "높음" if lo > 0 and d >= t else ("낮음" if hi < 0 and d <= -t else "동등")


NM = {"orig_P1": "원본 캡션 P1", "obj_P1": "사물 이름 P1", "ctx_P1": "맥락 라벨 P1", "clip_img": "CLIP 이미지", "dinov2": "DINOv2", "resnet50": "ResNet-50"}
R = list(NM)
L = ["# A 단계 확장: 타깃 종류 비교 결과 (피험자 1, 전처리 a, CLIP 텍스트·이미지 외 DINOv2·ResNet-50)", "", f"기준 `docs/stageA_target_type_criteria.md`. 이미지 {n}장 5겹. 탐색 단계.", "",
     "| 타깃 | top-1 [95% 구간] | top-5 [95% 구간] | 셔플 top-1 |", "|---|---|---|---|"]
for r in R:
    m1, l1, h1 = bs(g(r, "c1"))
    m5, l5, h5 = bs(g(r, "c5"))
    sh = f"{g(r + '_shuf', 'c1').mean():.4f}" if f"{r}_shuf__c1" in z else "-"
    L.append(f"| {NM[r]} | {m1:.4f} [{l1:.4f}, {h1:.4f}] | {m5:.4f} [{l5:.4f}, {h5:.4f}] | {sh} |")
L += ["", "## 쌍 비교 (A − B, top-1 / top-5)", "", "| A − B | top-1 차이 | 판정 | top-5 차이 | 판정 |", "|---|---|---|---|---|"]
for a_, b_ in (("clip_img", "orig_P1"), ("clip_img", "ctx_P1"), ("resnet50", "orig_P1"), ("resnet50", "ctx_P1"), ("dinov2", "orig_P1"), ("dinov2", "ctx_P1"),
               ("clip_img", "dinov2"), ("clip_img", "resnet50"), ("resnet50", "dinov2"), ("clip_img", "obj_P1")):
    row = []
    for k, t in (("c1", 0.02), ("c5", 0.03)):
        d, lo, hi = bs(g(a_, k) - g(b_, k))
        row += [f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]", vd(d, lo, hi, t)]
    L.append(f"| {NM[a_]} − {NM[b_]} | " + " | ".join(row) + " |")
sh_ok = all(g(r + "_shuf", "c1").mean() <= 0.005 for r in ("clip_img", "dinov2", "resnet50", "ctx_P1"))
L += ["", f"셔플 대조 top-1 ≤ 0.005: {'모두 충족' if sh_ok else '미충족 있음'}"]
Path("results/tables/target_type_compare.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
cols = ["#555", "#999", "#D9822B", "#2B6CB0", "#2F855A", "#9B59B6"]
for ax, (k, t) in zip(axs, (("c1", "top-1"), ("c5", "top-5"))):
    ys = [g(r, k).mean() for r in R]
    lo = [g(r, k).mean() - bs(g(r, k))[1] for r in R]
    hi = [bs(g(r, k))[2] - g(r, k).mean() for r in R]
    ax.bar(range(len(R)), ys, color=cols, yerr=[lo, hi], capsize=3)
    ax.set_xticks(range(len(R)))
    ax.set_xticklabels([NM[r].replace(" ", "\n") for r in R], fontsize=8)
    ax.set_ylabel(f"decoding {t}")
    ax.set_title(f"타깃 종류별 {t}")
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("docs/report_figures/a_target_type.png", dpi=200)
