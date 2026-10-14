"""E4 결과 정리(기준 docs/roi_weight_criteria.md). 사용: python3 src/fmri/roi_weight_report.py"""
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
D = Path("data/results/roi_weight")
z = np.load(D / "roi_weight.npz")
meta = json.load(open(D / "roi_weight_meta.json", encoding="utf-8"))
G = meta["groups"]
DIMS = ["장소 유형", "환경 속성", "활동", "암묵적 사건·의도", "공간 규모", "기타 보충"]
TG = ["orig_P1", "obj_P1", "ctx_P1"]
n = meta["n"]
rng = np.random.default_rng(3)
idx = np.stack([rng.integers(0, n, n) for _ in range(1000)])


def bs(v, ok=None):
    ok = np.ones(len(v), bool) if ok is None else ok
    vv = v[ok]
    ii = np.stack([rng.integers(0, len(vv), len(vv)) for _ in range(1000)])
    m = vv[ii].mean(1)
    return float(vv.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def verdict(d, lo, hi, t):
    return "개선" if lo > 0 and d >= t else ("악화" if hi < 0 and d <= -t else "동등")


L = ["# E4 ROI 분할·차원별 가중치 결과 (피험자 1, CLIP, 전처리 a)", "", f"기준 `docs/roi_weight_criteria.md`. 이미지 {n}장 5겹. 탐색 단계.", "",
     "## 시험 A: 그룹별 decoding (top-1 / top-5, 괄호는 ALL 대비 top-1 차이와 판정)", "", "| 그룹 | 복셀 수 | " + " | ".join(f"{t} top-1" for t in TG) + " | " + " | ".join(f"{t} top-5" for t in TG) + " |", "|---|---|---|---|---|---|---|---|"]
for g in G:
    row = [g, f"{meta['n_voxels'][g]:,}"]
    for k, t in (("c1", 0.02), ("c5", 0.03)):
        for r in TG:
            v = z[f"A__{g}__{r}__{k}"]
            m = v.mean()
            if g == "ALL":
                row.append(f"{m:.4f}")
            else:
                d, lo, hi = bs(v - z[f"A__ALL__{r}__{k}"])
                row.append(f"{m:.4f} ({d:+.3f}, {verdict(d, lo, hi, t)})")
    L.append("| " + " | ".join(row) + " |")
L += ["", "## 시험 B: 차원 × 그룹 top-1 (우연 수준은 라벨 겹침으로 생기는 값)", "", "| 차원 (이미지 수, 우연) | " + " | ".join(G) + " |", "|---|" + "---|" * len(G)]
mat1 = np.zeros((6, len(G)))
mat5 = np.zeros((6, len(G)))
for d in range(6):
    row = [f"{DIMS[d]} ({meta['dim_images'][d]:,}, {meta['chance_dim_c1'][str(d)]:.3f})"]
    for gi, g in enumerate(G):
        v = z[f"B__{d}__{g}__c1"]
        mat1[d, gi] = np.nanmean(v)
        mat5[d, gi] = np.nanmean(z[f"B__{d}__{g}__c5"])
        row.append(f"{mat1[d, gi]:.3f}")
    L.append("| " + " | ".join(row) + " |")
L += ["", "같은 행렬의 top-5:", "", "| 차원 | " + " | ".join(G) + " |", "|---|" + "---|" * len(G)]
for d in range(6):
    L.append(f"| {DIMS[d]} | " + " | ".join(f"{mat5[d, gi]:.3f}" for gi in range(len(G))) + " |")
F = ["F0_ctx_P1_all", "F1_dim_equal", "F2_dim_learned", "F3_roi_equal", "F4_dimroi_equal", "F5_dimroi_learned"]
L += ["", "## 시험 C: 결합과 가중치", "", "| 방법 | top-1 | F0 대비 | 판정 | top-5 | F0 대비 | 판정 |", "|---|---|---|---|---|---|---|"]
for f in F:
    row = [f]
    for k, t in (("c1", 0.02), ("c5", 0.03)):
        v = z[f"F__{f}__{k}"]
        if f == F[0]:
            row += [f"{v.mean():.4f}", "-", "-"]
        else:
            d, lo, hi = bs(v - z[f"F__{F[0]}__{k}"])
            row += [f"{v.mean():.4f}", f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]", verdict(d, lo, hi, t)]
    L.append("| " + " | ".join(row) + " |")
L += ["", "가중치 효과", ""]
for a_, b_ in (("F2_dim_learned", "F1_dim_equal"), ("F5_dimroi_learned", "F4_dimroi_equal")):
    parts = []
    for k, t, nm in (("c1", 0.02, "top-1"), ("c5", 0.03, "top-5")):
        d, lo, hi = bs(z[f"F__{a_}__{k}"] - z[f"F__{b_}__{k}"])
        parts.append(f"{nm} {d:+.4f} [{lo:+.4f}, {hi:+.4f}] {verdict(d, lo, hi, t)}")
    L.append(f"- {a_} − {b_}: " + ", ".join(parts))
W = np.array(meta["weights_mean"])
L += ["", "## 학습된 가중치 평균 (5폴드, 차원 × 그룹; AUC − 0.5)", "", "| 차원 | " + " | ".join(G) + " |", "|---|" + "---|" * len(G)]
for d in range(6):
    L.append(f"| {DIMS[d]} | " + " | ".join(f"{W[d, gi]:.3f}" for gi in range(len(G))) + " |")
Path("results/tables/roi_weight.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))

fig, ax = plt.subplots(figsize=(11, 4.4))
w = 0.26
for i, (r, c) in enumerate(zip(TG, ["#555", "#2B6CB0", "#D9822B"])):
    ax.bar(np.arange(len(G)) + (i - 1) * w, [z[f"A__{g}__{r}__c1"].mean() for g in G], w * 0.92, color=c, label={"orig_P1": "원본 캡션", "obj_P1": "사물 이름", "ctx_P1": "맥락 라벨"}[r])
ax.set_xticks(range(len(G)))
ax.set_xticklabels([f"{g}\n{meta['n_voxels'][g]:,}" for g in G], fontsize=8)
ax.set_ylabel("decoding top-1")
ax.set_title("복셀 그룹별 decoding top-1 (그룹 이름과 복셀 수)")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("docs/report_figures/e4_roi_groups.png", dpi=200)
plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
im = axs[0].imshow(mat1, cmap="Blues", aspect="auto")
axs[0].set_title("차원 × 그룹 top-1")
axs[1].imshow(mat1 - np.array([meta["chance_dim_c1"][str(d)] for d in range(6)])[:, None], cmap="Blues", aspect="auto")
axs[1].set_title("우연 수준을 뺀 값")
for ax in axs:
    ax.set_xticks(range(len(G)))
    ax.set_xticklabels(G, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(6))
    ax.set_yticklabels(DIMS, fontsize=9)
for ax, M in zip(axs, (mat1, mat1 - np.array([meta["chance_dim_c1"][str(d)] for d in range(6)])[:, None])):
    for i in range(6):
        for j in range(len(G)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig.tight_layout()
fig.savefig("docs/report_figures/e4_dim_roi_matrix.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.5, 4))
for j, (k, col, nm) in enumerate((("c1", "#2B6CB0", "top-1"), ("c5", "#D9822B", "top-5"))):
    for i, f in enumerate(F[1:]):
        d, lo, hi = bs(z[f"F__{f}__{k}"] - z[f"F__{F[0]}__{k}"])
        ax.errorbar(d, i + (j - 0.5) * 0.25, xerr=[[d - lo], [hi - d]], fmt="o", color=col, capsize=3, label=nm if i == 0 else None)
ax.axvline(0, color="#999", lw=0.8)
ax.set_yticks(range(5))
ax.set_yticklabels(F[1:], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("F0(전체 복셀, ctx_P1) 대비 차이 (막대 = 95% 구간)")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("docs/report_figures/e4_fusion_diff.png", dpi=200)
