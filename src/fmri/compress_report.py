"""B 단계 결과 정리(기준 docs/stageB_compression_criteria.md). 사용: python3 src/fmri/compress_report.py"""
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
D = Path("data/results/compress_compare")
Z = {}
for p in D.glob("compress_*_*.npz"):
    if "fast" in p.name:
        continue
    Z.update(dict(np.load(p)))
KS = [5, 8, 12, 20, 32, 50, 80, 128]
T = ["ctx_P1", "orig_P1", "obj_P1", "clip_img", "dinov2", "resnet50"]
NT = {"ctx_P1": "맥락 라벨", "orig_P1": "원본 캡션", "obj_P1": "사물 이름", "clip_img": "CLIP 이미지", "dinov2": "DINOv2", "resnet50": "ResNet-50"}
M = ["RRR", "DDR", "CCA", "PLS", "PCA", "RAND", "VPCA"]
n = len(Z["ctx_P1__NONE__0__c1"])
rng = np.random.default_rng(5)
idx = rng.integers(0, n, (1000, n))
g = lambda t, m, k, x="c1": Z[f"{t}__{m}__{k}__{x}"]


def bs(v):
    m = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def vd(d, lo, hi, t):
    return "높음" if lo > 0 and d >= t else ("낮음" if hi < 0 and d <= -t else "동등")


L = ["# B 단계: 복셀·축 압축 방법 비교 결과 (피험자 1, CLIP, 전처리 a, 전체 복셀)", "", f"기준 `docs/stageB_compression_criteria.md`. 이미지 {n}장 5겹. 주 지표는 복원한 전체 타깃(표준화 공간)에서 보류 이미지 1장 찾기. 주 k = 32. 탐색 단계.", "",
     "## 압축 없음(기준선) top-1 / top-5와 셔플", "", "| 타깃 | top-1 | top-5 | 셔플 top-1 |", "|---|---|---|---|"]
for t in T:
    L.append(f"| {NT[t]} | {g(t, 'NONE', 0).mean():.4f} | {g(t, 'NONE', 0, 'c5').mean():.4f} | {np.nanmean(g(t, 'NONE', 0, 's1')):.4f} |")
L += ["", "## 방법별 top-1 곡선 (k = " + ", ".join(map(str, KS)) + ")", ""]
for t in T:
    L += [f"### {NT[t]}", "", "| 방법 | " + " | ".join(f"k={k}" for k in KS) + " |", "|---|" + "---|" * len(KS)]
    for m in M:
        L.append(f"| {m} | " + " | ".join(f"{g(t, m, k).mean():.3f}" for k in KS) + " |")
    L.append("")
L += ["## 판정 1: 압축 이득 (방법 k=32 − 압축 없음, top-1 / top-5, 판정)", "", "| 방법 | " + " | ".join(NT[t] for t in T) + " |", "|---|" + "---|" * len(T)]
for m in M:
    row = []
    for t in T:
        r = []
        for x, th in (("c1", 0.02), ("c5", 0.03)):
            d, lo, hi = bs(g(t, m, 32, x) - g(t, "NONE", 0, x))
            r.append(f"{d:+.3f} {vd(d, lo, hi, th)}")
        row.append(" / ".join(r))
    L.append(f"| {m} | " + " | ".join(row) + " |")
L += ["", "## 판정 2: 방법 간 (k=32, top-1 점수와 무작위 대비)", "", "| 타깃 | " + " | ".join(M) + " | 무작위보다 높은 방법 수 |", "|---|" + "---|" * (len(M) + 1)]
for t in T:
    sc = [g(t, m, 32).mean() for m in M]
    hi = 0
    for m in M:
        if m == "RAND":
            continue
        d, lo, h = bs(g(t, m, 32) - g(t, "RAND", 32))
        hi += int(lo > 0 and d >= 0.02)
    L.append(f"| {NT[t]} | " + " | ".join(f"{x:.3f}" for x in sc) + f" | {hi}/{len(M) - 1} |")
L += ["", "## 판정 3: 캡션 기준선 (맥락 − 원본, k=32, 압축 없음 대비)", "", "| 방법 | top-1 차이 | 판정 | top-5 차이 | 판정 |", "|---|---|---|---|---|"]
for m in ["NONE"] + M:
    k = 0 if m == "NONE" else 32
    row = []
    for x, th in (("c1", 0.02), ("c5", 0.03)):
        d, lo, hi = bs(g("ctx_P1", m, k, x) - g("orig_P1", m, k, x))
        v = "맥락 추월" if lo > 0 and d >= th else ("원본이 높음" if hi < 0 and d <= -th else "격차 없음(동등)")
        row += [f"{d:+.4f} [{lo:+.3f}, {hi:+.3f}]", v]
    L.append(f"| {m} | " + " | ".join(row) + " |")
L += ["", "## 판정 4: 맥락 − 사물 (K1, k=32)", "", "| 방법 | top-1 차이 | 판정 |", "|---|---|---|"]
for m in ["NONE"] + M:
    k = 0 if m == "NONE" else 32
    d, lo, hi = bs(g("ctx_P1", m, k) - g("obj_P1", m, k))
    L.append(f"| {m} | {d:+.4f} | {vd(d, lo, hi, 0.02)} |")
bad = [(t, m) for t in T for m in M + ["NONE"] if np.nanmean(g(t, m, 32 if m != "NONE" else 0, "s1")) > 0.005]
L += ["", "## 판정 5: 셔플 대조 (k=32, 압축 없음), top-1 ≤ 0.005", "", ("모두 충족" if not bad else "초과: " + ", ".join(f"{NT[t]}·{m}" for t, m in bad))]
L += ["", "## 참고: 축으로 완벽 복원했을 때의 상한(oracle) top-1, k=32", "", "| 방법 | " + " | ".join(NT[t] for t in T) + " |", "|---|" + "---|" * len(T)]
for m in M[:-1]:
    L.append(f"| {m} | " + " | ".join(f"{np.nanmean(g(t, m, 32, 'o1')):.3f}" for t in T) + " |")
Path("results/tables/compress_compare.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))

fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
col = {"RRR": "#2B6CB0", "DDR": "#D9822B", "CCA": "#2F855A", "PLS": "#9B59B6", "PCA": "#C0392B", "RAND": "#bbb", "VPCA": "#17A2B8"}
for ax, t in zip(axs.ravel(), T):
    for m in M:
        ax.plot(KS, [g(t, m, k).mean() for k in KS], marker="o", ms=3, color=col[m], label=m)
    ax.axhline(g(t, "NONE", 0).mean(), color="k", ls="--", lw=1, label="압축 없음")
    ax.set_xscale("log")
    ax.set_title(NT[t])
    ax.spines[["top", "right"]].set_visible(False)
axs[1, 0].set_xlabel("k")
axs[0, 0].set_ylabel("top-1")
axs[0, 0].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig("docs/report_figures/b_compress_curves.png", dpi=200)
