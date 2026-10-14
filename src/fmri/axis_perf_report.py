"""B 단계 추가 결과 정리(기준 docs/stageB_axis_perf_criteria.md). 사용: python3 src/fmri/axis_perf_report.py"""
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
Z = {}
for p in Path("data/results/axis_perf").glob("axis_perf_*.npz"):
    if "fast" not in p.name:
        Z.update(dict(np.load(p)))
C = {}
for p in Path("data/results/compress_compare").glob("compress_*_*.npz"):
    if "fast" not in p.name:
        C.update(dict(np.load(p)))
T = ["ctx_P1", "orig_P1", "obj_P1", "clip_img", "dinov2", "resnet50"]
NT = {"ctx_P1": "맥락 라벨", "orig_P1": "원본 캡션", "obj_P1": "사물 이름", "clip_img": "CLIP 이미지", "dinov2": "DINOv2", "resnet50": "ResNet-50"}
M = ["RRR", "DDR", "CCA", "PLS", "PCA"]
n = len(Z["SEL__ctx_P1__RRR__c1"])
rng = np.random.default_rng(6)
idx = rng.integers(0, n, (1000, n))


def bs(v):
    m = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def vd(d, lo, hi, t):
    return "높음" if lo > 0 and d >= t else ("낮음" if hi < 0 and d <= -t else "동등")


L = ["# B 단계 추가: 축별 예측력과 축 선택 결과 (피험자 1, CLIP, 전처리 a, 전체 복셀)", "", f"기준 `docs/stageB_axis_perf_criteria.md`. 이미지 {n}장 5겹. 탐색 단계.", "",
     "## 축별 예측력 r (5폴드 평균)과 높은 축의 수 (무작위 방향 128개의 95번째 백분위 초과)", "",
     "| 타깃 | 무작위 기준 r | " + " | ".join(f"{m}: 높은 축 수 / 축1~5 평균 r / 축28~32 평균 r" for m in M) + " |", "|---|---|" + "---|" * len(M)]
for t in T:
    rr = Z[f"R__{t}__RAND"].mean(0)
    thr = np.percentile(rr, 95)
    row = []
    for m in M:
        r = Z[f"R__{t}__{m}"].mean(0)
        row.append(f"{int((r > thr).sum())} / {r[:5].mean():.2f} / {r[27:32].mean():.2f}")
    L.append(f"| {NT[t]} | {thr:.2f} | " + " | ".join(row) + " |")
L += ["", "## 안정성: 폴드마다 상위 10개 축(r 기준) 번호의 겹침 (폴드 쌍 평균 겹침 비율 / 5폴드 모두 상위 10에 든 축 수)", "", "| 타깃 | " + " | ".join(M) + " |", "|---|" + "---|" * len(M)]
for t in T:
    row = []
    for m in M:
        R = Z[f"R__{t}__{m}"]
        tops = [set(np.argsort(-R[i])[:10]) for i in range(R.shape[0])]
        ov = np.mean([len(tops[i] & tops[j]) / 10 for i in range(len(tops)) for j in range(i + 1, len(tops))])
        always = len(set.intersection(*tops))
        row.append(f"{ov:.2f} / {always}")
    L.append(f"| {NT[t]} | " + " | ".join(row) + " |")
L += ["", "## 축 선택 SEL(32): top-1 / top-5 (앞 32개 축, 압축 없음과 비교)", "", "| 타깃 | 방법 | 앞 32개 top-1 | SEL(32) top-1 | SEL − 앞32 | 판정 | SEL − 압축 없음 | 판정 | top-5 SEL − 압축 없음 | 판정 |", "|---|---|---|---|---|---|---|---|---|---|"]
for t in T:
    nn1, nn5 = C[f"{t}__NONE__0__c1"], C[f"{t}__NONE__0__c5"]
    for m in M:
        s1, s5 = Z[f"SEL__{t}__{m}__c1"], Z[f"SEL__{t}__{m}__c5"]
        f1 = Z[f"FIRST__{t}__{m}__c1"]
        d1, l1, h1 = bs(s1 - f1)
        d2, l2, h2 = bs(s1 - nn1)
        d3, l3, h3 = bs(s5 - nn5)
        L.append(f"| {NT[t]} | {m} | {f1.mean():.3f} | {s1.mean():.3f} | {d1:+.3f} | {vd(d1, l1, h1, 0.02)} | {d2:+.3f} | {vd(d2, l2, h2, 0.02)} | {d3:+.3f} | {vd(d3, l3, h3, 0.03)} |")
L += ["", "## 맥락 − 원본 캡션 (SEL(32), top-1 / top-5)", "", "| 방법 | top-1 차이 | 판정 | top-5 차이 | 판정 |", "|---|---|---|---|---|"]
for m in M:
    row = []
    for x, th in (("c1", 0.02), ("c5", 0.03)):
        d, lo, hi = bs(Z[f"SEL__ctx_P1__{m}__{x}"] - Z[f"SEL__orig_P1__{m}__{x}"])
        row += [f"{d:+.4f} [{lo:+.3f}, {hi:+.3f}]", "맥락 추월" if lo > 0 and d >= th else ("원본이 높음" if hi < 0 and d <= -th else "동등")]
    L.append(f"| {m} | " + " | ".join(row) + " |")
bad = [(t, m) for t in T for m in M if np.nanmean(Z[f"SEL__{t}__{m}__s1"]) > 0.005]
L += ["", "## 셔플 대조 SEL(32) top-1 ≤ 0.005", "", ("모두 충족" if not bad else "초과: " + ", ".join(f"{NT[t]}·{m}" for t, m in bad))]
Path("results/tables/axis_perf.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
col = {"RRR": "#2B6CB0", "DDR": "#D9822B", "CCA": "#2F855A", "PLS": "#9B59B6", "PCA": "#C0392B"}
for ax, t in zip(axs.ravel(), T):
    rr = Z[f"R__{t}__RAND"].mean(0)
    ax.axhspan(np.percentile(rr, 5), np.percentile(rr, 95), color="#ddd", label="무작위 방향(5~95%)")
    for m in M:
        ax.plot(np.arange(1, 129), Z[f"R__{t}__{m}"].mean(0), color=col[m], lw=1.2, label=m)
    ax.set_title(NT[t])
    ax.spines[["top", "right"]].set_visible(False)
axs[1, 0].set_xlabel("축 번호")
axs[0, 0].set_ylabel("축별 예측력 r")
axs[0, 0].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig("docs/report_figures/b_axis_r_curves.png", dpi=200)
