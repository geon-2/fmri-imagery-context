"""B 단계 2차 확인(피험자 1·2·5·7) 정리. 기준 docs/stageB_compression_criteria.md의 「2단계」. 사용: python3 src/fmri/compress_subjects_report.py"""
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
S = {"01": {}, "02": {}, "05": {}, "07": {}}
for p in D.glob("compress_*.npz"):
    if "fast" in p.name:
        continue
    key = "01"
    for s in ("02", "05", "07"):
        if f"_s{s}_" in p.name:
            key = s
    S[key].update(dict(np.load(p)))
T = ["ctx_P1", "orig_P1", "obj_P1", "clip_img", "dinov2", "resnet50"]
NT = {"ctx_P1": "맥락 라벨", "orig_P1": "원본 캡션", "obj_P1": "사물 이름", "clip_img": "CLIP 이미지", "dinov2": "DINOv2", "resnet50": "ResNet-50"}
M = ["RRR", "DDR", "CCA", "PLS", "PCA", "VPCA"]
SUBJ = ["01", "02", "05", "07"]
rng = np.random.default_rng(2)
g = lambda s, t, m, k, x="c1": S[s][f"{t}__{m}__{k}__{x}"]


def bs(v):
    m = v[rng.integers(0, len(v), (1000, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def vd(d, lo, hi, t):
    return "높음" if lo > 0 and d >= t else ("낮음" if hi < 0 and d <= -t else "동등")


L = ["# B 단계 2차 확인: 피험자 1·2·5·7 (k=32, 압축 없음 대비)", "", "기준 `docs/stageB_compression_criteria.md`의 2단계. 판정은 피험자별 이미지 단위 부트스트랩 쌍 비교. 「재현」은 4명 중 3명 이상에서 같은 판정일 때만.", "",
     "## 판정 1: 압축 이득 (top-1 차이 평균 / 판정 인원 높음-동등-낮음 / 재현 여부)", "", "| 방법 | " + " | ".join(NT[t] for t in T) + " |", "|---|" + "---|" * len(T)]
mean_gain = {}
for m in M:
    row = []
    for t in T:
        ds, vs = [], []
        for s in SUBJ:
            d, lo, hi = bs(g(s, t, m, 32) - g(s, t, "NONE", 0))
            ds.append(d)
            vs.append(vd(d, lo, hi, 0.02))
        mean_gain[(t, m)] = ds
        c = {k: vs.count(k) for k in ("높음", "동등", "낮음")}
        rep = [k for k, v in c.items() if v >= 3]
        row.append(f"{np.mean(ds):+.3f} / {c['높음']}-{c['동등']}-{c['낮음']} / {rep[0] if rep else '일관 안 됨'}")
    L.append(f"| {m} | " + " | ".join(row) + " |")
L += ["", "## 판정 3: 캡션 기준선 (맥락 − 원본, top-1)", "", "| 방법 | 피험자별 차이 (01/02/05/07) | 원본이 높음-동등-맥락이 높음 인원 | 재현 |", "|---|---|---|---|"]
for m, k in [("NONE", 0)] + [(m, 32) for m in M]:
    ds, vs = [], []
    for s in SUBJ:
        d, lo, hi = bs(g(s, "ctx_P1", m, k) - g(s, "orig_P1", m, k))
        ds.append(d)
        vs.append("원본이 높음" if hi < 0 and d <= -0.02 else ("맥락이 높음" if lo > 0 and d >= 0.02 else "동등"))
    c = {x: vs.count(x) for x in ("원본이 높음", "동등", "맥락이 높음")}
    rep = [x for x, v in c.items() if v >= 3]
    L.append(f"| {m} | {' / '.join(f'{x:+.3f}' for x in ds)} | {c['원본이 높음']}-{c['동등']}-{c['맥락이 높음']} | {rep[0] if rep else '일관 안 됨'} |")
L += ["", "## 판정 4: 맥락 − 사물 (K1, top-1)", "", "| 방법 | 피험자별 차이 | 재현 |", "|---|---|---|"]
for m, k in [("NONE", 0)] + [(m, 32) for m in M]:
    ds, vs = [], []
    for s in SUBJ:
        d, lo, hi = bs(g(s, "ctx_P1", m, k) - g(s, "obj_P1", m, k))
        ds.append(d)
        vs.append(vd(d, lo, hi, 0.02))
    L.append(f"| {m} | {' / '.join(f'{x:+.3f}' for x in ds)} | {'높음 재현' if vs.count('높음') >= 3 else '일관 안 됨'} |")
sh = {s: max(np.nanmean(g(s, t, m, 32 if m != 'NONE' else 0, 's1')) for t in T if f"{t}__NONE__0__c1" in S[s] for m in M + ["NONE"]) for s in SUBJ}
L += ["", "## 판정 5: 셔플 대조 (피험자별 최댓값, 기준 ≤ 0.005)", "", ", ".join(f"{s}: {v:.4f}" for s, v in sh.items()) + (" — 모두 충족" if max(sh.values()) <= 0.005 else " — 초과 있음")]
Path("results/tables/compress_subjects.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
fig, axs = plt.subplots(1, len(T), figsize=(17, 4), sharey=True)
col = ["#2B6CB0", "#D9822B", "#2F855A", "#9B59B6", "#C0392B", "#17A2B8"]
for ax, t in zip(axs, T):
    for i, m in enumerate(M):
        v = mean_gain[(t, m)]
        ax.scatter([i] * 4, v, color="#bbb", s=14, zorder=2)
        ax.scatter(i, np.mean(v), color=col[i], s=40, zorder=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(M)))
    ax.set_xticklabels(M, rotation=45, fontsize=8)
    ax.set_title(NT[t], fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel("압축 없음 대비 top-1 차이 (k=32, 회색=피험자별)")
fig.tight_layout()
fig.savefig("docs/report_figures/b_compress_subjects.png", dpi=200)
