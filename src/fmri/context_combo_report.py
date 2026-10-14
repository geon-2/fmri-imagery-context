"""맥락 조합 비교 정리(기준 docs/stageA_context_combo_criteria.md). 사용: python3 src/fmri/context_combo_report.py"""
import itertools
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
D = Path("data/results/context_combo")
SUBJ = ["01", "02", "05", "07"]
DIMS = ["place_type", "environment", "activity", "implied_event", "scale", "notes"]
KO = {"place_type": "장소 유형", "environment": "환경 속성", "activity": "활동", "implied_event": "사건·의도", "scale": "공간 규모", "notes": "기타 보충"}
Z = {s: np.load(D / f"context_combo_{s}.npz") for s in SUBJ}
rng = np.random.default_rng(9)


def get(s, name, m):
    z = Z[s]
    if name.endswith("_avg"):
        base = name[:-4]
        return np.nanmean(np.stack([z[f"{base}_s{k}__{m}"] for k in range(5)]), axis=0)
    return z[f"{name}__{m}"]


def bs(v, B=1000):
    v = v[~np.isnan(v)]
    ii = rng.integers(0, len(v), (B, len(v)))
    m = v[ii].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def vd(d, lo, hi, t):
    return "높음" if lo > 0 and d >= t else ("낮음" if hi < 0 and d <= -t else "동등")


conds = ["all6", "orig_P1", "obj_P1"] + [f"single_{d}" for d in DIMS] + [f"loo_{d}" for d in DIMS] + [f"pair_{a}+{b}" for a, b in itertools.combinations(DIMS, 2)] + ["half_avg", "one_avg"]


def label(c):
    if c.startswith("single_"):
        return f"한 차원만: {KO[c[7:]]}"
    if c.startswith("loo_"):
        return f"한 차원 제외: {KO[c[4:]]}"
    if c.startswith("pair_"):
        a, b = c[5:].split("+")
        return f"두 차원: {KO[a]}+{KO[b]}"
    return {"all6": "6차원 전부(기준)", "orig_P1": "원본 캡션", "obj_P1": "사물 이름", "half_avg": "각 차원 라벨 절반(시드 5평균)", "one_avg": "각 차원 라벨 1개(시드 5평균)"}[c]


L = ["# 맥락 정보 조합 비교 결과 (피험자 1·2·5·7, CLIP, 전처리 a, 전체 복셀)", "", "기준 `docs/stageA_context_combo_criteria.md`. 판정: 구간이 0 제외 + top-1 ≥ 0.02(top-5 ≥ 0.03). 「일관」은 4명 중 3명 이상 같은 방향.", "",
     "| 조건 | 피험자별 top-1 (01/02/05/07) | 4명 평균 top-1 | 평균 top-5 | 상한 | 기준 대비 판정(높음/동등/낮음 인원) | 원본 캡션 대비 판정 |", "|---|---|---|---|---|---|---|"]
summ = {}
for c in conds:
    t1 = [np.nanmean(get(s, c, "c1")) for s in SUBJ]
    t5 = [np.nanmean(get(s, c, "c5")) for s in SUBJ]
    ce = [np.nanmean(get(s, c, "ceil")) for s in SUBJ]
    res = []
    for ref in ("all6", "orig_P1"):
        cnt = {"높음": 0, "동등": 0, "낮음": 0}
        ds = []
        for s in SUBJ:
            d1 = get(s, c, "c1") - get(s, ref, "c1")
            d5 = get(s, c, "c5") - get(s, ref, "c5")
            m1, l1, h1 = bs(d1)
            cnt[vd(m1, l1, h1, 0.02)] += 1
            ds.append(m1)
        res.append((cnt, ds))
    summ[c] = (t1, res)
    fmt = lambda cnt: f"{cnt['높음']}/{cnt['동등']}/{cnt['낮음']}" if c not in ("all6",) else "-"
    L.append(f"| {label(c)} | {' / '.join(f'{x:.3f}' for x in t1)} | {np.mean(t1):.4f} | {np.mean(t5):.4f} | {np.mean(ce):.3f} | " + (fmt(res[0][0]) if c != "all6" else "-") + " | " + (fmt(res[1][0]) if c != "orig_P1" else "-") + " |")
sh = [np.nanmean(Z[s]["all6_shuf__c1"]) for s in SUBJ]
L += ["", f"셔플 대조 top-1(피험자별): {', '.join(f'{x:.4f}' for x in sh)} (기준 ≤ 0.005 {'충족' if max(sh) <= 0.005 else '미충족'})"]
L += ["", "## 일관 판정 (4명 중 3명 이상)", ""]
for c in conds[3:]:
    cnt = {"높음": 0, "동등": 0, "낮음": 0}
    for s in SUBJ:
        m1, l1, h1 = bs(get(s, c, "c1") - get(s, "all6", "c1"))
        cnt[vd(m1, l1, h1, 0.02)] += 1
    kind = [k for k, v in cnt.items() if v >= 3]
    L.append(f"- {label(c)}: 기준 대비 {kind[0] if kind else '일관되지 않음'} ({cnt['높음']}/{cnt['동등']}/{cnt['낮음']})")
Path("results/tables/context_combo.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))

fig, ax = plt.subplots(figsize=(9, 9))
order = conds[3:]
means = [np.mean(summ[c][0]) for c in order]
base = np.mean(summ["all6"][0])
orig = np.mean(summ["orig_P1"][0])
y = np.arange(len(order))
for i, c in enumerate(order):
    ax.scatter(summ[c][0], [i] * 4, color="#bbb", s=14, zorder=2)
ax.scatter(means, y, color="#2B6CB0", s=30, zorder=3, label="4명 평균")
ax.axvline(base, color="#D9822B", ls="--", label="6차원 전부")
ax.axvline(orig, color="#555", ls=":", label="원본 캡션")
ax.set_yticks(y)
ax.set_yticklabels([label(c) for c in order], fontsize=7)
ax.invert_yaxis()
ax.set_xlabel("decoding top-1 (회색 = 피험자별)")
ax.legend(frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("docs/report_figures/a_context_combo.png", dpi=200)
