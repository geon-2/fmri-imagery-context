"""E1 결과 정리: 표, 판정(기준은 docs/prep_axes_criteria.md), 그림 3장(PNG).
사용: python3 src/fmri/prep_sensitivity_report.py [--in-dir data/results/prep_sensitivity] [--fig-dir docs/report_figures]
"""
import argparse
import itertools
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

LABEL = {"orig_P1": "원본 캡션(P1)", "orig_V2": "원본 캡션(V2)", "obj_P1": "사물 이름(P1)", "ctx_P1": "맥락 라벨(P1)",
         "ctx_V2": "맥락 라벨(V2)", "obj_P1+ctx_P1": "사물+맥락(P1)"}
CONDS = list(LABEL)
VNAME = {"a": "a 현재(z-score+신뢰도≥0.1)", "b": "b z-score+선택 없음", "c": "c 표준화·선택 없음(MIRAGE식)", "d": "d z-score+SNR 상위 20%"}
COL = {"a": "#2B6CB0", "b": "#7B4FD6", "c": "#D9822B", "d": "#0F8F7F"}
CON = {"C1_ctx_minus_obj": ("ctx_P1", "obj_P1", "맥락 − 사물"), "C2_ctx_minus_orig": ("ctx_P1", "orig_P1", "맥락 − 원본 캡션"),
       "C3_objctx_minus_orig": ("obj_P1+ctx_P1", "orig_P1", "(사물+맥락) − 원본 캡션"), "C4_ctxV2_minus_ctxP1": ("ctx_V2", "ctx_P1", "맥락 V2 − 맥락 P1")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/results/prep_sensitivity")
    ap.add_argument("--model", default="MPNet")
    ap.add_argument("--fig-dir", default="docs/report_figures")
    ap.add_argument("--table", default="results/tables/prep_sensitivity.md")
    a = ap.parse_args()
    res = json.load(open(Path(a.in_dir) / f"prep_sensitivity_{a.model}.json", encoding="utf-8"))
    npz = np.load(Path(a.in_dir) / f"prep_sensitivity_{a.model}_boots.npz")
    V = list(res["variants"])
    P = {v: res["variants"][v]["P"] for v in V}
    boots = {v: {c: npz[f"{v}__{c}"] for c in P[v]} for v in V}
    Path(a.fig_dir).mkdir(parents=True, exist_ok=True)

    # --- 표와 판정
    ranks = {v: {c: int(r) for r, c in enumerate(sorted(CONDS, key=lambda c: -P[v][c]), 1)} for v in V}
    taus = {}
    for u, w in itertools.combinations(V, 2):
        taus[f"{u}-{w}"] = float(kendalltau([P[u][c] for c in CONDS], [P[w][c] for c in CONDS])[0])
    cons = {}
    for k, (x, y, _) in CON.items():
        cons[k] = {}
        for v in V:
            d = boots[v][x] - boots[v][y]
            diff = P[v][x] - P[v][y]
            lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
            cons[k][v] = {"diff": float(diff), "ci": [lo, hi], "sign": int(np.sign(diff)), "sig": bool(lo > 0 or hi < 0)}
    robust = {k: (len({cons[k][v]["sign"] for v in V}) == 1 and len({cons[k][v]["sig"] for v in V}) == 1) for k in CON}
    flips = {}
    for k, (x, y, _) in CON.items():
        flips[k] = {f"{u}-{w}": float(np.mean(np.sign(boots[u][x] - boots[u][y]) != np.sign(boots[w][x] - boots[w][y])))
                    for u, w in itertools.combinations(V, 2)}
    tau_ok = all(t >= 0.8 for t in taus.values())
    verdict = "전처리는 결론에 영향이 없다(주 설정 a 유지)" if all(robust.values()) and tau_ok else "전처리 의존(대비별로 보고, 주 설정은 사용자 결정)"
    out = {"P": P, "ranks": ranks, "kendall_tau": taus, "contrasts": cons, "robust": robust, "tau_all_ge_0.8": tau_ok, "flip_rate": flips,
           "verdict": verdict, "n_voxels": {v: res["variants"][v]["n_voxels"] for v in V}, "n_images": res["variants"][V[0]]["n_images"]}
    json.dump(out, open(Path(a.in_dir) / "prep_sensitivity_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    L = ["# E1 전처리 강건성 점검 결과", "", f"피험자 1, {a.model}, 학습용 이미지 {out['n_images']}장, 5겹 교차검증(탐색). 기준: `docs/prep_axes_criteria.md`.", "",
         "## 조건별 P (R2·R3 복셀 r 중앙값의 평균)", "", "| 조건 | " + " | ".join(f"{v} (복셀 {out['n_voxels'][v]:,})" for v in V) + " |", "|---|" + "---|" * len(V)]
    for c in CONDS:
        L.append(f"| {LABEL[c]} | " + " | ".join(f"{P[v][c]:.4f} (순위 {ranks[v][c]})" for v in V) + " |")
    L += ["", "## 변형 쌍별 순위 Kendall τ (기준 ≥ 0.8)", "", ", ".join(f"{k}: {t:.2f}" for k, t in taus.items()), "", "## 핵심 대비", "",
          "| 대비 | " + " | ".join(V) + " | 강건 | 최대 뒤집힘 비율 |", "|---|" + "---|" * (len(V) + 2)]
    for k, (_, _, nm) in CON.items():
        L.append(f"| {nm} | " + " | ".join(f"{cons[k][v]['diff']:+.4f} [{cons[k][v]['ci'][0]:+.4f}, {cons[k][v]['ci'][1]:+.4f}]" for v in V) +
                 f" | {'예' if robust[k] else '아니오'} | {max(flips[k].values()):.2f} |")
    L += ["", f"## 판정\n\n**{verdict}** (대비 강건 {sum(robust.values())}/4, 순위 τ 기준 {'충족' if tau_ok else '미충족'})"]
    Path(a.table).parent.mkdir(parents=True, exist_ok=True)
    Path(a.table).write_text("\n".join(L), encoding="utf-8")

    # --- 그림 1: 조건별 P, 변형 비교
    fig, ax = plt.subplots(figsize=(11, 5.2))
    w = 0.8 / len(V)
    for i, v in enumerate(V):
        xs = np.arange(len(CONDS)) + (i - (len(V) - 1) / 2) * w
        ys = [P[v][c] for c in CONDS]
        err = [[P[v][c] - np.percentile(boots[v][c], 2.5) for c in CONDS], [np.percentile(boots[v][c], 97.5) - P[v][c] for c in CONDS]]
        ax.bar(xs, ys, w * 0.92, color=COL[v], label=VNAME[v], yerr=err, capsize=2, ecolor="#333", error_kw={"lw": 0.8})
    ax.set_xticks(range(len(CONDS)))
    ax.set_xticklabels([LABEL[c] for c in CONDS], fontsize=10)
    lo = min(P[v][c] for v in V for c in CONDS)
    ax.set_ylim(max(0, lo - 0.05), max(P[v][c] for v in V for c in CONDS) + 0.025)
    ax.set_ylabel("P (R2·R3 복셀 r 중앙값의 평균)")
    ax.set_title("전처리 변형별 텍스트 조건의 encoding 점수 (피험자 1, 막대 오차 = 95% 부트스트랩)")
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.07), frameon=False, ncol=4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(Path(a.fig_dir) / "e1_prep_scores.png", dpi=200)
    plt.close(fig)

    # --- 그림 2: 순위 변화
    fig, ax = plt.subplots(figsize=(8.5, 5))
    cc = plt.cm.tab10(np.arange(len(CONDS)))
    for j, c in enumerate(CONDS):
        ax.plot(range(len(V)), [ranks[v][c] for v in V], "-o", color=cc[j], lw=2, ms=7, label=LABEL[c])
    ax.set_xticks(range(len(V)))
    ax.set_xticklabels([f"변형 {v}" for v in V])
    ax.set_yticks(range(1, len(CONDS) + 1))
    ax.invert_yaxis()
    ax.set_ylabel("순위 (1 = P가 가장 높음)")
    ax.set_title("전처리 변형에 따른 텍스트 조건 순위\n" + ", ".join(f"τ({k})={t:.2f}" for k, t in list(taus.items())[:3]) + " …", fontsize=11)
    ax.legend(fontsize=8.5, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(Path(a.fig_dir) / "e1_prep_ranks.png", dpi=200)
    plt.close(fig)

    # --- 그림 3: 핵심 대비 forest
    fig, axs = plt.subplots(1, len(CON), figsize=(13, 3.6), sharey=True)
    for ax, (k, (_, _, nm)) in zip(axs, CON.items()):
        for i, v in enumerate(V):
            c = cons[k][v]
            ax.errorbar(c["diff"], i, xerr=[[c["diff"] - c["ci"][0]], [c["ci"][1] - c["diff"]]], fmt="o", color=COL[v], capsize=3, ms=6)
        ax.axvline(0, color="#999", lw=0.8)
        ax.set_title(nm + ("\n(강건)" if robust[k] else "\n(전처리 의존)"), fontsize=10)
        ax.set_yticks(range(len(V)))
        ax.set_yticklabels([f"변형 {v}" for v in V])
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("P 차이 (95% 구간)", fontsize=9)
    fig.suptitle("핵심 대비의 전처리 변형별 크기와 부호", fontsize=11)
    fig.tight_layout()
    fig.savefig(Path(a.fig_dir) / "e1_prep_contrasts.png", dpi=200)
    plt.close(fig)
    print("\n".join(L))


if __name__ == "__main__":
    main()
