"""E2 결과 정리: 표, 판정(기준은 docs/prep_axes_criteria.md), 그림 4장(PNG).
사용: python3 src/fmri/axes_dir_report.py [--in-dir data/results/axes_dir_compare] [--fig-dir docs/report_figures]
"""
import argparse
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

METHODS = ("RRR", "DDR", "PLS", "CCA", "PCA", "RAND")
NAME = {"RRR": "RRR (인코딩 방향)", "DDR": "DDR (decoding 방향)", "PLS": "PLS", "CCA": "CCA", "PCA": "PCA(임베딩)", "RAND": "무작위"}
COL = {"RRR": "#2B6CB0", "DDR": "#D9822B", "PLS": "#7B4FD6", "CCA": "#0F8F7F", "PCA": "#777777", "RAND": "#BBBBBB"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/results/axes_dir_compare")
    ap.add_argument("--stem", default="axes_dir_compare_MPNet_ctx_P1")
    ap.add_argument("--fig-dir", default="docs/report_figures")
    ap.add_argument("--table", default="results/tables/axes_dir_compare.md")
    a = ap.parse_args()
    res = json.load(open(Path(a.in_dir) / f"{a.stem}.json", encoding="utf-8"))
    Path(a.fig_dir).mkdir(parents=True, exist_ok=True)
    ks = res["ks"]
    enc = {m: {x["k"]: x for x in res["encoding"][m]} for m in METHODS}
    dec = {m: {x["k"]: x for x in res["decoding"][m]} for m in METHODS}
    full = res["decoding"]["full"][0]

    # --- 판정
    verdict = {}
    for kk in ("k5", "k12", "k20"):
        c = res["contrasts"].get(kk)
        if not c:
            continue
        v = {}
        for pair in ("DDR-RRR", "CCA-RRR", "PLS-RRR", "PCA-RRR"):
            e, t1 = c[pair]["encoding_P"], c[pair]["decoding_top1"]
            ev = "차이 있음" if (e["ci"][0] > 0 or e["ci"][1] < 0) and abs(e["diff"]) >= 0.01 else "동등"
            tv = "차이 있음" if (t1["ci"][0] > 0 or t1["ci"][1] < 0) and abs(t1["diff"]) >= 0.02 else "동등"
            v[pair] = {"encoding_P": ev, "decoding_top1": tv, "enc_diff": e["diff"], "dec_diff": t1["diff"]}
        for m in ("RRR", "DDR", "CCA", "PLS", "PCA"):
            e, t1 = c[f"{m}-RAND"]["encoding_P"], c[f"{m}-RAND"]["decoding_top1"]
            v[f"{m}-RAND"] = {"encoding_above_random": bool(e["ci"][0] > 0), "decoding_above_random": bool(t1["ci"][0] > 0)}
        verdict[kk] = v
    json.dump({"verdict": verdict}, open(Path(a.in_dir) / "axes_dir_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # --- 표
    L = ["# E2 축 방법 비교 결과", "", f"피험자 1, {res['model']} {res['rep']}, 학습용 이미지 {res['n_images']}장, 복셀 {res['n_voxels']:,}개(전처리 a), 5겹 교차검증(탐색). 기준: `docs/prep_axes_criteria.md`.", "",
         "## encoding P (축 k개 위에서 복셀 재적합)", "", "| 방법 | " + " | ".join(f"k={k}" for k in ks) + " |", "|---|" + "---|" * len(ks)]
    for m in METHODS:
        L.append(f"| {NAME[m]} | " + " | ".join(f"{enc[m][k]['P']:.3f}" for k in ks) + " |")
    L += ["", "## decoding top-1 (보류 폴드 약 1,000장 중 1장 찾기; k=1은 퇴화라 제외)", "", "| 방법 | " + " | ".join(f"k={k}" for k in ks if k > 1) + " |", "|---|" + "---|" * (len(ks) - 1)]
    for m in METHODS:
        L.append(f"| {NAME[m]} | " + " | ".join(f"{dec[m][k]['top1']:.3f}" for k in ks if k > 1) + " |")
    L += ["", f"전체 768차원 복원 top-1 {full['top1']:.3f}, top-5 {full['top5']:.3f}.", "", "## 판정 (DDR 등 대 RRR, 같은 k)", "",
          "| k | 쌍 | encoding P 차이 | 판정 | decoding top-1 차이 | 판정 |", "|---|---|---|---|---|---|"]
    for kk, v in verdict.items():
        for pair in ("DDR-RRR", "CCA-RRR", "PLS-RRR", "PCA-RRR"):
            L.append(f"| {kk[1:]} | {pair} | {v[pair]['enc_diff']:+.4f} | {v[pair]['encoding_P']} | {v[pair]['dec_diff']:+.4f} | {v[pair]['decoding_top1']} |")
    L += ["", "## 무작위 축보다 높은가 (95% 구간 하한 > 0)", ""]
    for kk, v in verdict.items():
        L.append(f"- k={kk[1:]}: " + ", ".join(f"{m} (enc {'O' if v[f'{m}-RAND']['encoding_above_random'] else 'X'}, dec {'O' if v[f'{m}-RAND']['decoding_above_random'] else 'X'})" for m in ("RRR", "DDR", "CCA", "PLS", "PCA")))
    L += ["", "## 축 공간의 겹침 (평균 cos², 무작위 기대값 k/d)", ""]
    for key, val in res["overlap"].items():
        k = key.split("@")[1]
        L.append(f"- {key}: {val:.3f} (무작위 {res['overlap_random_expect'][f'k{k}']:.3f})")
    Path(a.table).parent.mkdir(parents=True, exist_ok=True)
    Path(a.table).write_text("\n".join(L), encoding="utf-8")

    # --- 그림 1: encoding k 곡선
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for m in METHODS:
        y = np.array([enc[m][k]["P"] for k in ks])
        ax.plot(ks, y, "-o", color=COL[m], lw=2 if m in ("RRR", "DDR") else 1.3, ms=5, label=NAME[m])
        if enc[m][ks[0]]["ci"] is not None and m in ("RRR", "DDR"):
            ax.fill_between(ks, [enc[m][k]["ci"][0] for k in ks], [enc[m][k]["ci"][1] for k in ks], color=COL[m], alpha=0.15)
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels(ks)
    ax.set_xlabel("축 개수 k")
    ax.set_ylabel("encoding P (R2·R3 복셀 r 중앙값의 평균)")
    ax.set_title("축을 찾는 방법별 encoding 점수 (피험자 1, 맥락 라벨 P1, MPNet)")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(Path(a.fig_dir) / "e2_encoding_k.png", dpi=200)
    plt.close(fig)

    # --- 그림 2: decoding top-1, top-5
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for ax, key, ttl in zip(axs, ("top1", "top5"), ("top-1", "top-5")):
        for m in METHODS:
            ax.plot([k for k in ks if k > 1], [dec[m][k][key] for k in ks if k > 1], "-o", color=COL[m], lw=2 if m in ("RRR", "DDR") else 1.3, ms=5, label=NAME[m])
        ax.axhline(full[key], color="#333", ls="--", lw=1, label="전체 768차원")
        ax.set_xscale("log")
        ax.set_xticks([k for k in ks if k > 1])
        ax.set_xticklabels([k for k in ks if k > 1])
        ax.set_xlabel("축 개수 k")
        ax.set_ylabel(f"decoding {ttl} (후보 약 1,000장)")
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle("축 k개로 복원한 decoding 정확도 (보류 폴드, 피험자 1)")
    fig.tight_layout()
    fig.savefig(Path(a.fig_dir) / "e2_decoding_k.png", dpi=200)
    plt.close(fig)

    # --- 그림 3: 축 공간 겹침 히트맵 (k=12)
    k = 12
    M = np.eye(len(METHODS[:-1]))
    names = METHODS[:-1]
    for i, x in enumerate(names):
        for j, y in enumerate(names):
            if i != j:
                key = f"{x}-{y}@{k}" if f"{x}-{y}@{k}" in res["overlap"] else f"{y}-{x}@{k}"
                M[i, j] = res["overlap"][key]
    fig, ax = plt.subplots(figsize=(5.8, 5))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="white" if M[i, j] > 0.5 else "black", fontsize=10)
    ax.set_title(f"축 공간 겹침 (k={k}, 평균 cos²)\n무작위 기대값 {res['overlap_random_expect'][f'k{k}']:.3f}", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(Path(a.fig_dir) / "e2_overlap.png", dpi=200)
    plt.close(fig)

    # --- 그림 4: RRR 대비 차이 (forest)
    pairs = ("DDR-RRR", "CCA-RRR", "PLS-RRR", "PCA-RRR")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (typ, ttl, thr) in zip(axs, (("encoding_P", "encoding P 차이", 0.01), ("decoding_top1", "decoding top-1 차이", 0.02))):
        yy = 0
        ticks, labels = [], []
        for kk in ("k5", "k12", "k20"):
            c = res["contrasts"].get(kk)
            if not c:
                continue
            for pair in pairs:
                e = c[pair][typ]
                ax.errorbar(e["diff"], yy, xerr=[[e["diff"] - e["ci"][0]], [e["ci"][1] - e["diff"]]], fmt="o", color=COL[pair.split("-")[0]], capsize=3, ms=6)
                ticks.append(yy)
                labels.append(f"k={kk[1:]} {pair}")
                yy += 1
            yy += 0.6
        ax.axvline(0, color="#999", lw=0.8)
        for s in (-thr, thr):
            ax.axvline(s, color="#BBB", lw=0.8, ls=":")
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel(ttl + " (점선 = 기준 크기, 막대 = 95% 구간)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("RRR(인코딩 방향) 대비 다른 축 방법의 차이", fontsize=11)
    fig.tight_layout()
    fig.savefig(Path(a.fig_dir) / "e2_contrasts.png", dpi=200)
    plt.close(fig)
    print("\n".join(L))


if __name__ == "__main__":
    main()
