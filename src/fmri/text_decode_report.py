"""1단계 결과 정리: 표, 판정(기준 docs/stage1_text_repr_criteria.md), 그림 3장(PNG).
사용: python3 src/fmri/text_decode_report.py
"""
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
res = json.load(open(R / "text_decode_compare/text_decode_compare.json", encoding="utf-8"))
t10 = json.load(open(R / "text_decode_compare/text_decode_t10.json", encoding="utf-8"))
FIG = Path("docs/report_figures")
LABEL = {"orig_P1": "원본 캡션\nP1", "orig_V2": "원본 캡션\nV2", "obj_P1": "사물 이름\nP1", "obj_V2": "사물 이름\nV2", "ctx_P1": "맥락 라벨\nP1",
         "ctx_V2": "맥락 라벨\nV2", "obj_P1+ctx_P1": "사물+맥락\nP1", "ctx_T10": "맥락 라벨\nT10", "ctx_P1_shuf": "셔플 대조"}
COL = {"CLIP": "#2B6CB0", "MPNet": "#D9822B"}
REPS = ["orig_P1", "orig_V2", "obj_P1", "obj_V2", "ctx_P1", "ctx_V2", "obj_P1+ctx_P1", "ctx_P1_shuf"]
NAMEK = {"K1_ctx_minus_obj": "K1 맥락P1 − 사물P1", "K2_ctx_minus_orig": "K2 맥락P1 − 원본P1", "K3_ctxV2_minus_ctxP1": "K3 맥락V2 − 맥락P1",
         "K4_objctx_minus_orig": "K4 (사물+맥락) − 원본P1", "K5_T10_minus_ctxP1": "K5 T10 − 맥락P1", "K6_T10_minus_ctxV2": "K6 T10 − 맥락V2"}

# T10 대비는 별도 실행에서 가져온다(같은 폴드, ctx_P1 점수가 동일함을 확인)
same = abs(t10["reps"]["CLIP"]["ctx_P1"]["top1"] - res["reps"]["CLIP"]["ctx_P1"]["top1"]) < 1e-9
res["reps"]["CLIP"]["ctx_T10"] = t10["reps"]["CLIP"]["ctx_T10"]
for k in ("K5_T10_minus_ctxP1", "K6_T10_minus_ctxV2"):
    res["contrasts"]["CLIP"][k] = t10["contrasts"]["CLIP"][k]

# encoding(보조) 순위 비교
enc = {}
for e in ("CLIP", "MPNet"):
    p = R / f"prep_sensitivity/prep_sensitivity_{e}.json"
    if p.exists():
        enc[e] = json.load(open(p, encoding="utf-8"))["variants"]["a"]["P"]
common = ["orig_P1", "orig_V2", "obj_P1", "ctx_P1", "ctx_V2", "obj_P1+ctx_P1"]
tau = {e: float(kendalltau([enc[e][c] for c in common], [res["reps"][e][c]["top1"] for c in common])[0]) for e in enc}

L = ["# 1단계 텍스트 표현 비교 결과 (decoding 주 지표)", "", f"피험자 1, 전처리 a, 학습용 이미지 {res['n_images']}장 5겹, 복셀 {res['n_voxels']:,}개(R2∪R3). 기준: `docs/stage1_text_repr_criteria.md`. 탐색 단계.", "",
     "## 표현별 decoding top-1 (95% 구간), top-5", "", "| 표현 | CLIP top-1 | CLIP top-5 | MPNet top-1 | MPNet top-5 |", "|---|---|---|---|---|"]
for r in REPS + ["ctx_T10"]:
    row = []
    for e in ("CLIP", "MPNet"):
        v = res["reps"][e].get(r)
        row += [f"{v['top1']:.4f} [{v['top1_ci'][0]:.4f}, {v['top1_ci'][1]:.4f}]", f"{v['top5']:.4f}"] if v else ["-", "-"]
    L.append(f"| {r} | " + " | ".join(row) + " |")
L += ["", f"우연 수준 약 0.001. 셔플 대조 CLIP {res['reps']['CLIP']['ctx_P1_shuf']['top1']:.4f}, MPNet {res['reps']['MPNet']['ctx_P1_shuf']['top1']:.4f} (기준 ≤ 0.005 {'충족' if max(res['reps'][e]['ctx_P1_shuf']['top1'] for e in ('CLIP', 'MPNet')) <= 0.005 else '미충족'}).",
      f"T10 실행의 ctx_P1 점수가 본 실행과 같은지: {'같음' if same else '다름(확인 필요)'}.", "", "## 핵심 대비 (top-1 차이, 판정: 구간이 0 제외 + |차이| ≥ 0.02)", "",
      "| 대비 | CLIP 차이 [95% 구간] | CLIP 판정 | MPNet 차이 [95% 구간] | MPNet 판정 |", "|---|---|---|---|---|"]
for k, nm in NAMEK.items():
    row = []
    for e in ("CLIP", "MPNet"):
        c = res["contrasts"][e].get(k)
        row += [f"{c['diff']:+.4f} [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]", c["verdict"]] if c else ["-", "-"]
    L.append(f"| {nm} | " + " | ".join(row) + " |")
er = res["encoder_rule"]
L += ["", "## 인코더 선정 규칙 (MPNet − CLIP, 핵심 표현 3개)", ""]
for r in ("orig_P1", "ctx_P1", "obj_P1+ctx_P1"):
    L.append(f"- {r}: {er[r]['mpnet_minus_clip']:+.4f} [{er[r]['ci'][0]:+.4f}, {er[r]['ci'][1]:+.4f}] → {'MPNet이 분명히 높음' if er[r]['mpnet_clearly_higher'] else '아님'}")
L += [f"- **판정: {er['verdict']}**", "", "## encoding(보조, E1 변형 a와 같은 방식)과 decoding 순위의 Kendall τ (표현 6개)", ""]
L += [f"- {e}: τ = {t:.2f}" for e, t in tau.items()]
Path("results/tables/text_decode_compare.md").write_text("\n".join(L), encoding="utf-8")
json.dump({"tau_enc_dec": tau, "same_ctxP1_in_t10_run": same, "encoder_rule": er}, open(R / "text_decode_compare/text_decode_report.json", "w"), ensure_ascii=False, indent=1)

# --- 그림 1: 표현별 top-1
fig, ax = plt.subplots(figsize=(11, 5))
w = 0.38
names = REPS[:-1] + ["ctx_T10", "ctx_P1_shuf"]
for i, e in enumerate(("CLIP", "MPNet")):
    xs = np.arange(len(names)) + (i - 0.5) * w
    ys = [res["reps"][e][r]["top1"] if r in res["reps"][e] else np.nan for r in names]
    lo = [res["reps"][e][r]["top1"] - res["reps"][e][r]["top1_ci"][0] if r in res["reps"][e] else 0 for r in names]
    hi = [res["reps"][e][r]["top1_ci"][1] - res["reps"][e][r]["top1"] if r in res["reps"][e] else 0 for r in names]
    ax.bar(xs, ys, w * 0.92, color=COL[e], label=e, yerr=[lo, hi], capsize=2, error_kw={"lw": 0.8})
ax.axhline(0.001, color="#999", ls=":", lw=1)
ax.text(len(names) - 0.6, 0.004, "우연 0.001", fontsize=8, color="#666", ha="right")
ax.set_xticks(range(len(names)))
ax.set_xticklabels([LABEL[r] for r in names], fontsize=9)
ax.set_ylabel("decoding top-1 (후보 약 1,000장)")
ax.set_title("텍스트 표현별 decoding 정확도 (피험자 1, 5겹 교차검증, 오차 = 95% 부트스트랩; T10은 CLIP만)")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "s1_decoding_top1.png", dpi=200)
plt.close(fig)

# --- 그림 2: 핵심 대비
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, e in zip(axs, ("CLIP", "MPNet")):
    ks = [k for k in NAMEK if k in res["contrasts"][e]]
    for j, k in enumerate(ks):
        c = res["contrasts"][e][k]
        ax.errorbar(c["diff"], j, xerr=[[c["diff"] - c["ci"][0]], [c["ci"][1] - c["diff"]]], fmt="o", color=COL[e], capsize=3, ms=6)
    ax.axvline(0, color="#999", lw=0.8)
    for s in (-0.02, 0.02):
        ax.axvline(s, color="#BBB", lw=0.8, ls=":")
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels([NAMEK[k] for k in ks], fontsize=9)
    ax.invert_yaxis()
    ax.set_title(e)
    ax.set_xlabel("top-1 차이 (점선 = 기준 ±0.02, 막대 = 95% 구간)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("핵심 대비의 decoding top-1 차이", fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "s1_contrasts.png", dpi=200)
plt.close(fig)

# --- 그림 3: encoding 대 decoding 순위
if enc:
    fig, axs = plt.subplots(1, len(enc), figsize=(5.6 * len(enc), 4.6))
    axs = np.atleast_1d(axs)
    for ax, e in zip(axs, enc):
        x = [enc[e][c] for c in common]
        y = [res["reps"][e][c]["top1"] for c in common]
        ax.scatter(x, y, color=COL[e], s=50)
        for c, xi, yi in zip(common, x, y):
            ax.annotate(LABEL[c].replace("\n", " "), (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_xlabel("encoding P")
        ax.set_ylabel("decoding top-1")
        ax.set_title(f"{e}: encoding과 decoding (Kendall τ = {tau[e]:.2f})")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "s1_enc_vs_dec.png", dpi=200)
    plt.close(fig)
print("\n".join(L))
