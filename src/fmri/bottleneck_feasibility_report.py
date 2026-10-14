"""병목 가능성 점검 정리(기준 docs/bottleneck_feasibility_criteria.md). 사용: python3 src/fmri/bottleneck_feasibility_report.py"""
import json
from pathlib import Path

import numpy as np

import imagery_roi as ir

D = Path("data/results/bottleneck_feasibility")
R = json.load(open(D / "bottleneck_feasibility.json", encoding="utf-8"))
SUBJ = list(R)
units = [(s, st) for s in SUBJ for st in ("A", "C")]
KS = (16, 32, 64)
KMAIN = 32
rng = np.random.default_rng(0)


def pooled_perm(kind, nperm=1000):
    obs = np.mean([R[s][st]["full"][kind] for s, st in units])
    P = {}
    for s in SUBJ:
        z = np.load(D / f"patterns_{s}.npz")
        for st in ("A", "C"):
            P[(s, st)] = (z[f"{st}_img1"], z[f"{st}_img2"]) if kind == "img" else (z[f"{st}_vis1"], z[f"{st}_vis2"])
    null = []
    for _ in range(nperm):
        null.append(np.mean([ir.ident_accuracy(a, b[rng.permutation(6)])[0] for a, b in P.values()]))
    return float(obs), float((np.sum(np.array(null) >= obs) + 1) / (nperm + 1))


L = ["# 병목 encoder–decoder 가능성 점검 결과 (피험자 1·2·5·7, 세트 A·C, 세트 B는 열지 않음)", "", "기준 `docs/bottleneck_feasibility_criteria.md`. 8단위(피험자 4 × 세트 2), 후보 6개(우연 1/6 = 0.167). 탐색 단계, 복원 성능 비교가 아님.", ""]
vo, vp = pooled_perm("vis")
io, ip = pooled_perm("img")
L += ["## 전제", "", f"- 양성 대조(viewing 식별, 전체 복셀): 평균 {vo:.3f}, 순열 p = {vp:.4f} → {'충족' if vp < 0.05 else '미충족'}", f"- imagery 신호(imagery 식별, 전체 복셀): 평균 {io:.3f}, 순열 p = {ip:.4f} → {'충족' if ip < 0.05 else '미충족'}", "",
      "## 단위별 식별 (전체 복셀 / encoder ctx_P1 z(k=32) / 무작위 k=32 / PCA k=32)", "", "| 단위 | imagery 전체 | imagery z | imagery 무작위 | imagery PCA | viewing 전체 | viewing z | viewing 무작위 | viewing PCA |", "|---|---|---|---|---|---|---|---|---|"]
dz, al = [], []
for s, st in units:
    u = R[s][st]
    z, r, p = u[f"enc_ctx_P1_{KMAIN}"], u[f"rand_{KMAIN}"], u[f"pca_{KMAIN}"]
    L.append(f"| {s}-{st} | {u['full']['img']:.2f} | {z['img']:.2f} | {r['img']:.2f} | {p['img']:.2f} | {u['full']['vis']:.2f} | {z['vis']:.2f} | {r['vis']:.2f} | {p['vis']:.2f} |")
    dz.append(z["img"] - r["img"])
    al.append(z["ratio_img"] > r["ratio_img_p95"])
L += ["", "## 정렬 비율 (imagery 분산 중 k=32 부분공간이 설명하는 비율)", "", "| 단위 | encoder ctx_P1 | encoder orig_P1 | encoder CLIP 이미지 | PCA | 무작위 평균 | 무작위 95% | perception 보류 이미지(encoder ctx_P1) | 기준 통과(encoder ctx_P1 > 무작위 95%) |", "|---|---|---|---|---|---|---|---|---|"]
for s, st in units:
    u = R[s][st]
    L.append(f"| {s}-{st} | {u[f'enc_ctx_P1_{KMAIN}']['ratio_img']:.3f} | {u[f'enc_orig_P1_{KMAIN}']['ratio_img']:.3f} | {u[f'enc_clip_img_{KMAIN}']['ratio_img']:.3f} | {u[f'pca_{KMAIN}']['ratio_img']:.3f} | {u[f'rand_{KMAIN}']['ratio_img']:.3f} | {u[f'rand_{KMAIN}']['ratio_img_p95']:.3f} | {u[f'enc_ctx_P1_{KMAIN}']['ratio_perc']:.3f} | {'예' if u[f'enc_ctx_P1_{KMAIN}']['ratio_img'] > u[f'rand_{KMAIN}']['ratio_img_p95'] else '아니오'} |")
L += ["", "## k별 요약 (8단위 평균)", "", "| k | imagery 식별 z (ctx) | z (orig) | z (CLIP 이미지) | 무작위 | PCA | 전체 복셀 | 정렬 비율 z(ctx) | 무작위 |", "|---|---|---|---|---|---|---|---|---|"]
for k in KS:
    g = lambda key, f: np.mean([R[s][st][key][f] for s, st in units])
    L.append(f"| {k} | {g(f'enc_ctx_P1_{k}', 'img'):.3f} | {g(f'enc_orig_P1_{k}', 'img'):.3f} | {g(f'enc_clip_img_{k}', 'img'):.3f} | {g(f'rand_{k}', 'img'):.3f} | {g(f'pca_{k}', 'img'):.3f} | {np.mean([R[s][st]['full']['img'] for s, st in units]):.3f} | {g(f'enc_ctx_P1_{k}', 'ratio_img'):.3f} | {g(f'rand_{k}', 'ratio_img'):.3f} |")
prem = vp < 0.05 and ip < 0.05
n_al, n_dz, mean_dz = int(sum(al)), int(sum(d >= 0 for d in dz)), float(np.mean(dz))
if not prem:
    verdict = "검증 불가(전제 미충족)"
elif n_al >= 6 and mean_dz > 0 and n_dz >= 6:
    verdict = "가능성 있음"
elif n_al <= 4 or mean_dz <= 0:
    verdict = "가능성 낮음"
else:
    verdict = "판단 불가"
L += ["", "## 판정 (기준 그대로)", "", f"- 전제 충족: {prem}", f"- 정렬 비율이 무작위 95%를 넘는 단위: {n_al}/8", f"- (z − 무작위) imagery 식별 차이 평균: {mean_dz:+.3f}, 0 이상인 단위 {n_dz}/8", f"- **판정: {verdict}**"]
Path("results/tables/bottleneck_feasibility.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
