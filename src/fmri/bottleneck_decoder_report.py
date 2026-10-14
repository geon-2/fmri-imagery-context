"""병목 가능성 점검 2 정리(기준 docs/bottleneck_decoder_feasibility_criteria.md). 사용: python3 src/fmri/bottleneck_decoder_report.py"""
import json
from pathlib import Path

import numpy as np

R = json.load(open("data/results/bottleneck_decoder_pilot/bottleneck_decoder_pilot.json", encoding="utf-8"))
SUBJ = list(R)
KS = (16, 32, 64, 128)
KMAIN = 64
rng = np.random.default_rng(0)
units = [(s, "A", "clip_img") for s in SUBJ] + [(s, "C", "ctx_P1") for s in SUBJ]


def acc_from_C(C):
    C = np.array(C)
    return ((C.argmax(1) == np.arange(len(C))).mean() + (C.argmax(0) == np.arange(len(C))).mean()) / 2


def pooled(kind, nperm=1000):
    Cs = [np.array(R[s][t]["full"][f"{st}_{kind}_C"]) for s, st, t in units]
    obs = np.mean([acc_from_C(C) for C in Cs])
    null = [np.mean([acc_from_C(C[:, rng.permutation(6)]) for C in Cs]) for _ in range(nperm)]
    return float(obs), float((np.sum(np.array(null) >= obs) + 1) / (nperm + 1))


def rand_mean(r, k, key):
    return float(np.mean([r[f"rand{i}_{k}"][key] for i in range(5)]))


L = ["# 병목 encoder→z→decoder 가능성 점검 2 결과 (decoder 포함, 학습은 perception만; 피험자 1·2·5·7, 세트 A·C)", "", "기준 `docs/bottleneck_decoder_feasibility_criteria.md`. 세트 B는 열지 않았다. 8단위(우연 1/6 = 0.167). 탐색 단계, 가능성 점검이며 복원 성능 비교가 아님.", ""]
vo, vp = pooled("vis")
io, ip = pooled("img")
L += ["## 전제", "", f"- 양성 대조(viewing 복원 식별, 풀 릿지): 평균 {vo:.3f}, 순열 p = {vp:.4f} → {'충족' if vp < 0.05 else '미충족'}", f"- imagery 복원 식별(풀 릿지): 평균 {io:.3f}, 순열 p = {ip:.4f} → {'충족' if ip < 0.05 else '미충족'}", ""]
L += ["## perception 유지 (보류 20%, top-1 / top-5, 피험자 4명 평균)", "", "| 타깃 | k | 풀 릿지 | 병목 encoder | 복셀 PCA | 무작위 | 병목 − 풀 릿지 (top-1) | 유지(≥ −0.02) |", "|---|---|---|---|---|---|---|---|"]
maint = {}
for t in ("clip_img", "ctx_P1", "orig_P1"):
    full1 = np.mean([R[s][t]["full"]["perc_top1"] for s in SUBJ])
    full5 = np.mean([R[s][t]["full"]["perc_top5"] for s in SUBJ])
    for k in KS:
        e1 = np.mean([R[s][t][f"enc_{k}"]["perc_top1"] for s in SUBJ])
        e5 = np.mean([R[s][t][f"enc_{k}"]["perc_top5"] for s in SUBJ])
        p1 = np.mean([R[s][t][f"pca_{k}"]["perc_top1"] for s in SUBJ])
        r1 = np.mean([rand_mean(R[s][t], k, "perc_top1") for s in SUBJ])
        r5 = np.mean([rand_mean(R[s][t], k, "perc_top5") for s in SUBJ])
        ok = e1 - full1 >= -0.02
        maint[(t, k)] = ok
        L.append(f"| {t} | {k} | {full1:.3f} / {full5:.3f} | {e1:.3f} / {e5:.3f} | {p1:.3f} | {r1:.3f} / {r5:.3f} | {e1 - full1:+.3f} | {'예' if ok else '아니오'} |")
L += ["", "## imagery 복원 식별 (후보 6개, 단위별; 세트 A는 CLIP 이미지 decoder, 세트 C는 ctx_P1 decoder)", "", "| 단위 | 풀 릿지 imagery | 병목 k=64 | 무작위 k=64 | PCA k=64 | 풀 릿지 viewing | 병목 viewing | 무작위 viewing |", "|---|---|---|---|---|---|---|---|"]
dz = []
for s, st, t in units:
    r = R[s][t]
    e, rm, p = r[f"enc_{KMAIN}"], {k: rand_mean(r, KMAIN, k) for k in (f"{st}_img", f"{st}_vis")}, r[f"pca_{KMAIN}"]
    L.append(f"| {s}-{st} | {r['full'][f'{st}_img']:.2f} | {e[f'{st}_img']:.2f} | {rm[f'{st}_img']:.2f} | {p[f'{st}_img']:.2f} | {r['full'][f'{st}_vis']:.2f} | {e[f'{st}_vis']:.2f} | {rm[f'{st}_vis']:.2f} |")
    dz.append(e[f"{st}_img"] - rm[f"{st}_img"])
L += ["", "## k별 imagery 복원 식별 (8단위 평균)", "", "| k | 병목 encoder | 복셀 PCA | 무작위 | 풀 릿지 | (병목 − 무작위) | 0 이상인 단위 수 |", "|---|---|---|---|---|---|---|"]
for k in KS:
    ee = [R[s][t][f"enc_{k}"][f"{st}_img"] for s, st, t in units]
    pp = [R[s][t][f"pca_{k}"][f"{st}_img"] for s, st, t in units]
    rr = [rand_mean(R[s][t], k, f"{st}_img") for s, st, t in units]
    ff = [R[s][t]["full"][f"{st}_img"] for s, st, t in units]
    d = np.array(ee) - np.array(rr)
    L.append(f"| {k} | {np.mean(ee):.3f} | {np.mean(pp):.3f} | {np.mean(rr):.3f} | {np.mean(ff):.3f} | {d.mean():+.3f} | {int((d >= 0).sum())}/8 |")
L += ["", "## 세트 C: orig_P1 decoder (보조)", "", "| 단위 | 풀 릿지 | 병목 k=64 | 무작위 k=64 |", "|---|---|---|---|"]
for s in SUBJ:
    r = R[s]["orig_P1"]
    L.append(f"| {s}-C | {r['full']['C_img']:.2f} | {r['enc_64']['C_img']:.2f} | {rand_mean(r, 64, 'C_img'):.2f} |")
prem = vp < 0.05 and ip < 0.05
dz = np.array(dz)
pm = all(maint[(t, KMAIN)] for t in ("clip_img", "ctx_P1"))
if not prem:
    verdict = "검증 불가(전제 미충족)"
elif pm and dz.mean() > 0 and (dz >= 0).sum() >= 6:
    verdict = "가능성 있음"
elif dz.mean() <= 0 or (dz >= 0).sum() < 4:
    verdict = "가능성 낮음"
else:
    verdict = "판단 불가"
L += ["", "## 판정 (기준 그대로)", "", f"- 전제 충족: {prem}", f"- perception 유지(k=64, CLIP 이미지·ctx_P1): {pm}", f"- (병목 − 무작위 k=64) imagery 식별 차이 평균 {dz.mean():+.3f}, 0 이상인 단위 {int((dz >= 0).sum())}/8", f"- **판정: {verdict}**"]
Path("results/tables/bottleneck_decoder_pilot.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
