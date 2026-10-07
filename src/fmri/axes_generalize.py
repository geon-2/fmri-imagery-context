"""RQ1 보강 3종: 축이 새 이미지·새 범주·같은 범주 안에서도 통하는가. 기준은 계산 전에 정했다(2026-10-08, Notion 「일반화 확인 3종」).
공통: 맥락 임베딩(MPNet ctx_P1, 라벨 있는 전체 이미지로 공통 표준화), 축 k=12, 피험자 4명(1·2·5·7), 방향·decoder는 학습 이미지에서만 적합.
주요 범주 = 이미지에서 마스크 면적이 가장 큰 COCO 범주(coco_primary.py).

[A] 피험자 간 공유를 shared1000에서 확인
  피험자 s의 학습 이미지로 구한 RRR 방향(12개)을 대상 피험자 t의 shared1000 예측에 그대로 쓴다(t의 학습 이미지에서 12차원 OLS만 적합).
  점수 P = shared1000에서 R2·R3 복셀 median r의 평균. 비교: t 자신의 RRR, t의 PCA, 무작위.
  기준: 출처 두 명의 평균 P가 (다른 사람 − 자기) ≥ −0.01 이고 (다른 사람 − PCA)의 95% 구간 하한 > 0.01 인 대상 피험자가 4명 중 3명 이상.
[B] 같은 주요 범주 안에서 순위(shared1000)
  복셀 → 12축 decoder(전체 학습 이미지)로 shared1000 이미지의 축 값을 읽고, 이미지마다 같은 주요 범주의 shared1000 이미지(10장 이상인 범주만)를
  후보로 정답의 백분위 순위(후보 중 정답보다 유사도가 낮은 비율, 우연 0.5)와 1등 비율(우연 1/후보 수)을 낸다.
  기준: B1 RRR의 평균 백분위 구간 하한 > 0.5 가 4명 모두. B2 (RRR − PCA) 평균 백분위 ≥ 0.02 이고 구간 하한 > 0 인 피험자 2명 이상.
[C] 주요 범주를 통째로 뺀 학습(학습 이미지, 범주 단위 5분할)
  범주를 크기 순으로 폴드에 탐욕적으로 배정해 5분할. 폴드마다 나머지 범주의 이미지로 방향·decoder를 적합하고 뺀 범주의 이미지(후보 = 같은 폴드 이미지 전체)로 평가.
  지표: 정답의 백분위 순위(우연 0.5), 1등·5등 비율(우연 1/후보 수). 
  기준: C1 RRR 백분위의 구간 하한 > 0.5 가 4명 모두. C2 (RRR − PCA) ≥ 0.02 이고 구간 하한 > 0 인 피험자 2명 이상.
  참고(기준 없음): 무작위 5분할 탐색 결과(이전 axes_decode)의 1등 비율과 비교.
[D] 학습 데이터 양 곡선(C와 같은 범주 단위 5분할, 학습 이미지를 25·50·75·100%만 사용, 100%는 C 결과)
  기준: 「데이터에 민감」 = 4명 중 3명 이상에서 (RRR 1등 비율 100% − 50%) ≥ 0.01. 아니면 「거의 포화」. 곡선 모양(25→100%)과 RRR−PCA 백분위 차이도 함께 보고(기준 없음).
부트스트랩 1,000회는 이미지 단위. 모든 판정은 MPNet 기준(CLIP은 하지 않음).
사용: PYTHONPATH=src/labeling:src/fmri python3 src/fmri/axes_generalize.py [--parts A,B,C]   시험: --selftest
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import axes_k_curve as ak
import axes_decode as dc
import embedding_compare as ec
from axes_shared import fit_dirs

K = 12
SUBJ = ["subj01", "subj02", "subj05", "subj07"]


def pct_rank(Sm, groups, min_n=2):
    """Sm[i,j]: i번 예측 vs j번 실제 좌표 유사도. 같은 groups 값끼리를 후보로 정답(i)의 백분위, 1등, 5등 여부, 후보 수."""
    n = len(Sm)
    pct = np.full(n, np.nan)
    t1 = np.full(n, np.nan)
    t5 = np.full(n, np.nan)
    nc = np.zeros(n)
    g = np.asarray(groups)
    for c in np.unique(g):
        idx = np.flatnonzero(g == c)
        if len(idx) < min_n:
            continue
        sub = Sm[np.ix_(idx, idx)]
        d = np.diag(sub)
        higher = (sub > d[:, None]).sum(1)
        pct[idx] = 1 - higher / (len(idx) - 1)
        t1[idx] = higher == 0
        t5[idx] = higher < 5
        nc[idx] = len(idx)
    return pct, t1, t5, nc


def boot_ci(v, B=1000, seed=0):
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    bs = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)])
    return float(v.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def diff_ci(a, b, B=1000, seed=0):
    ok = ~np.isnan(a) & ~np.isnan(b)
    d = (a - b)[ok]
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    return float(d.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def method_sims(Zh, Zt, cols):
    sims = {}
    for m in ("RRR", "PCA", "RAND"):
        sl = cols[m]
        sub = slice(sl.start, sl.start + K)
        sims[m] = dc.sim_matrix(Zh[:, sub], Zt[:, sub])
    sims["full"] = dc.sim_matrix(Zh[:, cols["full"]], Zt[:, cols["full"]])
    return sims


def eval_groups(sims, groups, B, min_n):
    res, per = {}, {}
    for m, Sm in sims.items():
        pct, t1, t5, nc = pct_rank(Sm, groups, min_n)
        per[m] = pct
        mean, ci = boot_ci(pct, B)
        ok = ~np.isnan(t1)
        res[m] = {"pct": mean, "pct_ci": ci, "top1": float(np.nanmean(t1)), "top5": float(np.nanmean(t5)),
                  "chance_top1": float(np.mean(1.0 / nc[ok])), "n_images": int(ok.sum())}
    d, ci = diff_ci(per["RRR"], per["PCA"], B)
    res["RRR_minus_PCA"] = {"diff": d, "ci": ci}
    return res


def score_p(Yte, preds, masks, B, seed=0):
    n = len(Yte)
    W = ec.boot_weights(n, B, seed + 7)
    sc = ak.Scorer(Yte, [np.arange(n)], masks, W)
    return {k: sc.score(v) for k, v in preds.items()}


def part_A(data, dirs, B, log=print):
    out = {}
    for t in SUBJ:
        X, Y, Xt, Yt, masks = data[t]["Xtr"], data[t]["Ytr"], data[t]["Xte"], data[t]["Yte"], data[t]["masks"]
        ym = Y.mean(0)
        Yc = Y - ym
        d = X.shape[1]
        rnd = np.linalg.qr(np.random.default_rng(9).standard_normal((d, K)))[0]
        preds = {"own_RRR": ak.predict_dir(X, Xt, Yc, ym, dirs[t]["RRR"], K),
                 "own_PCA": ak.predict_dir(X, Xt, Yc, ym, dirs[t]["PCA"], K),
                 "RAND": ak.predict_dir(X, Xt, Yc, ym, rnd, K)}
        srcs = [s for s in SUBJ if s != t]
        for s in srcs:
            preds[f"from_{s}"] = ak.predict_dir(X, Xt, Yc, ym, dirs[s]["RRR"], K)
        sc = score_p(Yt, preds, masks, B)
        P = {k: v["P"] for k, v in sc.items()}
        oth = np.mean([sc[f"from_{s}"]["boot"] for s in srcs], axis=0)
        othP = float(np.mean([P[f"from_{s}"] for s in srcs]))
        dp = oth - sc["own_PCA"]["boot"]
        out[t] = {"P": P, "other_mean": othP, "other_minus_own": othP - P["own_RRR"],
                  "other_minus_own_ci": [float(np.percentile(oth - sc["own_RRR"]["boot"], 2.5)), float(np.percentile(oth - sc["own_RRR"]["boot"], 97.5))],
                  "other_minus_PCA": othP - P["own_PCA"], "other_minus_PCA_ci": [float(np.percentile(dp, 2.5)), float(np.percentile(dp, 97.5))]}
        out[t]["pass"] = bool(out[t]["other_minus_own"] >= -0.01 and out[t]["other_minus_PCA_ci"][0] > 0.01)
        log(f"[A] {t}: 자기 {P['own_RRR']:.3f} 다른 사람 평균 {othP:.3f} PCA {P['own_PCA']:.3f} 무작위 {P['RAND']:.3f} -> {'통과' if out[t]['pass'] else '미통과'}")
    out["verdict"] = {"n_pass": int(sum(out[t]["pass"] for t in SUBJ)), "met": bool(sum(out[t]["pass"] for t in SUBJ) >= 3)}
    return out


def part_B(d, B, cat, log=print):
    S_tr, Y_tr, S_te, Y_te = d["Str"], d["Ytr"], d["Ste"], d["Yte"]
    rand = np.linalg.qr(np.random.default_rng(100).standard_normal((S_tr.shape[1], dc.KMAX)))[0]
    dirs = fit_dirs(S_tr, Y_tr)
    Z_tr, cols = dc.targets(S_tr, dirs, rand)
    Z_te, _ = dc.targets(S_te, dirs, rand)
    Zh = dc.decode(Y_tr, Z_tr, Y_te)
    groups = [cat[c] for c in d["test_ids"]]
    res = eval_groups(method_sims(Zh, Z_te, cols), groups, B, 10)
    res["n_categories_used"] = int(sum(1 for c, n in zip(*np.unique(groups, return_counts=True)) if n >= 10))
    return res


def assign_folds(cats, n_folds=5):
    from collections import Counter
    cnt = Counter(cats)
    size = [0] * n_folds
    fold_of = {}
    for c, n in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])):
        f = int(np.argmin(size))
        fold_of[c] = f
        size[f] += n
    return fold_of, size


def part_C(d, B, cat, log=print, frac=1.0):
    S_tr, Y_tr = d["Str"], d["Ytr"]
    cats = [cat[c] for c in d["train_ids"]]
    fold_of, size = assign_folds(cats)
    fold = np.array([fold_of[c] for c in cats])
    rand = np.linalg.qr(np.random.default_rng(100).standard_normal((S_tr.shape[1], dc.KMAX)))[0]
    per = {m: [] for m in ("RRR", "PCA", "RAND", "full")}
    t1 = {m: [] for m in per}
    t5 = {m: [] for m in per}
    chance = []
    for f in range(5):
        t0 = time.time()
        te = np.flatnonzero(fold == f)
        tr = np.flatnonzero(fold != f)
        if frac < 1.0:  # 학습 이미지를 무작위로 일부만(고정 시드, 비율이 커질수록 앞 비율을 포함)
            tr = np.sort(np.random.default_rng(11 + f).permutation(tr)[:int(frac * len(tr))])
        dirs = fit_dirs(S_tr[tr], Y_tr[tr])
        Z_tr, cols = dc.targets(S_tr[tr], dirs, rand)
        Z_te, _ = dc.targets(S_tr[te], dirs, rand)
        Zh = dc.decode(Y_tr[tr], Z_tr, Y_tr[te])
        sims = method_sims(Zh, Z_te, cols)
        g = np.zeros(len(te))
        for m, Sm in sims.items():
            p, a, b, nc = pct_rank(Sm, g, 2)
            per[m].append(p)
            t1[m].append(a)
            t5[m].append(b)
        chance.append(1.0 / len(te) * np.ones(len(te)))
        log(f"  [C] 폴드 {f + 1}/5: 뺀 범주 {sorted(c for c in fold_of if fold_of[c] == f)[:5]}... 이미지 {len(te)}장, RRR 백분위 {np.mean(per['RRR'][-1]):.3f} PCA {np.mean(per['PCA'][-1]):.3f} ({time.time() - t0:.0f}s)")
    res = {"fold_sizes": size}
    pool = {m: np.concatenate(v) for m, v in per.items()}
    for m in pool:
        mean, ci = boot_ci(pool[m], B)
        res[m] = {"pct": mean, "pct_ci": ci, "top1": float(np.mean(np.concatenate(t1[m]))), "top5": float(np.mean(np.concatenate(t5[m]))),
                  "chance_top1": float(np.mean(np.concatenate(chance)))}
    dd, ci = diff_ci(pool["RRR"], pool["PCA"], B)
    res["RRR_minus_PCA"] = {"diff": dd, "ci": ci}
    return res


def verdicts(A, Bres, Cres):
    v = {"A": A["verdict"]}
    b1 = [Bres[s]["RRR"]["pct_ci"][0] > 0.5 for s in SUBJ]
    b2 = [(Bres[s]["RRR_minus_PCA"]["diff"] >= 0.02 and Bres[s]["RRR_minus_PCA"]["ci"][0] > 0) for s in SUBJ]
    c1 = [Cres[s]["RRR"]["pct_ci"][0] > 0.5 for s in SUBJ]
    c2 = [(Cres[s]["RRR_minus_PCA"]["diff"] >= 0.02 and Cres[s]["RRR_minus_PCA"]["ci"][0] > 0) for s in SUBJ]
    v["B"] = {"B1_all4": bool(all(b1)), "B1_n": int(sum(b1)), "B2_n": int(sum(b2)), "B2_met": bool(sum(b2) >= 2)}
    v["C"] = {"C1_all4": bool(all(c1)), "C1_n": int(sum(c1)), "C2_n": int(sum(c2)), "C2_met": bool(sum(c2) >= 2)}
    return v


def selftest():
    rng = np.random.default_rng(0)
    n = 300
    g = np.repeat(np.arange(10), 30)
    Z = rng.standard_normal((n, 12))
    Sm_good = Z @ (Z + 0.3 * rng.standard_normal((n, 12))).T
    p, a, b, nc = pct_rank(Sm_good, g, 2)
    assert np.nanmean(p) > 0.8 and nc[0] == 30
    Sm_rand = rng.standard_normal((n, n))
    p, a, b, nc = pct_rank(Sm_rand, g, 2)
    assert abs(np.nanmean(p) - 0.5) < 0.05
    fo, size = assign_folds(["a"] * 50 + ["b"] * 30 + ["c"] * 30 + ["d"] * 20 + ["e"] * 20 + ["f"] * 10)
    assert max(size) - min(size) <= 40 and sum(size) == 160
    print("selftest 통과")


def load_all(stage, emb_dir, model):
    from text_compare import load_brain
    emb = Path(emb_dir)
    ids_all = json.load(open(emb / "ctx_ids.json"))
    E = np.load(emb / f"{model}__ctx_P1.npy").astype(np.float32)
    S = (E - E.mean(0)) / (E.std(0) + 1e-6)
    row = {c: i for i, c in enumerate(ids_all)}
    data = {}
    for s in SUBJ:
        train, Ytr, test, Yte, masks, nk, nv = load_brain(f"{stage}/fmri_{s}", ids_all)
        keep = masks["R2"] | masks["R3"]
        data[s] = {"Xtr": S[[row[c] for c in train]], "Ytr": Ytr[:, keep], "Xte": S[[row[c] for c in test]], "Yte": Yte[:, keep],
                   "masks": {g: m[keep] for g, m in masks.items()}, "train_ids": train, "test_ids": test}
        print(f"{s}: 학습 {len(train)}장, shared1000 {len(test)}장, 복셀 {int(keep.sum())}", flush=True)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-dir", default="data/stage")
    ap.add_argument("--emb-dir", default="data/embeddings_v2")
    ap.add_argument("--model", default="MPNet")
    ap.add_argument("--parts", default="A,B,C")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default="data/results/axes_generalize.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    cat = json.load(open("data/results/coco_primary.json"))
    data = load_all(a.stage_dir, a.emb_dir, a.model)
    res = json.load(open(a.out)) if Path(a.out).exists() else {}
    log = lambda s: print(s, flush=True)
    parts = a.parts.split(",")
    if "A" in parts:
        dirs = {s: fit_dirs(data[s]["Xtr"], data[s]["Ytr"]) for s in SUBJ}
        res["A"] = part_A(data, dirs, a.boot, log)
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    if "D" in parts:
        res.setdefault("D", {})
        for s in SUBJ:
            if s in res["D"]:
                continue
            d = {"Str": data[s]["Xtr"], "Ytr": data[s]["Ytr"], "Ste": data[s]["Xte"], "Yte": data[s]["Yte"], "train_ids": data[s]["train_ids"], "test_ids": data[s]["test_ids"]}
            res["D"][s] = {}
            for fr in (0.25, 0.5, 0.75):
                r = part_C(d, a.boot, cat, log, frac=fr)
                res["D"][s][str(fr)] = r
                log(f"[D] {s} {int(fr * 100)}%: RRR 1등 {r['RRR']['top1']:.3f} 백분위 {r['RRR']['pct']:.3f} PCA 1등 {r['PCA']['top1']:.3f} 백분위 {r['PCA']['pct']:.3f}")
                json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
            res["D"][s]["1.0"] = res["C"][s]
        gains = [res["D"][s]["1.0"]["RRR"]["top1"] - res["D"][s]["0.5"]["RRR"]["top1"] for s in SUBJ]
        res["D_verdict"] = {"gain_100_minus_50_top1": gains, "n_ge_0.01": int(sum(g >= 0.01 for g in gains)), "data_sensitive": bool(sum(g >= 0.01 for g in gains) >= 3)}
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
        log("D 판정: " + json.dumps(res["D_verdict"], ensure_ascii=False))
        return
    for key, fn in (("B", part_B), ("C", part_C)):
        if key not in parts:
            continue
        res.setdefault(key, {})
        for s in SUBJ:
            if s in res[key]:
                continue
            d = {"Str": data[s]["Xtr"], "Ytr": data[s]["Ytr"], "Ste": data[s]["Xte"], "Yte": data[s]["Yte"], "train_ids": data[s]["train_ids"], "test_ids": data[s]["test_ids"]}
            res[key][s] = fn(d, a.boot, cat, log)
            r = res[key][s]
            log(f"[{key}] {s}: RRR 백분위 {r['RRR']['pct']:.3f} {np.round(r['RRR']['pct_ci'], 3)} PCA {r['PCA']['pct']:.3f} 무작위 {r['RAND']['pct']:.3f} full {r['full']['pct']:.3f} | 1등 RRR {r['RRR']['top1']:.3f} (우연 {r['RRR']['chance_top1']:.3f}) | RRR−PCA {r['RRR_minus_PCA']['diff']:+.3f} {np.round(r['RRR_minus_PCA']['ci'], 3)}")
            json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    if all(k in res for k in "ABC") and all(s in res["B"] and s in res["C"] for s in SUBJ):
        res["verdicts"] = verdicts(res["A"], res["B"], res["C"])
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
        log("판정: " + json.dumps(res["verdicts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
