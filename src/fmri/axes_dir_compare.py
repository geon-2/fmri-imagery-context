"""E2 축 방법 비교: 인코딩 방향(RRR)과 decoding 방향(DDR), PLS, CCA, PCA, 무작위를 같은 k에서 encoding·decoding 지표로 비교한다.
기준은 실행 전에 docs/prep_axes_criteria.md와 Notion(E1·E2 기준)에 기록했다.

방향은 학습 폴드의 이미지로만 구한다(임베딩 공간 d차원의 방향 D[:, :k]).
  RRR  ridge 인코딩 계수의 적합값 SVD(기존)             DDR  복셀 → 임베딩 ridge 디코딩의 예측값 SVD(출력 공간 상위 방향)
  PLS  교차공분산 SVD(기존)                              CCA  복셀을 상위 PC 200개로 줄여 정규화 CCA(임베딩 쪽 가중치)
  PCA  임베딩 자체의 분산 방향(기존)                     RAND 무작위 직교 방향
지표  encoding P: 축 k개 위에서 복셀을 OLS로 다시 적합(axes_k_curve와 같음, 5겹, 이미지 부트스트랩 1,000회)
      decoding  : 복셀에서 k개 축 좌표를 ridge로 예측해 보류 폴드 이미지끼리 1장 찾기 top-1·top-5(후보 = 보류 폴드 전체)
사용: python3 src/fmri/axes_dir_compare.py --fmri-dir data/stage/fmri_subj01 --emb-dir data/embeddings_v2 --bundle-dir data/colab_bundle --out-dir data/results/axes_dir_compare
시험: python3 src/fmri/axes_dir_compare.py --selftest
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labeling"))
import axes_decode as ad  # noqa: E402
import axes_k_curve as ak  # noqa: E402
import embedding_compare as ec  # noqa: E402

KS = [1, 2, 3, 5, 8, 12, 20, 32, 50, 80, 128]
METHODS = ("RRR", "DDR", "PLS", "CCA", "PCA", "RAND")
DDR_LAMBDAS = 10.0 ** np.arange(2.0, 7.01, 1.0)
CCA_PCS, CCA_RIDGE = 200, 0.1


def zscore(Ytr, Yte):
    mu, sd = Ytr.mean(0), Ytr.std(0) + 1e-6
    return (Ytr - mu) / sd, (Yte - mu) / sd


def ddr_dirs(Z, Xa, rng, k_inner=3):
    """복셀 Z(n×V, 표준화) → 임베딩 Xa(n×d) ridge(람다 하나, 내부 교차검증)의 예측값 SVD. 반환 d×d(중요도 순), 선택한 람다."""
    n = len(Z)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k_inner)
    err = np.zeros(len(DDR_LAMBDAS))
    for f in folds:
        tr = np.setdiff1d(perm, f)
        U, s, _ = np.linalg.svd(Z[tr], full_matrices=False)
        UtX = U.T @ (Xa[tr] - Xa[tr].mean(0))
        Vt = np.linalg.svd(Z[tr], full_matrices=False)[2]
        for i, lam in enumerate(DDR_LAMBDAS):
            coef = Vt.T @ ((s / (s ** 2 + lam))[:, None] * UtX)
            err[i] += (((Xa[f] - Xa[tr].mean(0)) - Z[f] @ coef) ** 2).sum()
    lam = DDR_LAMBDAS[err.argmin()]
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    Xhat = U @ ((s ** 2 / (s ** 2 + lam))[:, None] * (U.T @ (Xa - Xa.mean(0))))
    return np.linalg.svd(Xhat, full_matrices=False)[2].T, float(lam), (U, s)


def cca_dirs(Xa, U, s, n_pc=CCA_PCS, ridge=CCA_RIDGE):
    """복셀 PC(백색화)와의 정규화 CCA. 임베딩 쪽 가중치 d×n_pc(정준상관 순)."""
    n, d = Xa.shape
    n_pc = min(n_pc, U.shape[1])
    Wy = U[:, :n_pc] * np.sqrt(n)                       # 단위 분산 복셀 PC
    Xc = Xa - Xa.mean(0)
    Cxx = Xc.T @ Xc / n + ridge * np.eye(d)
    ev, Q = np.linalg.eigh(Cxx)
    Cih = (Q / np.sqrt(ev)) @ Q.T
    M = Cih @ (Xc.T @ Wy / n)
    Um = np.linalg.svd(M, full_matrices=False)[0]
    return Cih @ Um


def overlap(D1, D2, k):
    Q1, Q2 = np.linalg.qr(D1[:, :k])[0], np.linalg.qr(D2[:, :k])[0]
    return float(np.linalg.norm(Q1.T @ Q2) ** 2 / k)


def run(X, Y, masks, ks=KS, B=1000, seed=0, log=print, overlap_ks=(5, 12, 20)):
    n, d = X.shape
    ks = [k for k in ks if k <= min(d, CCA_PCS)]
    kmax = max(ks)
    folds = ec.outer_folds(n, seed)
    W = ec.boot_weights(n, B, seed + 7) if B else None
    sc = ak.Scorer(Y, folds, masks, W)
    rng = np.random.default_rng(seed + 1)
    rand = np.linalg.qr(np.random.default_rng(seed + 100).standard_normal((d, kmax)))[0]
    enc = {m: {k: np.empty(Y.shape, np.float32) for k in ks} for m in METHODS}
    top = {m: {k: {1: np.zeros(n, bool), 5: np.zeros(n, bool)} for k in ks} for m in METHODS + ("full",)}
    ov = {}
    lam_used = []
    t0 = time.time()
    for fi, f in enumerate(folds):
        tr = np.setdiff1d(np.arange(n), f)
        p = ak.prep(X[tr], Y[tr], X[f], rng)
        Z, Zt = zscore(Y[tr], Y[f])
        Dddr, lam, (U, s) = ddr_dirs(Z, p["Xa"], rng)
        lam_used.append(lam)
        D = {"RRR": p["D"]["RRR"], "PLS": p["D"]["PLS"], "PCA": p["D"]["PCA"], "DDR": Dddr, "CCA": cca_dirs(p["Xa"], U, s), "RAND": rand}
        for m in METHODS:
            for k in ks:
                enc[m][k][f] = ak.predict_dir(p["Xa"], p["Xt"], p["Yc"], p["ym"], D[m], k)
        # decoding: 모든 방법의 좌표를 한 번에 예측(타깃은 서로 독립)
        Ztr_all = np.concatenate([p["Xa"] @ D[m][:, :kmax] for m in METHODS] + [p["Xa"]], axis=1).astype(np.float32)
        Zte_all = np.concatenate([p["Xt"] @ D[m][:, :kmax] for m in METHODS] + [p["Xt"]], axis=1).astype(np.float32)
        Zhat = ad.decode(Y[tr], Ztr_all, Y[f], seed=seed + fi)
        cols = {m: slice(i * kmax, (i + 1) * kmax) for i, m in enumerate(METHODS)}
        cols["full"] = slice(len(METHODS) * kmax, len(METHODS) * kmax + d)
        for m, sl in cols.items():
            for k in ([k for k in ks if k > 1] if m != "full" else [ks[0]]):
                sub = slice(sl.start, sl.start + k) if m != "full" else sl
                Sm = ad.sim_matrix(Zhat[:, sub], Zte_all[:, sub])
                rank = (Sm > np.diag(Sm)[:, None]).sum(1)
                top[m][k][1][f] = rank < 1
                top[m][k][5][f] = rank < 5
        for k in overlap_ks:
            if k <= kmax:
                for a, b in [(a, b) for i, a in enumerate(METHODS) for b in METHODS[i + 1:]]:
                    ov.setdefault(f"{a}-{b}@{k}", []).append(overlap(D[a], D[b], k))
        log(f"  폴드 {fi + 1}/{len(folds)} 완료 (DDR λ={lam:.0e}, {time.time() - t0:.0f}s)")
    res = {"n_images": n, "n_voxels": int(Y.shape[1]), "dim": d, "ks": ks, "ddr_lambda_per_fold": lam_used, "encoding": {}, "decoding": {}, "overlap": {}}
    boots = {}
    for m in METHODS:
        res["encoding"][m] = []
        for k in ks:
            r = sc.score(enc[m][k])
            boots[(m, k)] = r.get("boot")
            res["encoding"][m].append({"k": k, "P": r["P"], "ci": [float(np.percentile(r["boot"], 2.5)), float(np.percentile(r["boot"], 97.5))] if W is not None else None})
        # k=1은 코사인 유사도가 ±1로 퇴화해 top-k가 의미 없으므로 제외한다
        res["decoding"][m] = [{"k": k, "top1": float(top[m][k][1].mean()) if k > 1 else None, "top5": float(top[m][k][5].mean()) if k > 1 else None} for k in ks]
    res["decoding"]["full"] = [{"k": d, "top1": float(top["full"][ks[0]][1].mean()), "top5": float(top["full"][ks[0]][5].mean())}]
    # 비교: DDR − RRR (그리고 모든 방법 − 무작위) at k=5,12,20
    rngb = np.random.default_rng(seed + 33)
    idx = [rngb.integers(0, n, n) for _ in range(B)] if B else []
    res["contrasts"] = {}
    for k in (5, 12, 20):
        if k not in ks:
            continue
        row = {}
        for a, b in [("DDR", "RRR"), ("CCA", "RRR"), ("PLS", "RRR"), ("PCA", "RRR"), ("RRR", "RAND"), ("DDR", "RAND"), ("CCA", "RAND"), ("PLS", "RAND"), ("PCA", "RAND")]:
            e = {"diff": float(next(x["P"] for x in res["encoding"][a] if x["k"] == k) - next(x["P"] for x in res["encoding"][b] if x["k"] == k))}
            if B:
                dd = boots[(a, k)] - boots[(b, k)]
                e["ci"] = [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]
            row[f"{a}-{b}"] = {"encoding_P": e}
            for kk in (1, 5):
                v = top[a][k][kk].astype(float) - top[b][k][kk].astype(float)
                e2 = {"diff": float(v.mean())}
                if B:
                    bm = np.array([v[i].mean() for i in idx])
                    e2["ci"] = [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]
                row[f"{a}-{b}"][f"decoding_top{kk}"] = e2
        res["contrasts"][f"k{k}"] = row
    res["overlap"] = {key: float(np.mean(v)) for key, v in ov.items()}
    res["overlap_random_expect"] = {f"k{k}": k / d for k in overlap_ks}
    return res, {f"{m}__{k}": v for (m, k), v in boots.items() if v is not None}, {f"{m}__{k}__top{kk}": top[m][k][kk] for m in top for k in top[m] for kk in (1, 5)}


def selftest():
    rng = np.random.default_rng(0)
    n, d, V = 360, 20, 80
    sig = rng.standard_normal((n, d))
    Wt = rng.standard_normal((d, V))
    Y = (sig @ Wt * 0.5 + rng.standard_normal((n, V)) * 2).astype(np.float32)
    masks = {"R2": np.arange(V) < 50, "R3": np.arange(V) >= 30}
    old = ad.DEC_ALPHAS
    res, _, _ = run(sig.astype(np.float32), Y, masks, ks=[1, 3, 8, 12], B=30, log=lambda s: None)
    for m in METHODS:
        print(m, [round(x["P"], 3) for x in res["encoding"][m]], [None if x["top1"] is None else round(x["top1"], 3) for x in res["decoding"][m]])
    P = {m: res["encoding"][m][-1]["P"] for m in METHODS}
    assert P["RRR"] > P["RAND"] and P["DDR"] > P["RAND"] - 0.02, P
    assert res["decoding"]["full"][0]["top1"] > 0.05
    print("selftest 통과")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmri-dir")
    ap.add_argument("--emb-dir")
    ap.add_argument("--bundle-dir")
    ap.add_argument("--out-dir", default="data/results/axes_dir_compare")
    ap.add_argument("--model", default="MPNet")
    ap.add_argument("--rep", default="ctx_P1")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import prep_sensitivity as ps
    ids_all = json.load(open(Path(a.emb_dir) / "ctx_ids.json"))
    train, Y, masks, nkeep, nvox = ps.load_variant(a.fmri_dir, ids_all, True, "rel", log=lambda s: print(s, flush=True))
    row = {c: i for i, c in enumerate(ids_all)}
    X = np.load(Path(a.emb_dir) / f"{a.model}__{a.rep}.npy").astype(np.float32)[[row[c] for c in train]]
    print(f"E2: 이미지 {len(train)}장, 임베딩 {a.model} {a.rep} 차원 {X.shape[1]}, 복셀 {Y.shape[1]}", flush=True)
    res, boots, tops = run(X, Y, masks, B=a.boot, log=lambda s: print(s, flush=True))
    res.update(model=a.model, rep=a.rep)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / f"axes_dir_compare_{a.model}_{a.rep}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    np.savez_compressed(out / f"axes_dir_compare_{a.model}_{a.rep}_boots.npz", **boots)
    np.savez_compressed(out / f"axes_dir_compare_{a.model}_{a.rep}_top.npz", **tops)
    print("저장:", out)


if __name__ == "__main__":
    main()
