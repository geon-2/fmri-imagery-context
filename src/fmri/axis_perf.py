"""B 단계 추가: 축별 예측력 r, 무작위 대조, 축 선택(SEL). 기준은 실행 전에 docs/stageB_axis_perf_criteria.md와 Notion에 기록했다.
사용: python3 src/fmri/axis_perf.py --stage-dir data/stage/fmri_subj01 --emb-dir data/embeddings_v2 --bundle-dir data/colab_bundle --targets ctx_P1,orig_P1 --threads 2 --out-dir data/results/axis_perf
"""
import os
import sys

for i, a in enumerate(sys.argv):
    if a == "--threads":
        for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[v] = sys.argv[i + 1]

import argparse  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import axes_decode as ad  # noqa: E402
import axes_dir_compare as adc  # noqa: E402
import axes_k_curve as ak  # noqa: E402
import compress_compare as cc  # noqa: E402
import embedding_compare as ec  # noqa: E402
import prep_sensitivity as ps  # noqa: E402
import text_decode_compare as td  # noqa: E402

KMAX = 128
KSEL = 32
METHODS = ("RRR", "DDR", "CCA", "PLS", "PCA", "RAND")


def colcorr(P, Tt):
    P, Tt = P - P.mean(0), Tt - Tt.mean(0)
    return (P * Tt).sum(0) / (np.sqrt((P ** 2).sum(0) * (Tt ** 2).sum(0)) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-dir")
    ap.add_argument("--emb-dir")
    ap.add_argument("--bundle-dir")
    ap.add_argument("--targets", default="ctx_P1,orig_P1,obj_P1,clip_img,dinov2,resnet50")
    ap.add_argument("--out-dir", default="data/results/axis_perf")
    ap.add_argument("--threads", default=None)
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    cap = {r["cocoId"]: r for r in map(json.loads, gzip.open(Path(a.bundle_dir) / "captions.jsonl.gz", "rt"))}
    lab = {r["cocoId"]: r for r in map(json.loads, gzip.open(Path(a.bundle_dir) / "labels_pass0.jsonl.gz", "rt"))}
    ids_all = json.load(open(Path(a.emb_dir) / "ctx_ids.json"))
    train, Y, masks, nkeep, nvox = ps.load_variant(a.stage_dir, ids_all, True, "rel", log=lambda s: print(s, flush=True))
    T = cc.load_targets(a, ids_all, train, cap, lab)
    names = a.targets.split(",")
    n = len(train)
    if a.fast:
        n = 600
        Y = Y[:n]
        T = {k: v[:n] for k, v in T.items()}
    folds = ec.outer_folds(n, 0)
    if a.fast:
        folds = folds[:2]
    nf_ = len(folds)
    R = {(t, m): np.zeros((nf_, KMAX)) for t in names for m in METHODS}
    SEL = {(t, m): {x: np.full(n, np.nan, np.float32) for x in ("c1", "c5", "s1")} for t in names for m in METHODS}
    FIRST = {(t, m): {x: np.full(n, np.nan, np.float32) for x in ("c1", "c5")} for t in names for m in METHODS}
    rng = np.random.default_rng(1)
    t0 = time.time()
    print(f"축별 예측력: 이미지 {n}장, 타깃 {names}", flush=True)
    for fi, f in enumerate(folds):
        tr = np.setdiff1d(np.arange(n), f)
        Z, Zt = adc.zscore(Y[tr], Y[f])
        full = np.linalg.svd(Z, full_matrices=False)
        U, s, Vt = full
        perm = rng.permutation(len(tr))
        inner = []
        for sp in np.array_split(perm, 3):
            itr = np.setdiff1d(perm, sp)
            inner.append((itr, sp, np.linalg.svd(Z[itr], full_matrices=False)))
        prep, info, cols_tr, cols_te = {}, [], [], []
        for t in names:
            X = T[t]
            p = ak.prep(X[tr], Y[tr], X[f], rng)
            d = X.shape[1]
            Dd, lam = cc.ddr_cached(Z, p["Xa"], inner, full)
            D = {"RRR": p["D"]["RRR"], "PLS": p["D"]["PLS"], "PCA": p["D"]["PCA"], "DDR": Dd, "CCA": adc.cca_dirs(p["Xa"], U, s)}
            Q = {m: cc.qr_prefix(D[m], KMAX) for m in D}
            Q["RAND"] = np.linalg.qr(np.random.default_rng(100 + fi).standard_normal((d, KMAX)))[0].astype(np.float32)
            prep[t] = (p, Q)
            for m in METHODS:
                cols_tr.append(p["Xa"] @ Q[m])
                cols_te.append(p["Xt"] @ Q[m])
                info.append((t, m))
        Ctr, Cte = np.concatenate(cols_tr, axis=1), np.concatenate(cols_te, axis=1)
        Zhat = ad.decode(Y[tr], Ctr, Y[f], seed=fi)
        h = np.random.default_rng(50 + fi).permutation(len(tr))
        h1, h2 = h[:len(tr) // 2], h[len(tr) // 2:]
        Zin = ad.decode(Y[tr][h1], Ctr[h1], Y[tr][h2], seed=fi)
        shp = np.random.default_rng(7 + fi).permutation(len(f))
        for ci, (t, m) in enumerate(info):
            sl = slice(ci * KMAX, (ci + 1) * KMAX)
            p, Q = prep[t]
            R[(t, m)][fi] = colcorr(Zhat[:, sl], Cte[:, sl])
            rin = colcorr(Zin[:, sl], Ctr[h2][:, sl])
            sel = np.argsort(-rin)[:KSEL]
            for tag, idx in (("sel", sel), ("first", np.arange(KSEL))):
                rec = Zhat[:, sl][:, idx] @ Q[m][:, idx].T
                Sm = ad.sim_matrix(rec, p["Xt"])
                c1, c5 = td.credit(Sm)
                if tag == "sel":
                    SEL[(t, m)]["c1"][f], SEL[(t, m)]["c5"][f] = c1, c5
                    SEL[(t, m)]["s1"][f] = td.credit(Sm[:, shp])[0]
                else:
                    FIRST[(t, m)]["c1"][f], FIRST[(t, m)]["c5"][f] = c1, c5
        print(f"  폴드 {fi + 1}/{nf_} 완료 ({time.time() - t0:.0f}s)", flush=True)
    o = Path(a.out_dir)
    o.mkdir(parents=True, exist_ok=True)
    tag = "_".join(names) + ("_fast" if a.fast else "")
    np.savez_compressed(o / f"axis_perf_{tag}.npz", **{f"R__{t}__{m}": v for (t, m), v in R.items()},
                        **{f"SEL__{t}__{m}__{x}": v for (t, m), d in SEL.items() for x, v in d.items()},
                        **{f"FIRST__{t}__{m}__{x}": v for (t, m), d in FIRST.items() for x, v in d.items()})
    print("완료", flush=True)


if __name__ == "__main__":
    main()
