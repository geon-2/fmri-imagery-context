"""B 단계: 복셀·축 압축 방법 비교. 기준은 실행 전에 docs/stageB_compression_criteria.md와 Notion에 기록했다.

방법: RRR, DDR, CCA, PLS, PCA(타깃 자체), RAND(대조)는 타깃 공간의 축 k개를 학습 폴드에서 구하고, 복셀 → 축 좌표를 ridge로 예측해 축으로 전체 타깃을 복원한다.
      VPCA는 복셀 PC k개에서 전체 타깃을 ridge로 예측한다. NONE은 압축 없음.
주 지표: 복원한 전체 타깃(표준화 공간) 대 실제 타깃으로 보류 폴드 안에서 1장 찾기(top-1, top-5, 동점 부분 점수).
병렬: 방법·k는 한 번의 decoding 호출에 좌표를 묶어서 계산하고, 타깃 종류는 --targets로 나눠 여러 프로세스로 돌린다(--threads로 프로세스당 스레드 제한).
사용: python3 src/fmri/compress_compare.py --stage-dir data/stage/fmri_subj01 --emb-dir data/embeddings_v2 --bundle-dir data/colab_bundle --targets ctx_P1,orig_P1 --threads 2 --out-dir data/results/compress_compare
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
import embedding_compare as ec  # noqa: E402
import prep_sensitivity as ps  # noqa: E402
import text_decode_compare as td  # noqa: E402
from text_compare import build_sets, item_counts, load_features  # noqa: E402

KS = [5, 8, 12, 20, 32, 50, 80, 128]
KMAX = 128
AXIS_METHODS = ("RRR", "DDR", "CCA", "PLS", "PCA", "RAND")
IMG = ("clip_img", "dinov2", "resnet50")
K_SHUF = 32


def ddr_cached(Z, Xa, inner, full, lams=adc.DDR_LAMBDAS):
    """adc.ddr_dirs와 같은 계산이지만 복셀 SVD를 여러 타깃이 공유한다."""
    err = np.zeros(len(lams))
    for tr, f, (U, s, Vt) in inner:
        mu = Xa[tr].mean(0)
        UtX = U.T @ (Xa[tr] - mu)
        for i, lam in enumerate(lams):
            coef = Vt.T @ ((s / (s ** 2 + lam))[:, None] * UtX)
            err[i] += (((Xa[f] - mu) - Z[f] @ coef) ** 2).sum()
    lam = lams[err.argmin()]
    U, s, Vt = full
    Xhat = U @ ((s ** 2 / (s ** 2 + lam))[:, None] * (U.T @ (Xa - Xa.mean(0))))
    return np.linalg.svd(Xhat, full_matrices=False)[2].T, float(lam)


def qr_prefix(D, k=KMAX):
    return np.linalg.qr(D[:, :k])[0].astype(np.float32)


def load_targets(a, ids_all, train, cap, lab):
    counts = item_counts(cap, lab, train)
    F = load_features(a.emb_dir, "CLIP", ids_all, train)
    F.pop("ctx_V3", None)
    S = build_sets(F, counts, 0)
    T = {r: S[r].astype(np.float32) for r in ("orig_P1", "obj_P1", "ctx_P1")}
    allp = {k: Path(a.emb_dir) / f"IMGALL__{k}.npy" for k in IMG}
    if all(p.exists() for p in allp.values()):          # 전체 이미지 임베딩(ctx_ids.json 순서)에서 피험자 학습 이미지만 골라 쓴다
        row = {c: i for i, c in enumerate(ids_all)}
        sel = [row[c] for c in train]
        for k in IMG:
            T[k] = np.load(allp[k])[sel].astype(np.float32)
    else:
        for k in IMG:
            T[k] = np.load(Path(a.emb_dir) / f"IMG__{k}.npy").astype(np.float32)
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-dir")
    ap.add_argument("--emb-dir")
    ap.add_argument("--bundle-dir")
    ap.add_argument("--targets", default="ctx_P1,orig_P1,obj_P1,clip_img,dinov2,resnet50")
    ap.add_argument("--out-dir", default="data/results/compress_compare")
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", default=None)
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    cap = {r["cocoId"]: r for r in map(json.loads, gzip.open(Path(a.bundle_dir) / "captions.jsonl.gz", "rt"))}
    lab = {r["cocoId"]: r for r in map(json.loads, gzip.open(Path(a.bundle_dir) / "labels_pass0.jsonl.gz", "rt"))}
    ids_all = json.load(open(Path(a.emb_dir) / "ctx_ids.json"))
    train, Y, masks, nkeep, nvox = ps.load_variant(a.stage_dir, ids_all, True, "rel", log=lambda s: print(s, flush=True))
    T = load_targets(a, ids_all, train, cap, lab)
    names = a.targets.split(",")
    n = len(train)
    ks = KS
    if a.fast:
        n = 600
        Y = Y[:n]
        T = {k: v[:n] for k, v in T.items()}
        ks = [5, 12, 32]
    kmax = max(ks)
    folds = ec.outer_folds(n, 0)
    if a.fast:
        folds = folds[:2]
    methods = AXIS_METHODS + ("VPCA", "NONE")
    out = {(t, m, k): {x: np.full(n, np.nan, np.float32) for x in ("c1", "c5", "o1", "o5", "s1")} for t in names for m in methods for k in (ks if m != "NONE" else [0])}
    rng = np.random.default_rng(1)
    t0 = time.time()
    print(f"B 단계: 이미지 {n}장, 복셀 {Y.shape[1]}, 타깃 {names}, k {ks}", flush=True)
    for fi, f in enumerate(folds):
        tr = np.setdiff1d(np.arange(n), f)
        Z, Zt = adc.zscore(Y[tr], Y[f])
        full = np.linalg.svd(Z, full_matrices=False)
        perm = rng.permutation(len(tr))
        inner = []
        for sp in np.array_split(perm, 3):
            itr = np.setdiff1d(perm, sp)
            inner.append((itr, sp, np.linalg.svd(Z[itr], full_matrices=False)))
        U, s, Vt = full
        prep = {}
        cols_tr, cols_info = [], []
        for t in names:
            X = T[t]
            p = ak.prep(X[tr], Y[tr], X[f], rng)
            d = X.shape[1]
            Dd, lam = ddr_cached(Z, p["Xa"], inner, full)
            D = {"RRR": p["D"]["RRR"], "PLS": p["D"]["PLS"], "PCA": p["D"]["PCA"], "DDR": Dd, "CCA": adc.cca_dirs(p["Xa"], U, s)}
            Q = {m: qr_prefix(D[m], kmax) for m in D}
            Q["RAND"] = np.linalg.qr(np.random.default_rng(100 + fi).standard_normal((d, kmax)))[0].astype(np.float32)
            prep[t] = (p, Q)
            for m in AXIS_METHODS:
                cols_tr.append(p["Xa"] @ Q[m])
                cols_info.append((t, m))
            print(f"  [{t}] 폴드 {fi + 1} 축 계산 완료 (DDR λ={lam:.0e}, {time.time() - t0:.0f}s)", flush=True)
        Zhat = ad.decode(Y[tr], np.concatenate(cols_tr, axis=1), Y[f], seed=fi)
        for ci, (t, m) in enumerate(cols_info):
            p, Q = prep[t]
            C = Zhat[:, ci * kmax:(ci + 1) * kmax]
            shp = np.random.default_rng(7 + fi).permutation(len(f))
            for k in ks:
                rec = C[:, :k] @ Q[m][:, :k].T
                c1, c5 = td.credit(ad.sim_matrix(rec, p["Xt"]))
                orc = (p["Xt"] @ Q[m][:, :k]) @ Q[m][:, :k].T
                o1, o5 = td.credit(ad.sim_matrix(orc, p["Xt"]))
                o = out[(t, m, k)]
                o["c1"][f], o["c5"][f], o["o1"][f], o["o5"][f] = c1, c5, o1, o5
                if k == K_SHUF:
                    o["s1"][f] = td.credit(ad.sim_matrix(rec, p["Xt"])[:, shp])[0]
        # 압축 없음
        Xall = np.concatenate([prep[t][0]["Xa"] for t in names], axis=1)
        Zh0 = ad.decode(Y[tr], Xall, Y[f], seed=fi)
        c = 0
        for t in names:
            d = T[t].shape[1]
            p = prep[t][0]
            rec = Zh0[:, c:c + d]
            c += d
            c1, c5 = td.credit(ad.sim_matrix(rec, p["Xt"]))
            o = out[(t, "NONE", 0)]
            o["c1"][f], o["c5"][f] = c1, c5
            o["s1"][f] = td.credit(ad.sim_matrix(rec, p["Xt"])[:, np.random.default_rng(7 + fi).permutation(len(f))])[0]
        # VPCA
        for k in ks:
            Str, Ste = U[:, :k] * s[:k], Zt @ Vt[:k].T
            Zhv = ad.decode(Str.astype(np.float32), Xall, Ste.astype(np.float32), seed=fi)
            c = 0
            for t in names:
                d = T[t].shape[1]
                p = prep[t][0]
                rec = Zhv[:, c:c + d]
                c += d
                c1, c5 = td.credit(ad.sim_matrix(rec, p["Xt"]))
                o = out[(t, "VPCA", k)]
                o["c1"][f], o["c5"][f] = c1, c5
                if k == K_SHUF:
                    o["s1"][f] = td.credit(ad.sim_matrix(rec, p["Xt"])[:, np.random.default_rng(7 + fi).permutation(len(f))])[0]
        print(f"  폴드 {fi + 1}/{len(folds)} 완료 ({time.time() - t0:.0f}s)", flush=True)
    o_ = Path(a.out_dir)
    o_.mkdir(parents=True, exist_ok=True)
    tag = a.tag or "_".join(names)
    np.savez_compressed(o_ / f"compress_{tag}{'_fast' if a.fast else ''}.npz", **{f"{t}__{m}__{k}__{x}": v for (t, m, k), d in out.items() for x, v in d.items()})
    json.dump({"n": n, "n_voxels": int(Y.shape[1]), "targets": names, "ks": ks, "methods": list(methods)}, open(o_ / f"compress_{tag}{'_fast' if a.fast else ''}_meta.json", "w"), ensure_ascii=False)
    print("완료", flush=True)


if __name__ == "__main__":
    main()
