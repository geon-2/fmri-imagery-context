"""E3: 잡음 제거를 복셀 값에 적용한 뒤 1단계 decoding 비교를 반복한다. 기준은 실행 전에 docs/denoise_stage1_criteria.md와 Notion에 기록했다.

조건  D0 기준(현재)  D1 PCA 잡음 제거(k=100,300,1000)  D2 공간 스무딩(FWHM 1, 2 복셀)  D3 잡음 백색화(수축 α=0.5, 대각 수축)
고정  피험자 1, 전처리 a로 고른 복셀 6,728개, 학습용 4,999장 5겹, CLIP, 같은 ridge decoding과 평가(top-1·top-5, 동점 부분 점수).
누수  PCA 성분·잡음 공분산·백색화는 학습 폴드 이미지에서만 구한다. 스무딩은 복셀 좌표만 쓴다.
사용: python3 src/fmri/denoise_decode.py --fmri-dir data/stage/fmri_subj01 --emb-dir data/embeddings_v2 --bundle-dir data/colab_bundle --roi-dir data/raw/nsd_fmri/subj01/roi_betas
시험: python3 src/fmri/denoise_decode.py --selftest
"""
import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labeling"))
import axes_decode as ad  # noqa: E402
import embedding_compare as ec  # noqa: E402
import prep_sensitivity as ps  # noqa: E402
import text_decode_compare as td  # noqa: E402

REPS = ("orig_P1", "obj_P1", "ctx_P1", "ctx_V2", "obj_P1+ctx_P1", "ctx_T10", "ctx_P1_shuf")
ALPHA = 0.5
KS_PCA = (100, 300, 1000)
FWHMS = (1.0, 2.0)


def load_reps(fmri_dir, ids_all):
    """전처리 a의 반복 반응(전체 R2∪R3 복셀, 세션 z-score)과 a가 고른 복셀 마스크, 학습 이미지 목록."""
    fmri_dir = Path(fmri_dir)
    tm = list(csv.DictReader(open(fmri_dir / "trial_map.csv", encoding="utf-8")))
    groups = np.load(fmri_dir / "roi_groups.npz")
    allmask = groups["R2"] | groups["R3"]
    Y = np.empty((len(tm), int(allmask.sum())), np.float32)
    for s in range(1, 41):
        a = np.load(fmri_dir / f"session{s:02d}.npy").astype(np.float32)[:, allmask]
        Y[(s - 1) * 750:s * 750] = (a - a.mean(0)) / (a.std(0) + 1e-6)
    trials = {}
    for r in tm:
        trials.setdefault(r["cocoId"], []).append(int(r["trial"]))
    shared = {r["cocoId"] for r in tm if r["shared1000"] == "1"}
    have = set(ids_all)
    train = [c for c in trials if c not in shared and c in have and len(trials[c]) == 3]
    T = np.array([trials[c] for c in train])
    reps = Y[T]
    del Y
    masks_full = {g: groups[g][allmask] for g in ("R2", "R3")}
    keep = ps.select_voxels(reps, masks_full, "rel")
    return train, reps, keep, allmask


def voxel_coords(roi_dir, allmask):
    v = np.load(Path(roi_dir) / "voxels.npz")
    idx = np.unravel_index(v["flat_index"], tuple(v["shape"]))
    return np.stack(idx, axis=1)[allmask].astype(np.float32)


def smoothing_matrix(coords, fwhm):
    sigma = fwhm / 2.355
    r = max(1.0, 2.5 * sigma)
    tree = cKDTree(coords)
    d = tree.sparse_distance_matrix(tree, r, output_type="coo_matrix")
    w = np.exp(-(d.data ** 2) / (2 * sigma ** 2))
    W = sp.coo_matrix((w, (d.row, d.col)), shape=d.shape).tocsr() + sp.eye(len(coords), format="csr")  # 거리 0(자기 자신)은 sparse에서 빠지므로 더함
    W = sp.diags(1.0 / np.asarray(W.sum(1)).ravel()) @ W
    return W.tocsr()


def smooth_mean(reps, W):
    n, k, V = reps.shape
    out = np.zeros((n, V), np.float32)
    for r in range(k):
        out += (W @ reps[:, r, :].T).T
    return (out / k).astype(np.float32)


def pca_fold(Ytr, Yte, ks):
    mu = Ytr.mean(0)
    U, s, Vt = np.linalg.svd(Ytr - mu, full_matrices=False)
    out = {}
    for k in ks:
        Vk = Vt[:k]
        out[k] = ((Ytr - mu) @ Vk.T @ Vk + mu).astype(np.float32), ((Yte - mu) @ Vk.T @ Vk + mu).astype(np.float32)
    return out


def whiten_fold(reps_tr, Ytr, Yte, alpha=ALPHA):
    n, k, V = reps_tr.shape
    E = (reps_tr - reps_tr.mean(1, keepdims=True)).reshape(n * k, V)
    S = (E.T @ E) / (n * (k - 1))
    S = (1 - alpha) * S + alpha * np.diag(np.diag(S))
    ev, Q = np.linalg.eigh(S.astype(np.float64))
    Wm = ((Q / np.sqrt(np.maximum(ev, 1e-8))) @ Q.T).astype(np.float32)
    mu = Ytr.mean(0)
    return ((Ytr - mu) @ Wm).astype(np.float32), ((Yte - mu) @ Wm).astype(np.float32)


def evaluate(cond, fold_data, S, n, folds, log):
    """fold_data(fi, tr, f) -> (Ytr_t, Yte_t). 표현 S는 {이름: n×D}."""
    names = list(S)
    out = {r: {"c1": np.zeros(n, np.float32), "c5": np.zeros(n, np.float32)} for r in names}
    t0 = time.time()
    for fi, f in enumerate(folds):
        tr = np.setdiff1d(np.arange(n), f)
        Ytr, Yte = fold_data(fi, tr, f)
        Ztr = np.concatenate([S[r][tr] for r in names], axis=1).astype(np.float32)
        Zte = np.concatenate([S[r][f] for r in names], axis=1).astype(np.float32)
        Zhat = ad.decode(Ytr, Ztr, Yte, seed=fi)
        c = 0
        for r in names:
            d = S[r].shape[1]
            sl = slice(c, c + d)
            c += d
            c1, c5 = td.credit(ad.sim_matrix(Zhat[:, sl], Zte[:, sl]))
            out[r]["c1"][f], out[r]["c5"][f] = c1, c5
        log(f"  [{cond}] 폴드 {fi + 1}/{len(folds)} ({time.time() - t0:.0f}s)")
    return out


def selftest():
    rng = np.random.default_rng(0)
    coords = np.stack(np.meshgrid(np.arange(6), np.arange(6), np.arange(6), indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
    W = smoothing_matrix(coords, 2.0)
    assert abs(float(W.sum(1).min()) - 1) < 1e-4 and W.shape == (216, 216)
    reps = rng.standard_normal((50, 3, 216)).astype(np.float32)
    s = smooth_mean(reps, W)
    assert s.shape == (50, 216) and s.std() < reps.mean(1).std()                  # 스무딩은 분산을 줄인다
    Ytr, Yte = reps.mean(1)[:40], reps.mean(1)[40:]
    p = pca_fold(Ytr, Yte, (5, 20))
    assert p[5][0].shape == Ytr.shape
    a, b = whiten_fold(reps[:40], Ytr, Yte)
    assert a.shape == Ytr.shape and np.isfinite(a).all()
    print("selftest 통과")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmri-dir")
    ap.add_argument("--emb-dir")
    ap.add_argument("--bundle-dir")
    ap.add_argument("--roi-dir")
    ap.add_argument("--out-dir", default="data/results/denoise_decode")
    ap.add_argument("--conds", default="D0,D1,D2,D3")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    from text_compare import build_sets, item_counts, load_features
    cap = {r["cocoId"]: r for r in map(json.loads, gzip.open(Path(a.bundle_dir) / "captions.jsonl.gz", "rt"))}
    lab = {r["cocoId"]: r for r in map(json.loads, gzip.open(Path(a.bundle_dir) / "labels_pass0.jsonl.gz", "rt"))}
    ids_all = json.load(open(Path(a.emb_dir) / "ctx_ids.json"))
    train, reps, keep, allmask = load_reps(a.fmri_dir, ids_all)
    n = len(train)
    print(f"E3: 이미지 {n}장, 전체 복셀 {reps.shape[2]}, a 선택 복셀 {int(keep.sum())}", flush=True)
    counts = item_counts(cap, lab, train)
    F = load_features(a.emb_dir, "CLIP", ids_all, train)
    F.pop("ctx_V3", None)
    Sall = build_sets(F, counts, 0)
    row = {c: i for i, c in enumerate(ids_all)}
    Sall["ctx_T10"] = np.load(Path(a.emb_dir) / "CLIP__ctx_T10.npy").astype(np.float32)[[row[c] for c in train]]
    S = {r: Sall[r].astype(np.float32) for r in REPS}
    folds = ec.outer_folds(n, 0)
    Y0 = reps.mean(1)[:, keep]
    reps_k = reps[:, :, keep]
    conds = a.conds.split(",")
    o = Path(a.out_dir)
    o.mkdir(parents=True, exist_ok=True)
    log = lambda s: print(s, flush=True)
    results = {}

    def save(name, out):
        results[name] = out
        np.savez_compressed(o / f"denoise_{name}.npz", **{f"{r}__{k}": v[k] for r, v in out.items() for k in ("c1", "c5")})

    if "D0" in conds:
        save("D0", evaluate("D0", lambda fi, tr, f: (Y0[tr], Y0[f]), S, n, folds, log))
    if "D1" in conds:
        cache = {}

        def fold_pca(fi, tr, f):
            cache[fi] = pca_fold(Y0[tr], Y0[f], KS_PCA)
            return None
        # 폴드마다 한 번만 SVD하고 k별로 따로 decoding
        pcs = {}
        for fi, f in enumerate(folds):
            tr = np.setdiff1d(np.arange(n), f)
            pcs[fi] = pca_fold(Y0[tr], Y0[f], KS_PCA)
            log(f"  PCA 폴드 {fi + 1} 완료")
        for k in KS_PCA:
            save(f"D1_k{k}", evaluate(f"D1_k{k}", lambda fi, tr, f, k=k: pcs[fi][k], S, n, folds, log))
    if "D2" in conds:
        coords = voxel_coords(a.roi_dir, allmask)
        for fw in FWHMS:
            W = smoothing_matrix(coords, fw)
            Ys = smooth_mean(reps, W)[:, keep]
            log(f"  스무딩 FWHM {fw}: 평균 이웃 수 {W.getnnz(1).mean():.1f}")
            save(f"D2_fwhm{fw:g}", evaluate(f"D2_fwhm{fw:g}", lambda fi, tr, f, Ys=Ys: (Ys[tr], Ys[f]), S, n, folds, log))
    if "D3" in conds:
        save("D3", evaluate("D3", lambda fi, tr, f: whiten_fold(reps_k[tr], Y0[tr], Y0[f]), S, n, folds, log))
    json.dump({"conds": list(results), "n_images": n, "n_voxels": int(keep.sum())}, open(o / "denoise_index.json", "w"), ensure_ascii=False)
    print("저장:", o)


if __name__ == "__main__":
    main()
