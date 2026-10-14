"""E4: ROI 분할(A), 차원 × ROI(B), 가중 결합(C). 기준은 실행 전에 docs/roi_weight_criteria.md에 기록했다.
사용: python3 src/fmri/roi_weight.py --fmri-dir data/stage/fmri_subj01 --emb-dir data/embeddings_v2 --bundle-dir data/colab_bundle --out-dir data/results/roi_weight
빠른 시험: 위에 --fast (이미지 600장, 폴드 2개, 그룹 3개)
"""
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import axes_decode as ad  # noqa: E402
import denoise_decode as dd  # noqa: E402
import embedding_compare as ec  # noqa: E402
import text_decode_compare as td  # noqa: E402
from text_compare import build_sets, item_counts, load_features  # noqa: E402

import text_variants as tv  # noqa: E402

DIMS = tuple(tv.DIMS)  # V3 블록 순서
FINE = ("faces", "places", "bodies", "words", "S_ventral", "S_lateral", "S_parietal")
GROUPS = ("ALL", "R2", "R3") + FINE
TARGETS = ("orig_P1", "obj_P1", "ctx_P1")
FUSIONS = ("F1_dim_equal", "F2_dim_learned", "F3_roi_equal", "F4_dimroi_equal", "F5_dimroi_learned")


def auc_rank(S):
    """정답(대각)이 후보보다 높게 매겨진 비율의 평균(동점은 절반)."""
    d = np.diag(S)
    n = len(S)
    hi = (S > d[:, None] + 1e-6).sum(1)
    eq = (np.abs(S - d[:, None]) <= 1e-6).sum(1) - 1
    return float((1 - (hi + 0.5 * eq) / max(n - 1, 1)).mean())


def chance_c1(S_sub, rng, reps=5):
    """정답 짝을 무작위로 바꾼 뒤 같은 방식으로 계산한 top-1(라벨이 겹쳐서 생기는 우연 수준)."""
    out = []
    for _ in range(reps):
        p = rng.permutation(len(S_sub))
        c1, _ = td.credit(S_sub[:, p])
        out.append(float(c1.mean()))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmri-dir")
    ap.add_argument("--emb-dir")
    ap.add_argument("--bundle-dir")
    ap.add_argument("--out-dir", default="data/results/roi_weight")
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    cap = {r["cocoId"]: r for r in map(json.loads, gzip.open(Path(a.bundle_dir) / "captions.jsonl.gz", "rt"))}
    lab = {r["cocoId"]: r for r in map(json.loads, gzip.open(Path(a.bundle_dir) / "labels_pass0.jsonl.gz", "rt"))}
    ids_all = json.load(open(Path(a.emb_dir) / "ctx_ids.json"))
    train, reps, keep, allmask = dd.load_reps(a.fmri_dir, ids_all)
    grp = np.load(Path(a.fmri_dir) / "roi_groups.npz")
    gm = {"ALL": np.ones(int(keep.sum()), bool)}
    for g in ("R2", "R3") + FINE:
        gm[g] = grp[g][allmask][keep]
    Y0 = reps.mean(1)[:, keep]
    del reps
    row = {c: i for i, c in enumerate(ids_all)}
    F = load_features(a.emb_dir, "CLIP", ids_all, train)
    counts = item_counts(cap, lab, train)
    Sall = build_sets(F, counts, 0)
    T = {r: Sall[r].astype(np.float32) for r in TARGETS}
    blocks = F["ctx_V3"].astype(np.float32)
    dmask = np.load(Path(a.emb_dir) / "CLIP__ctx_V3.npz")["mask"][[row[c] for c in train]].astype(bool)
    n = len(train)
    groups = list(GROUPS)
    folds = ec.outer_folds(n, 0)
    if a.fast:
        n = 600
        Y0, blocks, dmask = Y0[:n], blocks[:n], dmask[:n]
        T = {r: v[:n] for r, v in T.items()}
        folds = ec.outer_folds(n, 0)[:2]
        groups = ["ALL", "faces", "places"]
    fine = [g for g in FINE if g in groups]
    print(f"E4: 이미지 {n}장, 그룹별 복셀 수 " + ", ".join(f"{g}={int(gm[g].sum())}" for g in groups) + f", 차원별 보유 이미지 수 {dmask.sum(0).tolist()}", flush=True)
    rng = np.random.default_rng(5)
    A = {g: {r: {"c1": np.full(n, np.nan, np.float32), "c5": np.full(n, np.nan, np.float32)} for r in TARGETS} for g in groups}
    B = {d: {g: {"c1": np.full(n, np.nan, np.float32), "c5": np.full(n, np.nan, np.float32)} for g in groups} for d in range(6)}
    CH = {d: [] for d in range(6)}
    Fz = {f: {"c1": np.full(n, np.nan, np.float32), "c5": np.full(n, np.nan, np.float32)} for f in ("F0_ctx_P1_all",) + FUSIONS}
    W = []
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for fi, f in enumerate(folds):
        tr = np.setdiff1d(np.arange(n), f)
        nf = len(f)
        simP1 = {}
        # 시험 A
        Ztr = np.concatenate([T[r][tr] for r in TARGETS], axis=1)
        for g in groups:
            Zh = ad.decode(Y0[tr][:, gm[g]], Ztr, Y0[f][:, gm[g]], seed=fi)
            c = 0
            for r in TARGETS:
                d = T[r].shape[1]
                S_ = ad.sim_matrix(Zh[:, c:c + d], T[r][f])
                c += d
                c1, c5 = td.credit(S_)
                A[g][r]["c1"][f], A[g][r]["c5"][f] = c1, c5
                if r == "ctx_P1":
                    simP1[g] = S_
        Fz["F0_ctx_P1_all"]["c1"][f], Fz["F0_ctx_P1_all"]["c5"][f] = A["ALL"]["ctx_P1"]["c1"][f], A["ALL"]["ctx_P1"]["c5"][f]
        print(f"  [폴드 {fi + 1}] 시험 A 완료 ({time.time() - t0:.0f}s)", flush=True)
        # 시험 B와 가중치
        S = {}
        w = np.zeros((6, len(groups)))
        for d in range(6):
            trd = tr[dmask[tr, d]]
            fd = f[dmask[f, d]]
            mloc = dmask[f, d]
            if len(trd) < 50 or len(fd) < 20:
                continue
            perm = rng.permutation(trd)
            h1, h2 = perm[:len(perm) // 2], perm[len(perm) // 2:]
            for gi, g in enumerate(groups):
                Xg = Y0[:, gm[g]]
                Zh = ad.decode(Xg[trd], blocks[trd, d], Xg[f], seed=fi)
                full = np.zeros((nf, nf), np.float32)
                full[:, mloc] = ad.sim_matrix(Zh, blocks[fd, d])
                S[(d, g)] = full
                sub = full[np.ix_(mloc, mloc)]
                c1, c5 = td.credit(sub)
                B[d][g]["c1"][fd], B[d][g]["c5"][fd] = c1, c5
                if g == "ALL":
                    CH[d].append(chance_c1(sub, rng))
                Zh2 = ad.decode(Xg[h1], blocks[h1, d], Xg[h2], seed=fi)
                w[d, gi] = max(auc_rank(ad.sim_matrix(Zh2, blocks[h2, d])) - 0.5, 0.0)
        W.append(w)
        print(f"  [폴드 {fi + 1}] 시험 B 완료 ({time.time() - t0:.0f}s)", flush=True)
        # 시험 C
        def fuse(keys, wts):
            tot = np.zeros((nf, nf), np.float32)
            for k, wt in zip(keys, wts):
                tot += wt * S[k]
            return tot

        gi_all = groups.index("ALL")
        kd = [(d, "ALL") for d in range(6) if (d, "ALL") in S]
        mats = {"F1_dim_equal": fuse(kd, [1.0] * len(kd)),
                "F2_dim_learned": fuse(kd, [w[d, gi_all] for d, _ in kd]),
                "F3_roi_equal": sum(simP1[g] for g in fine) / len(fine),
                "F4_dimroi_equal": fuse([(d, g) for d in range(6) for g in fine if (d, g) in S], [1.0] * sum(1 for d in range(6) for g in fine if (d, g) in S)),
                "F5_dimroi_learned": fuse([(d, g) for d in range(6) for g in fine if (d, g) in S], [w[d, groups.index(g)] for d in range(6) for g in fine if (d, g) in S])}
        for k, M in mats.items():
            c1, c5 = td.credit(M)
            Fz[k]["c1"][f], Fz[k]["c5"][f] = c1, c5
        print(f"  [폴드 {fi + 1}] 시험 C 완료 ({time.time() - t0:.0f}s)", flush=True)
        np.savez_compressed(out / "roi_weight_partial.npz", **{f"A__{g}__{r}__{k}": A[g][r][k] for g in groups for r in TARGETS for k in ("c1", "c5")},
                            **{f"B__{d}__{g}__{k}": B[d][g][k] for d in range(6) for g in groups for k in ("c1", "c5")},
                            **{f"F__{x}__{k}": Fz[x][k] for x in Fz for k in ("c1", "c5")}, W=np.array(W))
    meta = {"groups": groups, "n_voxels": {g: int(gm[g].sum()) for g in groups}, "dim_images": dmask.sum(0).tolist(), "n": n,
            "dims": DIMS, "chance_dim_c1": {d: float(np.mean(CH[d])) if CH[d] else None for d in range(6)}, "weights_mean": np.mean(W, axis=0).tolist()}
    np.savez_compressed(out / "roi_weight.npz", **{f"A__{g}__{r}__{k}": A[g][r][k] for g in groups for r in TARGETS for k in ("c1", "c5")},
                        **{f"B__{d}__{g}__{k}": B[d][g][k] for d in range(6) for g in groups for k in ("c1", "c5")},
                        **{f"F__{x}__{k}": Fz[x][k] for x in Fz for k in ("c1", "c5")}, W=np.array(W))
    json.dump(meta, open(out / "roi_weight_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("저장:", out, f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
