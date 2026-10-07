"""RQ1 보강 D: 하위공간과 맥락 정보가 사물(COCO 80범주)로 설명되는 정도, 그리고 사물 정보를 뺀 맥락의 k 곡선.

D1: 한 피험자의 RRR 상위 k방향 점수(전체 라벨 이미지에 투영)를 COCO 범주 유무(80차원 multi-hot)로 5폴드 회귀했을 때의 교차검증 R².
    방향별 R²와, 하위공간 전체 분산 중 범주로 설명되는 비율(분산 가중). 사물 이름 임베딩(obj_P1, 768차원)으로도 같은 회귀를 한다.
    범주가 거의 다 설명하면 이 방향은 사물·범주 축이다. 범주 위치를 섞은 대조(셔플)와 함께 보고한다.
D2: 맥락 임베딩에서 사물 정보로 선형 예측되는 부분을 5폴드 교차 예측으로 빼서(사물 이름 임베딩 + 80범주 multi-hot) 잔차 임베딩을
    {emb-dir}/{model}__ctxres_P1.npy로 저장한다(뇌 자료는 쓰지 않는다). 이어서 axes_k_curve.py --rep ctxres_P1 로 같은 k 곡선과
    영역별 분석을 돌린다. 잔차 임베딩이 뇌 반응을 사물 이름 없이도 설명하면(셔플 ≈ 0 대비) 사물 이름 너머 정보가 있다.
    선형 제거라 비선형 사물 정보는 남을 수 있다.

사용(Colab): python3 axes_object_check.py --fmri-dir .../roi_betas --emb-dir .../embeddings_v2 --bundle-dir .../colab_bundle --model MPNet --out-dir .../axes_object
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

import embedding_compare as ec
from axes_k_curve import save_json
from axes_shared import fit_dirs
from text_compare import load_brain

LAM = 100.0


def multihot(ids, cap):
    names = sorted({n for c in ids for n in (cap[c]["object_only"] or "").split(", ") if n})
    ix = {n: i for i, n in enumerate(names)}
    M = np.zeros((len(ids), len(names)), np.float32)
    for r, c in enumerate(ids):
        for n in (cap[c]["object_only"] or "").split(", "):
            if n:
                M[r, ix[n]] = 1
    return M, names


def cv_ridge_pred(P, T, lam=LAM, k=5, seed=0):
    """P(n×p)로 T(n×q)를 k폴드 교차 예측(절편 포함, P는 학습 폴드 기준 표준화 없이 열 평균만 뺌)."""
    n = len(P)
    out = np.empty(T.shape, np.float32)
    for f in ec.outer_folds(n, seed, k):
        tr = np.setdiff1d(np.arange(n), f)
        mu, ym = P[tr].mean(0), T[tr].mean(0)
        A = P[tr] - mu
        W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (T[tr] - ym))
        out[f] = (P[f] - mu) @ W + ym
    return out


def r2_cols(T, Th):
    return 1 - ((T - Th) ** 2).sum(0) / ((T - T.mean(0)) ** 2).sum(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmri-dir")
    ap.add_argument("--emb-dir")
    ap.add_argument("--bundle-dir")
    ap.add_argument("--model", default="MPNet")
    ap.add_argument("--rep", default="ctx_P1")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--tag", default="subj01")
    a = ap.parse_args()
    emb = Path(a.emb_dir)
    ids = json.load(open(emb / "ctx_ids.json"))
    cap = {r["cocoId"]: r for r in map(json.loads, gzip.open(Path(a.bundle_dir) / "captions.jsonl.gz", "rt"))}
    E = np.load(emb / f"{a.model}__{a.rep}.npy").astype(np.float32)
    O = np.load(emb / f"{a.model}__obj_P1.npy").astype(np.float32)
    M, names = multihot(ids, cap)
    S = (E - E.mean(0)) / (E.std(0) + 1e-6)
    log = lambda s: print(s, flush=True)
    log(f"이미지 {len(ids)}장, 사물 범주 {len(names)}개")

    # D1
    row = {c: i for i, c in enumerate(ids)}
    train, Ytr, *_ , masks, _nk, _nv = load_brain(a.fmri_dir, ids)
    keep = masks["R2"] | masks["R3"]
    X = S[[row[c] for c in train]]
    D = fit_dirs(X, Ytr[:, keep])["RRR"][:, :a.k]
    Z = S @ D
    rng = np.random.default_rng(0)
    Ms = M[rng.permutation(len(M))]
    res = {"tag": a.tag, "model": a.model, "rep": a.rep, "k": a.k, "n_images": len(ids), "n_categories": len(names)}
    out = {}
    for name, P in (("coco80_multihot", M), ("obj_embedding", O), ("coco80_shuffled", Ms)):
        r2 = r2_cols(Z, cv_ridge_pred(P, Z))
        var = Z.var(0)
        out[name] = {"R2_per_direction": [float(x) for x in r2], "R2_variance_weighted": float((r2 * var).sum() / var.sum())}
        log(f"D1 {name:16s} R² 방향별 " + " ".join(f"{x:.2f}" for x in r2) + f" | 분산 가중 {out[name]['R2_variance_weighted']:.2f}")
    res["D1"] = out

    # D2
    T = E
    pred = cv_ridge_pred(np.concatenate([O, M], axis=1), T)
    R = (T - pred).astype(np.float32)
    frac_removed = float(1 - R.var(0).sum() / T.var(0).sum())
    log(f"D2 맥락 임베딩 분산 중 사물 정보로 선형 예측되는 비율 {frac_removed:.2f}")
    try:
        np.save(emb / f"{a.model}__ctxres_P1.npy", R.astype(np.float16))
    except OSError as e:  # Drive가 끊겼을 때
        fb = Path("/content/out_fallback"); fb.mkdir(parents=True, exist_ok=True)
        np.save(fb / f"{a.model}__ctxres_P1.npy", R.astype(np.float16))
        print(f"!! {emb}에 저장 실패({e}); /content/out_fallback에 저장했다. 이 파일을 embeddings_v2로 복사해야 k 곡선 셀이 읽는다.", flush=True)
    res["D2"] = {"fraction_ctx_variance_predicted_by_objects": frac_removed, "saved": f"{a.model}__ctxres_P1.npy"}
    # 대조: 맥락 임베딩 대비 잔차의 코사인(평균)
    cos = float(np.mean((E * R).sum(1) / (np.linalg.norm(E, axis=1) * np.linalg.norm(R, axis=1) + 1e-9)))
    res["D2"]["mean_cosine_ctx_vs_residual"] = cos
    log(f"   맥락과 잔차 임베딩의 평균 코사인 {cos:.2f}")
    save_json(res, Path(a.out_dir) / f"axes_object_{a.tag}_{a.model}_{a.rep}.json")


if __name__ == "__main__":
    main()
