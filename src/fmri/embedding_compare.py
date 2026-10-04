"""임베딩 모델 비교(원본 캡션 임베딩 × 피험자 1 뇌 반응). 기준은 Notion 「임베딩 모델 비교 기준」(2026-10-04)에 사전 기록.

선택에는 피험자 1이 본 이미지 중 shared1000을 뺀 학습용 이미지만 쓴다(shared1000은 RQ 평가용으로 남긴다).
  encoding: 임베딩 → 복셀 반응 ridge, 이미지 단위 5겹 교차검증, 복셀별 alpha를 학습 폴드 안의 3겹으로 선택.
            복셀별 Pearson r(폴드별 열 평균을 뺀 뒤 합쳐 계산)의 영역 묶음별 중앙값. 주 점수 P = R2와 R3의 중앙값 평균.
  RSA     : 학습용 이미지에서 시드 고정으로 뽑은 1,000장, 뇌 RDM(1-Pearson)과 모델 RDM(1-코사인)의 Spearman. R2와 R3의 평균.
  부트스트랩: 이미지 단위 1,000회(다모델 쌍 비교는 같은 표본). 교체 규칙은 Notion 기록 그대로.
복셀은 학습용 이미지의 3회 반복 사이 평균 쌍별 상관이 0.1 이상인 것만 쓴다.

사용(Colab): python3 src/fmri/embedding_compare.py --fmri-dir .../roi_betas --emb-dir .../embeddings --out-dir .../compare
시험:        python3 src/fmri/embedding_compare.py --selftest
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ALPHAS = 10.0 ** np.arange(-1.0, 5.01, 0.5)  # 13개
MODELS = {"CLIP": "clip_vit_l14", "MPNet": "mpnet_base_v2", "Qwen3": "qwen3_emb_0p6b", "GTR-T5": "gtr_t5_base"}
GROUPS = ("R1", "R2", "R3", "EARLY")
MIN_RELIABILITY = 0.1
EFFECT_FLOOR = 0.01  # 교체 논의의 최소 차이(중앙값 r)


# ---------- ridge encoding ----------
def standardize(Xtr, *others):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    return [(Xtr - mu) / sd] + [(X - mu) / sd for X in others]


def choose_alpha(X, Y, rng, k=3):
    n = len(X)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    err = np.zeros((len(ALPHAS), Y.shape[1]), np.float64)
    for f in folds:
        tr = np.setdiff1d(perm, f)
        Xa, Xv = standardize(X[tr], X[f])
        ym = Y[tr].mean(0)
        U, s, Vt = np.linalg.svd(Xa, full_matrices=False)
        UtY = U.T @ (Y[tr] - ym)
        for ai, a in enumerate(ALPHAS):
            coef = Vt.T @ ((s / (s ** 2 + a))[:, None] * UtY)
            err[ai] += (((Y[f] - ym) - Xv @ coef) ** 2).sum(0)
    return err.argmin(0)  # 복셀별 alpha 인덱스


def fit_predict(Xtr, Ytr, Xte, ai):
    Xa, Xt = standardize(Xtr, Xte)
    ym = Ytr.mean(0)
    U, s, Vt = np.linalg.svd(Xa, full_matrices=False)
    UtY = U.T @ (Ytr - ym)
    pred = np.empty((len(Xte), Ytr.shape[1]), np.float32)
    for k, a in enumerate(ALPHAS):
        cols = np.where(ai == k)[0]
        if len(cols):
            coef = Vt.T @ ((s / (s ** 2 + a))[:, None] * UtY[:, cols])
            pred[:, cols] = Xt @ coef + ym[cols]
    return pred


def outer_folds(n, seed=0, k_outer=5):
    return np.array_split(np.random.default_rng(seed).permutation(n), k_outer)


def fold_center(M, folds):
    """폴드마다 열 평균을 뺀다. 교차검증 예측을 한꺼번에 합쳐 상관을 구할 때 생기는 음의 편향
    (학습 평균이 검증 평균과 반대로 움직이는 인공물)을 없앤다. 예측과 실제 모두에 같은 폴드로 적용한다."""
    M = M.copy()
    for f in folds:
        M[f] -= M[f].mean(0)
    return M


def oof_predict(X, Y, seed=0, k_outer=5):
    """같은 시드면 모든 모델이 같은 바깥 폴드를 쓴다."""
    n = len(X)
    outer = outer_folds(n, seed, k_outer)
    inner_rng = np.random.default_rng(seed + 1)
    oof = np.empty(Y.shape, np.float32)
    for f in outer:
        tr = np.setdiff1d(np.arange(n), f)
        oof[f] = fit_predict(X[tr], Y[tr], X[f], choose_alpha(X[tr], Y[tr], inner_rng))
    return oof


# ---------- 이미지 단위 부트스트랩(가중 Pearson) ----------
def boot_r(P, Y, W):
    """P, Y: n×V, W: B×n 가중치(합 1). 반환 B×V 복셀별 Pearson r."""
    mx, my = W @ P, W @ Y
    vx = W @ (P * P) - mx ** 2
    vy = W @ (Y * Y) - my ** 2
    cxy = W @ (P * Y) - mx * my
    return cxy / np.sqrt(np.maximum(vx, 1e-12) * np.maximum(vy, 1e-12))


def boot_weights(n, B, seed):
    rng = np.random.default_rng(seed)
    return (rng.multinomial(n, np.ones(n) / n, size=B) / n).astype(np.float32)


def group_medians(R, masks):
    return {g: np.median(R[:, m], axis=1) for g, m in masks.items()}  # 그룹 → B 길이 벡터


def encoding_scores(P, Y, masks, W, chunk=250):
    n = len(P)
    point = group_medians(boot_r(P, Y, np.full((1, n), 1.0 / n, np.float32)), masks)
    boots = {g: [] for g in masks}
    for i in range(0, len(W), chunk):
        for g, v in group_medians(boot_r(P, Y, W[i:i + chunk]), masks).items():
            boots[g].append(v)
    return {g: float(point[g][0]) for g in masks}, {g: np.concatenate(v) for g, v in boots.items()}


# ---------- RSA ----------
def rdm_pairs(Z, kind):
    if kind == "brain":
        Zc = Z - Z.mean(1, keepdims=True)
        Zc /= np.linalg.norm(Zc, axis=1, keepdims=True) + 1e-12
    else:
        Zc = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return 1.0 - Zc @ Zc.T


def rank_matrix(D):
    n = len(D)
    iu = np.triu_indices(n, 1)
    M = np.zeros((n, n))
    r = rankdata(D[iu])
    M[iu] = r
    return M + M.T


def weighted_pearson_pairs(Xm, Ym, C):
    """Xm, Ym: 대칭 순위 행렬(대각 0). C: B×n 이미지 복원추출 횟수. 중복 이미지끼리의 쌍은 제외한 쌍 가중 Pearson."""
    def S(M):
        return 0.5 * ((C @ M) * C).sum(1)
    one = np.ones_like(Xm) - np.eye(len(Xm))
    sw, sx, sy = S(one), S(Xm), S(Ym)
    sxx, syy, sxy = S(Xm * Xm), S(Ym * Ym), S(Xm * Ym)
    mx, my = sx / sw, sy / sw
    return (sxy / sw - mx * my) / np.sqrt(np.maximum(sxx / sw - mx ** 2, 1e-12) * np.maximum(syy / sw - my ** 2, 1e-12))


def rsa_scores(Ybrain, emb, masks, groups, B, seed):
    n = len(Ybrain)
    rng = np.random.default_rng(seed)
    C = rng.multinomial(n, np.ones(n) / n, size=B).astype(np.float64)
    ones = np.ones((1, n))
    Mrank = {name: rank_matrix(rdm_pairs(E, "model")) for name, E in emb.items()}
    out = {name: {} for name in emb}
    for g in groups:
        Bm = rank_matrix(rdm_pairs(Ybrain[:, masks[g]], "brain"))
        for name in emb:
            pt = float(weighted_pearson_pairs(Bm, Mrank[name], ones)[0])
            out[name][g] = (pt, weighted_pearson_pairs(Bm, Mrank[name], C))
    return out


# ---------- 입출력과 판정 ----------
def load_data(fmri_dir, emb_dir):
    fmri_dir, emb_dir = Path(fmri_dir), Path(emb_dir)
    tm = list(csv.DictReader(open(fmri_dir / "trial_map.csv", encoding="utf-8")))
    groups = np.load(fmri_dir / "roi_groups.npz")
    allmask = np.zeros(len(groups["R1"]), bool)
    for g in GROUPS:
        allmask |= groups[g]
    nvox = int(allmask.sum())
    Yz = np.empty((len(tm), nvox), np.float32)
    for s in range(1, 41):
        a = np.load(fmri_dir / f"session{s:02d}.npy").astype(np.float32)[:, allmask]
        Yz[(s - 1) * 750:s * 750] = (a - a.mean(0)) / (a.std(0) + 1e-6)  # 세션 안 복셀별 z-score
    trials = {}
    for r in tm:
        trials.setdefault(r["cocoId"], []).append(int(r["trial"]))
    shared = {r["cocoId"] for r in tm if r["shared1000"] == "1"}
    ids = json.load(open(emb_dir / "ids.json"))
    row = {c: i for i, c in enumerate(ids)}
    train = [c for c in trials if c not in shared and c in row and len(trials[c]) == 3]
    T = np.array([trials[c] for c in train])  # n×3
    # 신뢰도: 학습용 이미지의 반복 간 평균 쌍별 상관(복셀별)
    def pc(a, b):
        a, b = a - a.mean(0), b - b.mean(0)
        return (a * b).sum(0) / (np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0)) + 1e-12)
    rel = (pc(Yz[T[:, 0]], Yz[T[:, 1]]) + pc(Yz[T[:, 0]], Yz[T[:, 2]]) + pc(Yz[T[:, 1]], Yz[T[:, 2]])) / 3
    keep = rel >= MIN_RELIABILITY
    Y = Yz[T].mean(1)[:, keep]  # 반복 평균
    reps = Yz[T][:, :, keep]
    del Yz
    masks = {g: groups[g][allmask][keep] for g in GROUPS}
    emb = {}
    for name, stem in MODELS.items():
        E = np.load(emb_dir / f"{stem}_caption.npy").astype(np.float32)
        emb[name] = E[[row[c] for c in train]]
    extra = {}
    for label, f in (("object-only(CLIP)", "clip_vit_l14_object.npy"), ("CLIP-image", "clip_vit_l14_image.npy")):
        if (emb_dir / f).exists():
            extra[label] = np.load(emb_dir / f).astype(np.float32)[[row[c] for c in train]]
    return train, Y, reps, masks, emb, extra, int(keep.sum()), nvox


def run(Y, reps, masks, emb, extra, B=1000, n_rsa=1000, seed=0, log=print):
    n = len(Y)
    W = boot_weights(n, B, seed + 10)
    folds = outer_folds(n, seed)
    Yc = fold_center(Y, folds)
    res = {"n_images": n, "n_voxels": Y.shape[1], "groups": {g: int(m.sum()) for g, m in masks.items()}, "encoding": {}, "rsa": {}}
    boots = {}
    for name, X in {**emb, **extra}.items():
        t = time.time()
        P = fold_center(oof_predict(X, Y, seed), folds)
        pt, bt = encoding_scores(P, Yc, masks, W)
        res["encoding"][name] = pt
        boots[name] = bt
        log(f"[encoding] {name}: " + ", ".join(f"{g}={pt[g]:.4f}" for g in GROUPS) + f"  ({time.time() - t:.0f}s)")
        del P
    # 대조: 행을 섞은 CLIP
    perm = np.random.default_rng(seed + 99).permutation(n)
    P = fold_center(oof_predict(emb["CLIP"][perm], Y, seed), folds)
    pt, bt = encoding_scores(P, Yc, masks, W)
    res["shuffle_control"] = {g: {"r": pt[g], "ci": [float(np.percentile(bt[g], 2.5)), float(np.percentile(bt[g], 97.5))]} for g in GROUPS}
    log("[shuffle] " + ", ".join(f"{g}={pt[g]:.4f}" for g in GROUPS))
    # RSA
    rng = np.random.default_rng(seed + 5)
    sub = np.sort(rng.choice(n, min(n_rsa, n), replace=False))
    rsa = rsa_scores(Y[sub], {k: v[sub] for k, v in {**emb, **extra}.items()}, masks, GROUPS, B, seed + 6)
    for name in rsa:
        res["rsa"][name] = {g: rsa[name][g][0] for g in GROUPS}
        log(f"[rsa] {name}: " + ", ".join(f"{g}={rsa[name][g][0]:.4f}" for g in GROUPS))
    # 판정
    def primary(src):
        return (src["R2"] + src["R3"]) / 2
    Pb = {m: (boots[m]["R2"] + boots[m]["R3"]) / 2 for m in emb}
    res["primary"] = {m: {"P": primary(res["encoding"][m]), "ci": [float(np.percentile(Pb[m], 2.5)), float(np.percentile(Pb[m], 97.5))]} for m in emb}
    res["rsa_primary"] = {m: primary(res["rsa"][m]) for m in emb}
    res["decision"] = {}
    for m in emb:
        if m == "CLIP":
            continue
        d = Pb[m] - Pb["CLIP"]
        lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
        diff = res["primary"][m]["P"] - res["primary"]["CLIP"]["P"]
        c1 = bool(lo > 0 and diff >= EFFECT_FLOOR)
        c2 = bool(res["rsa_primary"][m] > res["rsa_primary"]["CLIP"])
        c3 = sum(res["encoding"][m][g] > res["encoding"]["CLIP"][g] for g in ("R1", "R2", "R3")) >= 2
        res["decision"][m] = {"diff_vs_CLIP": float(diff), "diff_ci": [lo, hi], "cond_i": c1, "cond_ii": c2, "cond_iii": bool(c3),
                              "replace_discussion": bool(c1 and c2 and c3)}
    res["verdict"] = "교체 논의: " + ", ".join(m for m, d in res["decision"].items() if d["replace_discussion"]) if any(
        d["replace_discussion"] for d in res["decision"].values()) else "CLIP 유지"
    return res


def selftest():
    rng = np.random.default_rng(0)
    n, d, V = 400, 24, 40
    sig = rng.standard_normal((n, d))
    W0 = rng.standard_normal((d, V)) * 0.5
    Y = (sig @ W0 + rng.standard_normal((n, V)) * 2.0).astype(np.float32)
    emb = {"CLIP": sig + rng.standard_normal((n, d)) * 0.5, "MPNet": sig + rng.standard_normal((n, d)) * 3.0,
           "Qwen3": rng.standard_normal((n, d)), "GTR-T5": rng.standard_normal((n, d))}
    emb = {k: v.astype(np.float32) for k, v in emb.items()}
    masks = {"R1": np.arange(V) < 30, "R2": np.arange(V) < 20, "R3": np.arange(V) >= 15, "EARLY": np.arange(V) < 10}
    res = run(Y, None, masks, emb, {}, B=60, n_rsa=120, seed=1)
    print(json.dumps({k: res[k] for k in ("primary", "rsa_primary", "shuffle_control", "verdict")}, indent=1, default=float))
    c, q = res["primary"]["CLIP"]["P"], res["primary"]["Qwen3"]["P"]
    s = res["shuffle_control"]["R2"]
    assert c > 0.1 and abs(q) < 0.1, (c, q)       # 신호 있는 모델은 양수, 무작위는 0 근처
    assert s["ci"][0] <= 0 <= s["ci"][1], s       # 섞은 대조는 0을 포함
    assert res["rsa_primary"]["CLIP"] > res["rsa_primary"]["Qwen3"]
    print("selftest 통과")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmri-dir")
    ap.add_argument("--emb-dir")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    train, Y, reps, masks, emb, extra, nkeep, nvox = load_data(a.fmri_dir, a.emb_dir)
    print(f"학습용 이미지 {len(train)}장, 복셀 {nkeep}/{nvox} (신뢰도 ≥ {MIN_RELIABILITY}), 영역 묶음 { {g: int(m.sum()) for g, m in masks.items()} }", flush=True)
    res = run(Y, reps, masks, emb, extra, B=a.boot, log=lambda s: print(s, flush=True))
    res["n_reliable_voxels"], res["n_voxels_total"] = nkeep, nvox
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "embedding_compare_result.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print("\n판정:", res["verdict"])
    print(json.dumps(res["decision"], ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
