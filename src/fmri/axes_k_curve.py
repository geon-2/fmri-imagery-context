"""RQ1 ② 저차원 맥락 축: k 곡선(RRR / PLS-SVD / PCA / 무작위), ΔK, 학습곡선, 하위공간 안정성, shared1000 확인.
기준은 Notion 「RQ1 판정 설계 초안」과 2026-10-05 확정 사항(k 격자, 한 표준오차 규칙, 부트스트랩 1,000회, 재현은 2명 이상).

방법(모두 같은 틀): 학습 폴드에서 입력(임베딩) 공간의 방향 k개를 고르고, 그 k차원에서 OLS로 복셀 반응을 다시 적합한다.
  full: 전체 차원 ridge(복셀별 alpha, embedding_compare와 같음)
  RRR : ridge 계수 B의 적합값 XB를 SVD해 얻은 출력 공간 상위 k방향 V_k로 입력 방향 B·V_k를 만든다(뇌에 맞춘 축).
  PLS : 교차공분산 XᵀY의 좌특이벡터(PLS-SVD, 비순차 PLS)
  PCA : 임베딩 자체의 분산 상위 k방향
  무작위: 무작위 직교 k방향(3회 평균)
점수 P: R2·R3 영역 복셀 median r의 평균(폴드별 열 평균 제거 후). 적합은 R2∪R3 복셀만 쓴다.
"유지" k* = 전체 차원 성능에서 한 표준오차(5폴드 P의 SE)를 뺀 값 이상이 되는 가장 작은 k. ΔK = k*(PCA) − k*(RRR).
k*에 도달하지 못하면 임베딩 차원 d로 센다. 부트스트랩의 k*는 해당 표본의 전체 차원 P − (고정 SE)를 기준으로 다시 정한다.

사용(Colab):
  python3 axes_k_curve.py --fmri-dir .../roi_betas --emb-dir .../embeddings_v2 --model MPNet --out-dir .../axes
  (--learning-curve, --stab-boot 100, --confirm 은 선택)
시험: python3 axes_k_curve.py --selftest
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import embedding_compare as ec

KS = [1, 2, 3, 5, 8, 12, 20, 32, 50, 80, 128]


def save_json(obj, path, **kw):
    """결과 저장. Drive가 끊겨 쓰기에 실패하면 계산 결과를 잃지 않도록 /content/out_fallback에 대신 저장하고 알린다."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(obj, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float, **kw)
        print("저장:", path, flush=True)
    except OSError as e:
        fb = Path("/content/out_fallback") / path.name
        fb.parent.mkdir(parents=True, exist_ok=True)
        json.dump(obj, open(fb, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float, **kw)
        print(f"!! {path} 저장 실패({e}). 대신 {fb}에 저장했다. Drive를 다시 마운트한 뒤 복사할 것.", flush=True)
METHODS = ("RRR", "PLS", "PCA", "RAND")


# ---------- 방향 구하기와 예측 ----------
def ridge_dirs(Xa, Yc, ai):
    """학습 폴드: 복셀별 alpha ridge 계수 B와 방법별 입력 방향(d×d, 중요도 순)."""
    U, s, Vt = np.linalg.svd(Xa, full_matrices=False)
    UtY = U.T @ Yc
    B = np.empty((Xa.shape[1], Yc.shape[1]), np.float32)
    for j, a in enumerate(ec.ALPHAS):
        cols = np.where(ai == j)[0]
        if len(cols):
            B[:, cols] = Vt.T @ ((s / (s ** 2 + a))[:, None] * UtY[:, cols])
    Q = s[:, None] * (Vt @ B)                    # XaB = U·Q
    Vh = np.linalg.svd(Q, full_matrices=False)[2]
    Uc = np.linalg.svd(Xa.T @ Yc, full_matrices=False)[0]
    return B, {"RRR": B @ Vh.T, "PCA": Vt.T, "PLS": Uc}


def predict_dir(Xa, Xt, Yc, ym, D, k):
    d = D[:, :k]
    coef = np.linalg.lstsq(Xa @ d, Yc, rcond=None)[0]
    return ((Xt @ d) @ coef + ym).astype(np.float32)


def prep(Xtr, Ytr, Xte, rng):
    Xa, Xt = ec.standardize(Xtr, Xte)
    ym = Ytr.mean(0)
    Yc = Ytr - ym
    B, D = ridge_dirs(Xa, Yc, ec.choose_alpha(Xtr, Ytr, rng))
    return {"Xa": Xa, "Xt": Xt, "ym": ym, "Yc": Yc, "B": B, "D": D}


# ---------- 점수(점추정, 폴드별, 부트스트랩) ----------
class Scorer:
    def __init__(self, Y, folds, masks, W):
        self.folds, self.W = folds, W
        self.m = [masks["R2"], masks["R3"]]
        self.Yc = ec.fold_center(Y, folds)
        if W is not None:
            self.mY = W @ self.Yc
            self.vY = W @ (self.Yc * self.Yc) - self.mY ** 2

    def _P(self, P, Y):
        P, Y = P - P.mean(0), Y - Y.mean(0)
        r = (P * Y).sum(0) / (np.sqrt((P ** 2).sum(0) * (Y ** 2).sum(0)) + 1e-12)
        return float(np.mean([np.median(r[m]) for m in self.m]))

    def score(self, oof):
        Pc = ec.fold_center(oof, self.folds)
        out = {"P": self._P(Pc, self.Yc), "fold_P": [self._P(Pc[f], self.Yc[f]) for f in self.folds]}
        if self.W is not None:
            res = []
            for i in range(0, len(self.W), 250):
                w = self.W[i:i + 250]
                mP = w @ Pc
                vP = w @ (Pc * Pc) - mP ** 2
                c = w @ (Pc * self.Yc) - mP * self.mY[i:i + 250]
                r = c / np.sqrt(np.maximum(vP, 1e-12) * np.maximum(self.vY[i:i + 250], 1e-12))
                res.append(np.mean([np.median(r[:, m], axis=1) for m in self.m], axis=0))
            out["boot"] = np.concatenate(res)
        return out


def kstar(Pk, tau, ks, d):
    hit = [k for k, p in zip(ks, Pk) if p >= tau]
    return hit[0] if hit else d


def run(X, Y, masks, ks=KS, B=1000, seed=0, n_rand=3, log=print):
    n, d = X.shape
    ks = [k for k in ks if k <= d]
    folds = ec.outer_folds(n, seed)
    W = ec.boot_weights(n, B, seed + 7) if B else None
    sc = Scorer(Y, folds, masks, W)
    rng = np.random.default_rng(seed + 1)
    fo = []
    for f in folds:
        tr = np.setdiff1d(np.arange(n), f)
        fo.append((f, prep(X[tr], Y[tr], X[f], rng), tr))
    log(f"폴드 준비 완료 ({n}장, 차원 {d}, 복셀 {Y.shape[1]})")
    rands = [np.linalg.qr(np.random.default_rng(seed + 100 + j).standard_normal((d, max(ks))))[0] for j in range(n_rand)]

    def curve(method, j=0):
        out = []
        for k in ks:
            oof = np.empty(Y.shape, np.float32)
            for f, p, _ in fo:
                D = rands[j] if method == "RAND" else p["D"][method]
                oof[f] = predict_dir(p["Xa"], p["Xt"], p["Yc"], p["ym"], D, k)
            out.append(sc.score(oof))
        return out

    oof = np.empty(Y.shape, np.float32)
    for f, p, _ in fo:
        oof[f] = p["Xt"] @ p["B"] + p["ym"]
    full = sc.score(oof)
    log(f"full P={full['P']:.4f}")
    curves = {}
    for m in METHODS:
        t0 = time.time()
        if m == "RAND":
            cs = [curve(m, j) for j in range(n_rand)]
            cur = [{"P": float(np.mean([c[i]["P"] for c in cs])),
                    "fold_P": list(np.mean([c[i]["fold_P"] for c in cs], axis=0)),
                    **({"boot": np.mean([c[i]["boot"] for c in cs], axis=0)} if W is not None else {})} for i in range(len(ks))]
        else:
            cur = curve(m)
        curves[m] = cur
        log(f"{m:5s} " + " ".join(f"k{k}={c['P']:.3f}" for k, c in zip(ks, cur)) + f"  ({time.time() - t0:.0f}s)")

    se = float(np.std(full["fold_P"], ddof=1) / np.sqrt(len(folds)))
    tau = full["P"] - se
    kst = {m: kstar([c["P"] for c in curves[m]], tau, ks, d) for m in METHODS}
    reached = {m: any(c["P"] >= tau for c in curves[m]) for m in METHODS}
    res = {"n_images": n, "n_voxels": int(Y.shape[1]), "dim": d, "ks": ks, "full": {"P": full["P"], "se": se},
           "tau": tau, "kstar": kst, "kstar_reached_in_grid": reached, "deltaK": kst["PCA"] - kst["RRR"], "curves": {}}
    if not reached["PCA"]:
        # PCA·PLS·무작위는 축소 공간에서 OLS로 다시 적합하므로 ridge 전체 차원 성능에 격자 안에서 도달하지 못할 수 있다.
        # 그때 k*=d는 도달 실패를 뜻하는 상한이라 ΔK는 하한(격자 최대 k − k*(RRR) 이상)으로만 읽는다. 같은 k에서의 RRR−PCA 차이(rrr_minus_pca)를 함께 본다.
        res["deltaK_note"] = f"PCA는 격자(최대 k={max(ks)}) 안에서 기준 {tau:.4f}에 도달하지 못함: ΔK는 하한 {max(ks) - kst['RRR']}"
    for m in METHODS:
        res["curves"][m] = [{"k": k, "P": c["P"], "se": float(np.std(c["fold_P"], ddof=1) / np.sqrt(len(folds))),
                             **({"ci": [float(np.percentile(c["boot"], 2.5)), float(np.percentile(c["boot"], 97.5))]} if W is not None else {})}
                            for k, c in zip(ks, curves[m])]
    if W is not None:
        res["full"]["ci"] = [float(np.percentile(full["boot"], 2.5)), float(np.percentile(full["boot"], 97.5))]
        kb = {m: np.array([kstar([curves[m][i]["boot"][b] for i in range(len(ks))], full["boot"][b] - se, ks, d) for b in range(B)]) for m in METHODS}
        dk = kb["PCA"] - kb["RRR"]
        res["kstar_boot"] = {m: [float(np.percentile(v, 2.5)), float(np.median(v)), float(np.percentile(v, 97.5))] for m, v in kb.items()}
        res["deltaK_ci"] = [float(np.percentile(dk, 2.5)), float(np.percentile(dk, 97.5))]
        # RRR이 같은 k의 PCA보다 높은가(k별 차이의 부트스트랩 구간)
        res["rrr_minus_pca"] = [{"k": k, "diff": curves["RRR"][i]["P"] - curves["PCA"][i]["P"],
                                 "ci": [float(np.percentile(curves["RRR"][i]["boot"] - curves["PCA"][i]["boot"], 2.5)),
                                        float(np.percentile(curves["RRR"][i]["boot"] - curves["PCA"][i]["boot"], 97.5))]} for i, k in enumerate(ks)]
    log(f"k*: {kst}  ΔK={res['deltaK']}" + (f"  (95% 구간 {res['deltaK_ci']})" if W is not None else ""))
    return res


# ---------- 학습곡선 ----------
def learning_curve(X, Y, masks, sizes, ks=KS, seed=0, log=print):
    """이미지를 늘려 가며 full P와 k*(RRR, PCA)가 포화하는지 본다(부트스트랩 없음, 같은 순서의 부분집합)."""
    order = np.random.default_rng(seed + 11).permutation(len(X))
    rows = []
    for sz in sizes:
        idx = np.sort(order[:min(sz, len(X))])
        r = run(X[idx], Y[idx], masks, ks=ks, B=0, seed=seed, n_rand=1, log=lambda s: None)
        rows.append({"n": int(len(idx)), "full_P": r["full"]["P"], "kstar_RRR": r["kstar"]["RRR"], "kstar_PCA": r["kstar"]["PCA"],
                     "P_RRR_at_kstar": [c["P"] for c in r["curves"]["RRR"] if c["k"] == r["kstar"]["RRR"]][:1]})
        log(f"학습곡선 n={rows[-1]['n']}  full P={rows[-1]['full_P']:.4f}  k*(RRR)={rows[-1]['kstar_RRR']}  k*(PCA)={rows[-1]['kstar_PCA']}")
    return rows


# ---------- 하위공간 안정성 ----------
def overlap(Q1, Q2):
    """두 부분공간(열이 직교기저)의 겹침 = 평균 cos² (주각 기준, 1이면 동일). 무작위 기대값은 k/d."""
    return float(np.linalg.norm(Q1.T @ Q2) ** 2 / Q1.shape[1])


def subspace_stability(X, Y, ks, B=100, seed=0, log=print):
    """전체 표본의 RRR 하위공간과 부트스트랩 재표본의 RRR 하위공간의 겹침, 그리고 RRR과 PCA 하위공간의 겹침."""
    n, d = X.shape
    Xa = ec.standardize(X)[0]
    ym = Y.mean(0)
    Yc = Y - ym
    ai = ec.choose_alpha(X, Y, np.random.default_rng(seed))
    _, D0 = ridge_dirs(Xa, Yc, ai)
    rng = np.random.default_rng(seed + 5)
    res = {k: [] for k in ks}
    for b in range(B):
        i = rng.integers(0, n, n)
        _, Db = ridge_dirs(Xa[i], Yc[i], ai)
        for k in ks:
            res[k].append(overlap(np.linalg.qr(D0["RRR"][:, :k])[0], np.linalg.qr(Db["RRR"][:, :k])[0]))
        if (b + 1) % 10 == 0:
            log(f"안정성 {b + 1}/{B}")
    out = []
    for k in ks:
        out.append({"k": k, "overlap_mean": float(np.mean(res[k])), "ci": [float(np.percentile(res[k], 2.5)), float(np.percentile(res[k], 97.5))],
                    "chance": k / d, "rrr_vs_pca": overlap(np.linalg.qr(D0["RRR"][:, :k])[0], D0["PCA"][:, :k])})
    return out


# ---------- shared1000 확인(탐색에서 정한 k*를 그대로 쓴다) ----------
def confirm(Xtr, Ytr, Xte, Yte, masks, kstars, B=1000, seed=0, log=print):
    """탐색 표본 전체로 방향·계수를 학습하고 shared1000에서만 평가한다. 방법별로 탐색의 k*와 full을 비교한다."""
    p = prep(Xtr, Ytr, Xte, np.random.default_rng(seed + 1))
    folds = [np.arange(len(Xte))]
    W = ec.boot_weights(len(Xte), B, seed + 7)
    sc = Scorer(Yte, folds, masks, W)
    rng = np.random.default_rng(seed + 100)
    Qr = np.linalg.qr(rng.standard_normal((Xtr.shape[1], max(kstars.values()))))[0]
    out = {"full": sc.score((p["Xt"] @ p["B"] + p["ym"]).astype(np.float32))}
    for m, k in kstars.items():
        D = Qr if m == "RAND" else p["D"][m]
        out[m] = sc.score(predict_dir(p["Xa"], p["Xt"], p["Yc"], p["ym"], D, k))
    res = {m: {"P": v["P"], "ci": [float(np.percentile(v["boot"], 2.5)), float(np.percentile(v["boot"], 97.5))]} for m, v in out.items()}
    for a, b in (("RRR", "PCA"), ("RRR", "RAND"), ("RRR", "full")):
        if a in out and b in out:
            dd = out[a]["boot"] - out[b]["boot"]
            res[f"{a}_minus_{b}"] = {"diff": out[a]["P"] - out[b]["P"], "ci": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]}
    log(json.dumps(res, ensure_ascii=False, default=float))
    return res


# ---------- 시험 ----------
def selftest():
    rng = np.random.default_rng(0)
    n, d, V = 900, 24, 40
    L = rng.standard_normal((n, d)) * np.linspace(3.0, 0.3, d)          # 분산이 큰 방향부터
    A = np.linalg.qr(rng.standard_normal((d, d)))[0]
    X = (L @ A).astype(np.float32)
    Y = (L[:, -3:] @ (rng.standard_normal((3, V)) * 4.0) + rng.standard_normal((n, V)) * 2.0).astype(np.float32)  # 신호는 분산 가장 작은 3방향
    masks = {"R1": np.ones(V, bool), "R2": np.arange(V) < 30, "R3": np.arange(V) >= 10, "EARLY": np.arange(V) < 5}
    res = run(X, Y, masks, ks=[1, 2, 3, 5, 8, 12], B=30, seed=1, n_rand=2, log=lambda s: None)
    c = {m: {x["k"]: x["P"] for x in res["curves"][m]} for m in METHODS}
    print("k=3  RRR %.3f PLS %.3f PCA %.3f RAND %.3f | full %.3f | k* %s ΔK %s" % (c["RRR"][3], c["PLS"][3], c["PCA"][3], c["RAND"][3], res["full"]["P"], res["kstar"], res["deltaK"]))
    assert c["RRR"][3] > c["PCA"][3] + 0.05, "신호가 작은 분산 방향에 있으면 RRR이 PCA보다 높아야 한다"
    assert c["RRR"][3] > c["RAND"][3] + 0.05
    assert res["kstar"]["RRR"] <= 5 and res["deltaK"] > 0 and res["deltaK_ci"][0] > 0
    ov = subspace_stability(X, Y, [3], B=10, seed=1, log=lambda s: None)[0]
    assert ov["overlap_mean"] > 0.7 and ov["overlap_mean"] > 3 * ov["chance"], ov
    # 신호가 없으면(순수 잡음) 곡선이 모두 0 근처이고 ΔK 주장이 서지 않는다
    Yn = rng.standard_normal((n, V)).astype(np.float32)
    rn = run(X, Yn, masks, ks=[1, 3, 5], B=0, seed=1, n_rand=1, log=lambda s: None)
    assert abs(rn["full"]["P"]) < 0.06, rn["full"]
    print("selftest 통과")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmri-dir")
    ap.add_argument("--emb-dir")
    ap.add_argument("--model", default="MPNet")
    ap.add_argument("--rep", default="ctx_P1", help="ctx_P1(주) 또는 ctx_V2(보조)")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--tag", default="subj01")
    ap.add_argument("--roi", default=None, help="한 영역만으로 적합·평가(예: places, faces, bodies, words, S_early, S_ventral, EARLY). 없으면 R2∪R3")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--learning-curve", action="store_true")
    ap.add_argument("--stab-boot", type=int, default=0, help="하위공간 안정성 부트스트랩 횟수(0이면 건너뜀)")
    ap.add_argument("--confirm", action="store_true", help="shared1000 확인(탐색 결과 json의 k*를 읽는다). 탐색을 먼저 끝낸 뒤에만")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    from text_compare import load_brain
    emb = Path(a.emb_dir)
    ids_all = json.load(open(emb / "ctx_ids.json"))
    train, Ytr, test, Yte, masks, nkeep, nvox = load_brain(a.fmri_dir, ids_all, extra=(a.roi,) if a.roi else ())
    keep = masks[a.roi] if a.roi else masks["R2"] | masks["R3"]
    Ytr, Yte, masks = Ytr[:, keep], Yte[:, keep], {g: m[keep] for g, m in masks.items()}
    if a.roi:
        masks = {"R2": np.ones(int(keep.sum()), bool), "R3": np.ones(int(keep.sum()), bool)}  # 점수는 이 영역 복셀의 median r
    row = {c: i for i, c in enumerate(ids_all)}
    E = np.load(emb / f"{a.model}__{a.rep}.npy").astype(np.float32)
    Xtr, Xte = E[[row[c] for c in train]], E[[row[c] for c in test]]
    print(f"{a.tag}: 탐색 {len(train)}장, shared1000 {len(test)}장, 복셀({a.roi or "R2∪R3"}) {int(keep.sum())}, 차원 {Xtr.shape[1]}", flush=True)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = out / f"axes_{a.tag}_{a.model}_{a.rep}{'_' + a.roi if a.roi else ''}"
    log = lambda s: print(s, flush=True)
    if a.confirm:
        ex = json.load(open(f"{name}.json"))
        # 모든 방법을 RRR의 k*에서 비교한다(PCA·PLS의 k*는 격자 안에서 전체 차원 ridge에 도달하지 못해 차원 d로 센 값이라 비교에 쓰지 않는다)
        ks = {m: int(ex["kstar"]["RRR"]) for m in METHODS}
        res = confirm(Xtr, Ytr, Xte, Yte, masks, ks, B=a.boot, log=log)
        save_json({"kstar_from_exploration": ks, **res}, f"{name}_confirm.json")
        return
    res = run(Xtr, Ytr, masks, B=a.boot, log=log)
    if a.learning_curve:
        sizes = [1000, 2000, 3000, len(Xtr)]
        res["learning_curve"] = learning_curve(Xtr, Ytr, masks, sizes, log=log)
    if a.stab_boot:
        res["stability"] = subspace_stability(Xtr, Ytr, [k for k in KS if k <= 50], B=a.stab_boot, log=log)
    save_json(res, f"{name}.json")


if __name__ == "__main__":
    main()
