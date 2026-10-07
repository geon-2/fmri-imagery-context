"""RQ1 보강: brain decoding. 복셀 → 축 좌표를 해독할 수 있는가. 지표와 기준은 계산 전에 Notion에 기록했다(2026-10-07).

축 좌표 = 맥락 임베딩(공통 표준화)을 방향 D의 앞 k개에 투영한 값. 방향은 학습 폴드의 뇌·텍스트 공변으로 구한 RRR,
텍스트 분산 방향 PCA, 무작위, 기준으로 768차원 전체(full). decoder는 R2∪R3 복셀 → 좌표 ridge(학습 폴드에서만 적합).
RRR·PCA·무작위의 방향은 중요도 순이라 k=5,12,20 목표는 처음 20개 좌표의 앞부분이다(한 번의 다출력 ridge로 모두 해독).
지표: 쌍 비교 식별 정확도(우연 50%), 1,000지선다 top-1/top-5, 처음 12축의 평균 상관.
  탐색: 학습 이미지 5폴드(폴드마다 방향을 새로 구함). 확인: shared1000, 학습 표본 전체로 적합. 부트스트랩은 평가 이미지 단위.
사용(로컬): python3 axes_decode.py --fmri-dir .../stage/fmri_subj01 --emb-dir .../embeddings_v2 --model MPNet --out-dir .../results/axes_decode --tag subj01
시험:       python3 axes_decode.py --selftest
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import embedding_compare as ec
from axes_k_curve import save_json
from axes_shared import fit_dirs

KS = (5, 12, 20)
KMAX = 20
KMAIN = 12  # 판정 k
METHODS = ("RRR", "PCA", "RAND")
DEC_ALPHAS = 10.0 ** np.arange(2.0, 7.01, 0.5)  # 복셀이 많아 encoding보다 큰 정규화


def targets(S_ids, dirs, rand):
    """방법별 좌표(앞 KMAX개)와 full을 열로 붙인다. 반환: Z(n × (3*KMAX + d)), 열 구간 사전."""
    Zs, cols, c = [], {}, 0
    for m in METHODS:
        D = rand if m == "RAND" else dirs[m]
        Zs.append(S_ids @ D[:, :KMAX])
        cols[m] = slice(c, c + KMAX)
        c += KMAX
    Zs.append(S_ids)
    cols["full"] = slice(c, c + S_ids.shape[1])
    return np.concatenate(Zs, axis=1).astype(np.float32), cols


def standardize_axes(Zhat, Ztrue):
    """평가 표본의 평균을 빼고 실제 좌표의 표준편차로 나눈다(두 쪽에 같은 값)."""
    mu, sd = Ztrue.mean(0), Ztrue.std(0) + 1e-9
    return (Zhat - mu) / sd, (Ztrue - mu) / sd


def sim_matrix(Zhat, Ztrue):
    a, b = standardize_axes(Zhat, Ztrue)
    a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-9
    b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-9
    return a @ b.T


def pair_margin(Sm):
    """쌍 (i,j)가 맞으면 양수: sim(i,i)+sim(j,j) − sim(i,j) − sim(j,i)."""
    d = np.diag(Sm)
    return d[:, None] + d[None, :] - Sm - Sm.T


def pair_acc(M, idx=None):
    if idx is not None:
        M = M[np.ix_(idx, idx)]
    n = len(M)
    iu = np.triu_indices(n, 1)
    v = M[iu]
    # 부트스트랩에서 같은 이미지가 중복되면 마진이 0이므로 제외
    v = v[v != 0]
    return float((v > 0).mean())


def topk(Sm, k):
    rank = (Sm > np.diag(Sm)[:, None]).sum(1)
    return float((rank < k).mean())


def axis_corr(Zhat, Ztrue, k=12):
    a, b = Zhat[:, :k] - Zhat[:, :k].mean(0), Ztrue[:, :k] - Ztrue[:, :k].mean(0)
    r = (a * b).sum(0) / (np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0)) + 1e-12)
    return float(r.mean())


def evaluate(Zhat, Ztrue, cols, boot=0, seed=0):
    """방법·k별 지표. boot>0이면 쌍 비교 정확도의 이미지 단위 부트스트랩도 낸다."""
    out, Ms = {}, {}
    n = len(Ztrue)
    for m, sl in cols.items():
        for k in (KS if m != "full" else (None,)):
            sub = slice(sl.start, sl.start + k) if k else sl
            Sm = sim_matrix(Zhat[:, sub], Ztrue[:, sub])
            M = pair_margin(Sm)
            key = f"{m}_k{k}" if k else "full"
            out[key] = {"pair_acc": pair_acc(M), "top1": topk(Sm, 1), "top5": topk(Sm, 5)}
            if m != "full":
                out[key]["axis_corr12"] = axis_corr(Zhat[:, sub], Ztrue[:, sub], min(12, k))
            Ms[key] = M
    if boot:
        rng = np.random.default_rng(seed)
        idxs = [rng.integers(0, n, n) for _ in range(boot)]
        accs = {k: np.array([pair_acc(M, i) for i in idxs]) for k, M in Ms.items()}
        for k, v in accs.items():
            out[k]["pair_acc_ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        diffs = {}
        for k in KS:
            for ref in ("PCA", "RAND"):
                d = accs[f"RRR_k{k}"] - accs[f"{ref}_k{k}"]
                diffs[f"RRR_minus_{ref}_k{k}"] = {"diff": out[f"RRR_k{k}"]["pair_acc"] - out[f"{ref}_k{k}"]["pair_acc"],
                                                  "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
        d = accs[f"RRR_k{KMAIN}"] - accs["full"]
        diffs["RRR_main_minus_full"] = {"diff": out[f"RRR_k{KMAIN}"]["pair_acc"] - out["full"]["pair_acc"], "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
        out["_diffs"] = diffs
    return out


def decode(Xv_tr, Z_tr, Xv_te, seed=0):
    old = ec.ALPHAS
    ec.ALPHAS = DEC_ALPHAS
    try:
        ai = ec.choose_alpha(Xv_tr, Z_tr, np.random.default_rng(seed + 3))
        return ec.fit_predict(Xv_tr, Z_tr, Xv_te, ai)
    finally:
        ec.ALPHAS = old


def run(S_tr, Y_tr, S_te, Y_te, B=1000, seed=0, log=print, n_folds=5):
    """S_*: 공통 표준화 임베딩(학습/확인), Y_*: 복셀. 탐색(교차검증)과 확인을 모두 낸다."""
    d = S_tr.shape[1]
    rand = np.linalg.qr(np.random.default_rng(seed + 100).standard_normal((d, KMAX)))[0]
    res = {"explore_folds": [], "n_train": len(S_tr), "n_test": len(S_te), "n_voxels": Y_tr.shape[1]}
    folds = ec.outer_folds(len(S_tr), seed, n_folds)
    for fi, f in enumerate(folds):
        t0 = time.time()
        tr = np.setdiff1d(np.arange(len(S_tr)), f)
        dirs = fit_dirs(S_tr[tr], Y_tr[tr], seed)
        Z_tr, cols = targets(S_tr[tr], dirs, rand)
        Z_te, _ = targets(S_tr[f], dirs, rand)
        Zh = decode(Y_tr[tr], Z_tr, Y_tr[f], seed)
        res["explore_folds"].append(evaluate(Zh, Z_te, cols))
        r_ = res["explore_folds"][-1]
        log(f"  폴드 {fi + 1}/{n_folds}: RRR k{KMAIN} 쌍정확도 {r_[f'RRR_k{KMAIN}']['pair_acc']:.3f}, PCA {r_[f'PCA_k{KMAIN}']['pair_acc']:.3f}, full {r_['full']['pair_acc']:.3f} ({time.time() - t0:.0f}s)")
    keys = [k for k in res["explore_folds"][0] if not k.startswith("_")]
    res["explore"] = {k: {m: float(np.mean([r[k][m] for r in res["explore_folds"]])) for m in res["explore_folds"][0][k] if not m.endswith("_ci")} for k in keys}
    res["explore_se"] = {k: float(np.std([r[k]["pair_acc"] for r in res["explore_folds"]], ddof=1) / np.sqrt(n_folds)) for k in keys}
    # 확인
    dirs = fit_dirs(S_tr, Y_tr, seed)
    Z_tr, cols = targets(S_tr, dirs, rand)
    Z_te, _ = targets(S_te, dirs, rand)
    Zh = decode(Y_tr, Z_tr, Y_te, seed)
    res["confirm"] = evaluate(Zh, Z_te, cols, boot=B, seed=seed)
    log("확인(shared1000): " + ", ".join(f"{k}={res['confirm'][k]['pair_acc']:.3f}" for k in (f"RRR_k{KMAIN}", f"PCA_k{KMAIN}", f"RAND_k{KMAIN}", "full")))
    c = res["confirm"]["_diffs"][f"RRR_minus_PCA_k{KMAIN}"]
    res["verdict_k12"] = {"diff": c["diff"], "ci": c["ci"], "meets_primary": bool(c["ci"][0] > 0 and c["diff"] >= 0.01),
                          "rrr_vs_full": res["confirm"]["_diffs"]["RRR_main_minus_full"]}
    return res


def selftest():
    rng = np.random.default_rng(0)
    n, d, V = 900, 24, 120
    L = rng.standard_normal((n, d)) * np.linspace(3.0, 0.3, d)
    A = np.linalg.qr(rng.standard_normal((d, d)))[0]
    S = (L @ A).astype(np.float32)
    S = (S - S.mean(0)) / S.std(0)
    W = rng.standard_normal((3, V))
    Y = (L[:, -3:] @ W * 4.0 + rng.standard_normal((n, V)) * 2.0).astype(np.float32)   # 신호는 분산 작은 3방향
    global KS, KMAX, KMAIN
    KS, KMAX, KMAIN = (2, 3, 5), 5, 3
    r = run(S[:600], Y[:600], S[600:], Y[600:], B=30, n_folds=3, log=lambda s: None)
    e = r["explore"]
    print("탐색 쌍정확도 k3: RRR %.3f PCA %.3f RAND %.3f full %.3f" % (e["RRR_k3"]["pair_acc"], e["PCA_k3"]["pair_acc"], e["RAND_k3"]["pair_acc"], e["full"]["pair_acc"]))
    assert e["RRR_k3"]["pair_acc"] > e["PCA_k3"]["pair_acc"] + 0.05, "신호가 작은 분산 방향에 있으면 RRR 좌표가 더 잘 해독돼야 한다"
    assert e["RRR_k3"]["pair_acc"] > 0.7
    # 신호 없음: 우연 수준
    Yn = rng.standard_normal((n, V)).astype(np.float32)
    rn = run(S[:600], Yn[:600], S[600:], Yn[600:], B=10, n_folds=3, log=lambda s: None)
    assert abs(rn["explore"]["RRR_k3"]["pair_acc"] - 0.5) < 0.06, rn["explore"]["RRR_k3"]
    selftest_roi()
    print("selftest 통과")


# ---------- 영역별 decoding (기준은 Notion 「영역별 brain decoding」에 계산 전 기록) ----------
ROIS_DEC = ["EARLY", "S_midventral", "S_midlateral", "S_midparietal", "S_ventral", "S_lateral", "S_parietal", "places", "faces", "bodies", "words"]
HIGH = ["S_ventral", "S_lateral", "S_parietal", "places", "bodies"]


def targets_main(S_ids, dirs, rand):
    """RRR, PCA, 무작위의 처음 KMAIN축 좌표."""
    Zs, cols, c = [], {}, 0
    for m in METHODS:
        D = rand if m == "RAND" else dirs[m]
        Zs.append(S_ids @ D[:, :KMAIN])
        cols[m] = slice(c, c + KMAIN)
        c += KMAIN
    return np.concatenate(Zs, axis=1).astype(np.float32), cols


def run_roi(S_tr, Y_tr, S_te, Y_te, masks, pool_mask, nvox=400, draws=5, B=1000, seed=0, log=print, rois=None):
    """축(방향)은 pool_mask 복셀로 학습 이미지에서 구해 고정하고, 입력 복셀만 영역별로 바꿔(영역마다 nvox개 무작위, draws회) 해독한다."""
    rois = rois or list(masks)
    d = S_tr.shape[1]
    rand = np.linalg.qr(np.random.default_rng(seed + 100).standard_normal((d, KMAX)))[0]
    dirs = fit_dirs(S_tr, Y_tr[:, pool_mask], seed)
    Z_tr, cols = targets_main(S_tr, dirs, rand)
    Z_te, _ = targets_main(S_te, dirs, rand)
    n = len(S_te)
    idxs = [np.random.default_rng(seed + 7 + b).integers(0, n, n) for b in range(B)]
    allm = dict(masks)
    allm["POOL"] = pool_mask
    per = {}
    for r in list(rois) + ["POOL"]:
        cand = np.flatnonzero(allm[r])
        zh = []
        for dd in range(draws):
            pick = np.random.default_rng(seed + 31 * dd + 5).choice(cand, min(nvox, len(cand)), replace=False)
            zh.append(decode(Y_tr[:, pick], Z_tr, Y_te[:, pick], seed + dd))
        res = {"n_voxels_available": int(len(cand)), "n_voxels_used": int(min(nvox, len(cand)))}
        boots = {}
        for m in METHODS:
            sl = cols[m]
            res[m] = {"axis_corr": float(np.mean([axis_corr(z[:, sl], Z_te[:, sl], KMAIN) for z in zh]))}
            tops = [sim_matrix(z[:, sl], Z_te[:, sl]) for z in zh]
            res[m]["top1"] = float(np.mean([topk(S, 1) for S in tops]))
            res[m]["top5"] = float(np.mean([topk(S, 5) for S in tops]))
            boots[m] = np.array([np.mean([axis_corr(z[i][:, sl], Z_te[i][:, sl], KMAIN) for z in zh]) for i in idxs]) if B else None
        per[r] = {"res": res, "boots": boots}
        log(f"  {r:14s} 복셀 {res['n_voxels_used']:4d}  축 상관 RRR {res['RRR']['axis_corr']:.3f} PCA {res['PCA']['axis_corr']:.3f} RAND {res['RAND']['axis_corr']:.3f} | 1등 RRR {res['RRR']['top1']:.3f} PCA {res['PCA']['top1']:.3f}")
    out = {"rois": {r: per[r]["res"] for r in per}, "nvox": nvox, "draws": draws, "n_test": n}
    if B:
        def ci(v):
            return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        hi = [r for r in HIGH if r in per]
        if hi and "EARLY" in per:
            h1 = np.mean([per[r]["boots"]["RRR"] for r in hi], axis=0) - per["EARLY"]["boots"]["RRR"]
            h1_pt = float(np.mean([per[r]["res"]["RRR"]["axis_corr"] for r in hi]) - per["EARLY"]["res"]["RRR"]["axis_corr"])
            out["H1_high_minus_EARLY"] = {"diff": h1_pt, "ci": ci(h1), "met": bool(ci(h1)[0] > 0 and h1_pt >= 0.03)}
        h2 = np.mean([per[r]["boots"]["RRR"] - per[r]["boots"]["PCA"] for r in rois], axis=0)
        h2_pt = float(np.mean([per[r]["res"]["RRR"]["axis_corr"] - per[r]["res"]["PCA"]["axis_corr"] for r in rois]))
        out["H2_RRR_minus_PCA_mean_over_rois"] = {"diff": h2_pt, "ci": ci(h2), "met": bool(ci(h2)[0] > 0 and h2_pt >= 0.03)}
        out["RRR_minus_PCA_by_roi"] = {r: {"diff": float(per[r]["res"]["RRR"]["axis_corr"] - per[r]["res"]["PCA"]["axis_corr"]), "ci": ci(per[r]["boots"]["RRR"] - per[r]["boots"]["PCA"])} for r in per}
    return out


def selftest_roi():
    rng = np.random.default_rng(0)
    n, d, V = 1200, 24, 200
    L = rng.standard_normal((n, d)) * np.linspace(3.0, 0.3, d)
    A = np.linalg.qr(rng.standard_normal((d, d)))[0]
    S = (L @ A).astype(np.float32)
    S = (S - S.mean(0)) / S.std(0)
    W = rng.standard_normal((3, 100))
    sig = L[:, -3:] @ W * 4.0 + rng.standard_normal((n, 100)) * 2.0
    Y = np.concatenate([sig, rng.standard_normal((n, 100)) * 2.0], axis=1).astype(np.float32)
    masks = {"A": np.arange(200) < 100, "B": np.arange(200) >= 100}
    pool = np.arange(200) < 100
    global KMAIN, KMAX
    KMAIN, KMAX = 3, 5
    r = run_roi(S[:800], Y[:800], S[800:], Y[800:], masks, pool, nvox=50, draws=2, B=20, log=lambda s: None, rois=["A", "B"])
    a, b = r["rois"]["A"]["RRR"]["axis_corr"], r["rois"]["B"]["RRR"]["axis_corr"]
    print("영역 시험: 정보 있는 영역 %.3f, 잡음 영역 %.3f" % (a, b))
    assert a > b + 0.2 and abs(b) < 0.15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmri-dir")
    ap.add_argument("--emb-dir")
    ap.add_argument("--model", default="MPNet")
    ap.add_argument("--rep", default="ctx_P1")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--tag", default="subj01")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--roi-mode", action="store_true", help="영역별 decoding(복셀 수를 맞춰 영역마다 따로 해독)")
    ap.add_argument("--nvox", type=int, default=400)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    out = Path(a.out_dir) / f"axes_decode{'_roi' if a.roi_mode else ''}_{a.tag}_{a.model}_{a.rep}.json"
    if a.skip_existing and out.exists():
        print("이미 있음, 건너뜀:", out)
        return
    from text_compare import load_brain
    emb = Path(a.emb_dir)
    ids_all = json.load(open(emb / "ctx_ids.json"))
    E = np.load(emb / f"{a.model}__{a.rep}.npy").astype(np.float32)
    S = (E - E.mean(0)) / (E.std(0) + 1e-6)
    row = {c: i for i, c in enumerate(ids_all)}
    extra = tuple(r for r in ROIS_DEC if r not in ("EARLY",)) if a.roi_mode else ()
    train, Ytr, test, Yte, masks, nk, nv = load_brain(a.fmri_dir, ids_all, extra=extra)
    keep = masks["R2"] | masks["R3"]
    if a.roi_mode:
        print(f"{a.tag} {a.model}: 영역별 decoding, 학습 {len(train)}장, 확인 {len(test)}장, 영역당 복셀 {a.nvox}개 x {a.draws}회", flush=True)
        res = run_roi(S[[row[c] for c in train]], Ytr, S[[row[c] for c in test]], Yte, {r: masks[r] for r in ROIS_DEC}, keep,
                      nvox=a.nvox, draws=a.draws, B=a.boot, log=lambda s: print(s, flush=True))
        res.update(tag=a.tag, model=a.model, rep=a.rep)
        save_json(res, out)
        print("H1:", res["H1_high_minus_EARLY"], "\nH2:", res["H2_RRR_minus_PCA_mean_over_rois"], flush=True)
        return
    print(f"{a.tag} {a.model}: 학습 {len(train)}장, 확인 {len(test)}장, 복셀 {int(keep.sum())}", flush=True)
    res = run(S[[row[c] for c in train]], Ytr[:, keep], S[[row[c] for c in test]], Yte[:, keep], B=a.boot, log=lambda s: print(s, flush=True))
    res.update(tag=a.tag, model=a.model, rep=a.rep)
    save_json(res, out)


if __name__ == "__main__":
    main()
