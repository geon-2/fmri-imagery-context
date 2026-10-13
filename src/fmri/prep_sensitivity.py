"""E1 전처리 강건성 점검: 전처리 변형 4개에서 텍스트 조건의 encoding 점수와 순위가 같은지 본다.
기준은 실행 전에 docs/prep_axes_criteria.md와 Notion(E1 전처리 강건성 점검 + E2 축 방법 비교)에 기록했다.

변형  a 세션 z-score + 학습 반복 신뢰도 ≥ 0.1 | b z-score + 선택 없음 | c 표준화 없음 + 선택 없음(MIRAGE식) | d z-score + 영역별 SNR 상위 20%
공통  피험자 1, MPNet, shared1000을 뺀 라벨 있는 학습용 이미지, 반복 3회 평균, 5겹 교차검증, 주 점수 P = R2·R3 복셀 r 중앙값의 평균.
      복셀은 R2∪R3만 읽는다(복셀별로 독립이라 R1·EARLY를 빼도 R2·R3 점수는 같다).
사용: python3 src/fmri/prep_sensitivity.py --fmri-dir data/stage/fmri_subj01 --emb-dir data/embeddings_v2 --bundle-dir data/colab_bundle --out-dir data/results/prep_sensitivity
시험: python3 src/fmri/prep_sensitivity.py --selftest
"""
import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labeling"))  # text_variants
from embedding_compare import boot_weights, encoding_scores, fold_center, oof_predict, outer_folds
from text_compare import build_sets, item_counts, load_features

VARIANTS = {"a": dict(zscore=True, sel="rel"), "b": dict(zscore=True, sel="none"),
            "c": dict(zscore=False, sel="none"), "d": dict(zscore=True, sel="snr20")}
CONDS = ("orig_P1", "orig_V2", "obj_P1", "ctx_P1", "ctx_V2", "obj_P1+ctx_P1", "ctx_P1_shuf")
GROUPS2 = ("R2", "R3")
MIN_REL = 0.1


def pair_corr(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    return (a * b).sum(0) / (np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0)) + 1e-12)


def select_voxels(reps, masks, mode):
    """reps: n×3×V 학습 반복. 반환 bool 길이 V."""
    V = reps.shape[2]
    if mode == "none":
        return np.ones(V, bool)
    if mode == "rel":
        rel = (pair_corr(reps[:, 0], reps[:, 1]) + pair_corr(reps[:, 0], reps[:, 2]) + pair_corr(reps[:, 1], reps[:, 2])) / 3
        return rel >= MIN_REL
    if mode == "snr20":
        sig = reps.mean(1).var(0)                       # 이미지 간 분산(반복 평균)
        noise = reps.var(1, ddof=1).mean(0) / 3          # 반복 평균의 잡음 분산
        snr = sig / (noise + 1e-12)
        keep = np.zeros(V, bool)
        for g in GROUPS2:
            idx = np.where(masks[g])[0]
            thr = np.percentile(snr[idx], 80)
            keep[idx[snr[idx] >= thr]] = True
        return keep
    raise ValueError(mode)


def load_variant(fmri_dir, ids_all, zscore, sel, log=print):
    fmri_dir = Path(fmri_dir)
    tm = list(csv.DictReader(open(fmri_dir / "trial_map.csv", encoding="utf-8")))
    groups = np.load(fmri_dir / "roi_groups.npz")
    allmask = groups["R2"] | groups["R3"]
    Y = np.empty((len(tm), int(allmask.sum())), np.float32)
    for s in range(1, 41):
        a = np.load(fmri_dir / f"session{s:02d}.npy").astype(np.float32)[:, allmask]
        Y[(s - 1) * 750:s * 750] = (a - a.mean(0)) / (a.std(0) + 1e-6) if zscore else a
    trials = {}
    for r in tm:
        trials.setdefault(r["cocoId"], []).append(int(r["trial"]))
    shared = {r["cocoId"] for r in tm if r["shared1000"] == "1"}
    have = set(ids_all)
    train = [c for c in trials if c not in shared and c in have and len(trials[c]) == 3]
    T = np.array([trials[c] for c in train])
    reps = Y[T]                                         # n×3×V
    del Y
    masks_full = {g: groups[g][allmask] for g in GROUPS2}
    keep = select_voxels(reps, masks_full, sel)
    Ym = reps.mean(1)[:, keep]
    masks = {g: masks_full[g][keep] for g in GROUPS2}
    log(f"  이미지 {len(train)}장, 복셀 {int(keep.sum())}/{len(keep)} (R2 {int(masks['R2'].sum())}, R3 {int(masks['R3'].sum())})")
    return train, Ym, masks, int(keep.sum()), len(keep)


def run_variant(name, cfg, ids_all, F_fn, counts_fn, fmri_dir, B, seed, log=print):
    t0 = time.time()
    log(f"[변형 {name}] {cfg}")
    train, Y, masks, nkeep, nvox = load_variant(fmri_dir, ids_all, cfg["zscore"], cfg["sel"], log)
    F = F_fn(train)
    counts = counts_fn(train)
    S = build_sets(F, counts, seed)
    S = {k: v for k, v in S.items() if k in CONDS or k.replace("obj_P1+ctx_P1_shuf", "x") in CONDS}
    n = len(Y)
    W = boot_weights(n, B, seed + 10)
    folds = outer_folds(n, seed)
    Yc = fold_center(Y, folds)
    out = {"n_images": n, "n_voxels": nkeep, "n_voxels_total": nvox, "groups": {g: int(m.sum()) for g, m in masks.items()}, "P": {}, "enc": {}}
    boots = {}
    for cond in CONDS:
        X = S[cond]
        P = fold_center(oof_predict(X.astype(np.float32), Y, seed), folds)
        pt, bt = encoding_scores(P, Yc, masks, W)
        pb = (bt["R2"] + bt["R3"]) / 2
        out["P"][cond] = float((pt["R2"] + pt["R3"]) / 2)
        out["enc"][cond] = {g: float(pt[g]) for g in GROUPS2}
        boots[cond] = pb
        log(f"  {cond:16s} P={out['P'][cond]:.4f}  R2={pt['R2']:.4f} R3={pt['R3']:.4f}  ({time.time() - t0:.0f}s)")
        del P
    return out, boots


def contrasts(P, boots):
    C = {"C1_ctx_minus_obj": ("ctx_P1", "obj_P1"), "C2_ctx_minus_orig": ("ctx_P1", "orig_P1"),
         "C3_objctx_minus_orig": ("obj_P1+ctx_P1", "orig_P1"), "C4_ctxV2_minus_ctxP1": ("ctx_V2", "ctx_P1")}
    res = {}
    for k, (a, b) in C.items():
        d = boots[a] - boots[b]
        res[k] = {"diff": float(P[a] - P[b]), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    return res


def selftest():
    rng = np.random.default_rng(0)
    r = select_voxels(rng.standard_normal((50, 3, 30)), {"R2": np.arange(30) < 20, "R3": np.arange(30) >= 10}, "snr20")
    assert r.sum() >= 4
    a, b = rng.standard_normal((40, 5)), rng.standard_normal((40, 5))
    assert np.allclose(pair_corr(a, a), 1.0)
    print("selftest 통과")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmri-dir")
    ap.add_argument("--emb-dir")
    ap.add_argument("--bundle-dir")
    ap.add_argument("--out-dir", default="data/results/prep_sensitivity")
    ap.add_argument("--model", default="MPNet")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--variants", default="a,b,c,d")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    cap = {r["cocoId"]: r for r in map(json.loads, gzip.open(Path(a.bundle_dir) / "captions.jsonl.gz", "rt"))}
    lab = {r["cocoId"]: r for r in map(json.loads, gzip.open(Path(a.bundle_dir) / "labels_pass0.jsonl.gz", "rt"))}
    ids_all = json.load(open(Path(a.emb_dir) / "ctx_ids.json"))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = {"model": a.model, "variants": {}, "contrasts": {}}
    allb = {}
    for v in a.variants.split(","):
        r, b = run_variant(v, VARIANTS[v], ids_all, lambda tr: load_features(a.emb_dir, a.model, ids_all, tr),
                           lambda tr: item_counts(cap, lab, tr), a.fmri_dir, a.boot, 0, log=lambda s: print(s, flush=True))
        r["contrasts"] = contrasts(r["P"], b)
        res["variants"][v] = r
        allb[v] = b
        json.dump(res, open(out / f"prep_sensitivity_{a.model}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        np.savez_compressed(out / f"prep_sensitivity_{a.model}_boots.npz", **{f"{vv}__{c}": x for vv, bb in allb.items() for c, x in bb.items()})
    print("저장:", out)


if __name__ == "__main__":
    main()
