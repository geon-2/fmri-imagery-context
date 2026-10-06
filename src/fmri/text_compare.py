"""세 텍스트(원본, 사물, 맥락 라벨)와 변형(P1, V2, V3)의 encoding 설명력을 비교한다. 탐색 단계(피험자 1, 라벨이 있는 학습용 이미지).

기준은 Notion 「텍스트 3종의 임베딩 구성 설계」와 「RQ1 판정 설계 초안」(2026-10-04)에 사전 기록.
- 영역 묶음과 주 점수 P = R2와 R3의 복셀 r 중앙값 평균, 교차검증 방식, 부트스트랩은 embedding_compare.py와 같다.
- 셔플 대조: 텍스트를 다른 이미지의 것으로 바꾼다. 항목 수(캡션 수, 사물 수, 맥락 구 수)가 같은 이미지끼리만 바꿔 길이와 형식 효과를 남긴다.
- 맥락의 고유 기여(사물 너머): U_adj = P(사물 + 맥락) − P(사물 + 셔플한 맥락). 주 방식(P1)과 V2 모두에서
  95% 부트스트랩 구간의 하한이 0보다 크고 차이가 0.01 이상이어야 「확인됨(탐색)」. 이 단계는 탐색이며 확인은 shared1000에서 따로 한다.

사용(Colab): python3 src/fmri/text_compare.py --fmri-dir .../roi_betas --emb-dir .../embeddings_v2 --out-dir .../text_compare --model MPNet
시험:        python3 src/fmri/text_compare.py --selftest
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from embedding_compare import (GROUPS, MIN_RELIABILITY, EFFECT_FLOOR, boot_weights, encoding_scores, fold_center,
                               oof_predict, outer_folds)

SINGLES = ("orig_P1", "orig_V2", "obj_P1", "obj_V2", "ctx_P1", "ctx_V2", "ctx_V3")


def matched_perm(counts, seed):
    """같은 항목 수를 가진 이미지끼리만 서로 바꾸는 순열."""
    rng = np.random.default_rng(seed)
    perm = np.arange(len(counts))
    for v in np.unique(counts):
        idx = np.where(counts == v)[0]
        perm[idx] = rng.permutation(idx)
    return perm


def build_sets(F, counts, seed=0):
    """F: {이름: N×d 또는 N×6×d(V3)}. 반환: {세트 이름: N×D 특징}. 셔플은 항목 수가 같은 이미지끼리 바꾼다."""
    X = {k: (F[k].reshape(len(F[k]), -1) if k == "ctx_V3" else F[k]) for k in SINGLES if k in F}
    p = {"orig": matched_perm(counts["orig"], seed + 1), "obj": matched_perm(counts["obj"], seed + 2),
         "ctx": matched_perm(counts["ctx"], seed + 3)}
    S = dict(X)
    for v in ("P1", "V2"):
        S[f"obj_{v}+ctx_{v}"] = np.concatenate([X[f"obj_{v}"], X[f"ctx_{v}"]], axis=1)
        S[f"obj_{v}+ctx_{v}_shuf"] = np.concatenate([X[f"obj_{v}"], X[f"ctx_{v}"][p["ctx"]]], axis=1)
        S[f"ctx_{v}_shuf"] = X[f"ctx_{v}"][p["ctx"]]
        S[f"obj_{v}_shuf"] = X[f"obj_{v}"][p["obj"]]
        S[f"orig_{v}_shuf"] = X[f"orig_{v}"][p["orig"]]
    return S


def primary(point, boot):
    return (point["R2"] + point["R3"]) / 2, (boot["R2"] + boot["R3"]) / 2


def finish(res, boots):
    """대비와 「맥락의 고유 기여」 판정(탐색과 확인이 같은 규칙을 쓴다)."""
    def diff(a, b):
        if a not in boots or b not in boots:
            return None
        d = boots[a] - boots[b]
        return {"diff": float(res["P"][a]["P"] - res["P"][b]["P"]), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}

    res["contrasts"] = {}
    for v in ("P1", "V2"):
        res["contrasts"][v] = {
            "ctx_above_shuffle": diff(f"ctx_{v}", f"ctx_{v}_shuf"),
            "obj_above_shuffle": diff(f"obj_{v}", f"obj_{v}_shuf"),
            "orig_above_shuffle": diff(f"orig_{v}", f"orig_{v}_shuf"),
            "ctx_unique_beyond_obj_raw": diff(f"obj_{v}+ctx_{v}", f"obj_{v}"),
            "ctx_unique_beyond_obj_adj": diff(f"obj_{v}+ctx_{v}", f"obj_{v}+ctx_{v}_shuf"),
            "obj_unique_beyond_ctx_raw": diff(f"obj_{v}+ctx_{v}", f"ctx_{v}"),
            "joint_vs_orig": diff(f"obj_{v}+ctx_{v}", f"orig_{v}"),
        }
    ok = {}
    for v in ("P1", "V2"):
        c = res["contrasts"][v]["ctx_unique_beyond_obj_adj"]
        ok[v] = bool(c and c["ci"][0] > 0 and c["diff"] >= EFFECT_FLOOR)
    res["ctx_unique_claim"] = {"P1": ok["P1"], "V2": ok["V2"], "verdict": "확인됨(탐색)" if ok["P1"] and ok["V2"] else (
        "구성에 의존" if ok["P1"] or ok["V2"] else "확인하지 못함")}
    return res


def run(Y, F, counts, masks, B=1000, seed=0, log=print, skip=()):
    n = len(Y)
    W = boot_weights(n, B, seed + 10)
    folds = outer_folds(n, seed)
    Yc = fold_center(Y, folds)
    S = build_sets(F, counts, seed)
    res = {"n_images": n, "n_voxels": Y.shape[1], "groups": {g: int(m.sum()) for g, m in masks.items()}, "encoding": {}, "P": {}}
    boots = {}
    for name, X in S.items():
        if name in skip:
            continue
        t = time.time()
        P = fold_center(oof_predict(X.astype(np.float32), Y, seed), folds)
        pt, bt = encoding_scores(P, Yc, masks, W)
        pp, pb = primary(pt, bt)
        res["encoding"][name] = pt
        res["P"][name] = {"P": float(pp), "ci": [float(np.percentile(pb, 2.5)), float(np.percentile(pb, 97.5))]}
        boots[name] = pb
        log(f"[encoding] {name:22s} P={pp:.4f} ci=[{res['P'][name]['ci'][0]:.4f}, {res['P'][name]['ci'][1]:.4f}]  " +
            ", ".join(f"{g}={pt[g]:.3f}" for g in GROUPS) + f"  ({time.time() - t:.0f}s)")
        del P

    return finish(res, boots)


def confirm(Ytr, Ftr, ctr, Yte, Fte, cte, masks, B=1000, seed=0, log=print, skip=()):
    """탐색 표본 전체로 적합하고 shared1000에서만 평가한다. 같은 세트·같은 판정 규칙을 쓴다(alpha는 학습 표본 안에서 선택).
    셔플 대조는 학습과 평가 각각에서 항목 수가 같은 이미지끼리 바꾼다."""
    from embedding_compare import choose_alpha, fit_predict
    n = len(Yte)
    W = boot_weights(n, B, seed + 10)
    Yc = fold_center(Yte, [np.arange(n)])
    Str, Ste = build_sets(Ftr, ctr, seed), build_sets(Fte, cte, seed)
    res = {"n_images": n, "n_train": len(Ytr), "n_voxels": Yte.shape[1], "groups": {g: int(m.sum()) for g, m in masks.items()}, "encoding": {}, "P": {}}
    boots = {}
    for name in Str:
        if name in skip:
            continue
        t = time.time()
        Xtr, Xte = Str[name].astype(np.float32), Ste[name].astype(np.float32)
        P = fold_center(fit_predict(Xtr, Ytr, Xte, choose_alpha(Xtr, Ytr, np.random.default_rng(seed + 1))), [np.arange(n)])
        pt, bt = encoding_scores(P, Yc, masks, W)
        pp, pb = primary(pt, bt)
        res["encoding"][name] = pt
        res["P"][name] = {"P": float(pp), "ci": [float(np.percentile(pb, 2.5)), float(np.percentile(pb, 97.5))]}
        boots[name] = pb
        log(f"[confirm] {name:22s} P={pp:.4f} ci=[{res['P'][name]['ci'][0]:.4f}, {res['P'][name]['ci'][1]:.4f}]  ({time.time() - t:.0f}s)")
    res = finish(res, boots)
    res["ctx_unique_claim"]["verdict"] = res["ctx_unique_claim"]["verdict"].replace("(탐색)", "(확인, shared1000)")
    return res


def load_brain(fmri_dir, ctx_ids, extra=()):
    """라벨이 있는 이미지만 쓴다. 학습용 = shared1000을 뺀 이미지, 평가용(shared1000)은 확인 단계에서 따로 쓴다."""
    fmri_dir = Path(fmri_dir)
    tm = list(csv.DictReader(open(fmri_dir / "trial_map.csv", encoding="utf-8")))
    groups = np.load(fmri_dir / "roi_groups.npz")
    allmask = np.zeros(len(groups["R1"]), bool)
    for g in tuple(GROUPS) + tuple(extra):
        allmask |= groups[g]
    Yz = np.empty((len(tm), int(allmask.sum())), np.float32)
    for s in range(1, 41):
        a = np.load(fmri_dir / f"session{s:02d}.npy").astype(np.float32)[:, allmask]
        Yz[(s - 1) * 750:s * 750] = (a - a.mean(0)) / (a.std(0) + 1e-6)
    trials = {}
    for r in tm:
        trials.setdefault(r["cocoId"], []).append(int(r["trial"]))
    shared = {r["cocoId"] for r in tm if r["shared1000"] == "1"}
    have = set(ctx_ids)
    train = [c for c in trials if c not in shared and c in have and len(trials[c]) == 3]
    test = [c for c in trials if c in shared and c in have and len(trials[c]) == 3]
    Ttr = np.array([trials[c] for c in train])

    def pc(a, b):
        a, b = a - a.mean(0), b - b.mean(0)
        return (a * b).sum(0) / (np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0)) + 1e-12)
    rel = (pc(Yz[Ttr[:, 0]], Yz[Ttr[:, 1]]) + pc(Yz[Ttr[:, 0]], Yz[Ttr[:, 2]]) + pc(Yz[Ttr[:, 1]], Yz[Ttr[:, 2]])) / 3
    keep = rel >= MIN_RELIABILITY
    Ytr = Yz[Ttr].mean(1)[:, keep]
    Tte = np.array([trials[c] for c in test])
    Yte = Yz[Tte].mean(1)[:, keep]
    masks = {g: groups[g][allmask][keep] for g in tuple(GROUPS) + tuple(extra)}
    return train, Ytr, test, Yte, masks, int(keep.sum()), int(allmask.sum())


def load_features(emb_dir, model, ids_all, wanted):
    """embeddings_v2의 {모델}__{표현}.npy와 V3 블록을 wanted 이미지 순서로 읽는다."""
    emb_dir = Path(emb_dir)
    row = {c: i for i, c in enumerate(ids_all)}
    sel = [row[c] for c in wanted]
    F = {}
    for k in ("orig_P1", "orig_V2", "obj_P1", "obj_V2", "ctx_P1", "ctx_V2"):
        F[k] = np.load(emb_dir / f"{model}__{k}.npy").astype(np.float32)[sel]
    if (emb_dir / f"{model}__ctx_V3.npz").exists():
        F["ctx_V3"] = np.load(emb_dir / f"{model}__ctx_V3.npz")["blocks"].astype(np.float32)[sel]
    return F


def item_counts(cap, lab, wanted):
    import text_variants as tv
    return {"orig": np.array([len(tv.original_units(cap[c]["captions"])) for c in wanted]),
            "obj": np.array([len(tv.object_names(cap[c]["object_only"])) for c in wanted]),
            "ctx": np.array([len(tv.context_units(lab[c]["labels"])) for c in wanted])}


def selftest():
    rng = np.random.default_rng(0)
    n, d, V = 500, 16, 30
    z_obj, z_ctx = rng.standard_normal((n, d)), rng.standard_normal((n, d))
    Y = (z_obj @ rng.standard_normal((d, V)) * 0.5 + z_ctx @ rng.standard_normal((d, V)) * 0.5 + rng.standard_normal((n, V)) * 2).astype(np.float32)
    noise = lambda: rng.standard_normal((n, d)).astype(np.float32)
    F = {"orig_P1": z_obj + z_ctx + noise(), "orig_V2": z_obj + z_ctx + noise(),
         "obj_P1": z_obj + noise() * 0.3, "obj_V2": z_obj + noise() * 0.5,
         "ctx_P1": z_ctx + noise() * 0.3, "ctx_V2": z_ctx + noise() * 0.5,
         "ctx_V3": np.stack([z_ctx + noise() * 0.5] * 6, axis=1)}
    F = {k: v.astype(np.float32) for k, v in F.items()}
    counts = {"orig": np.full(n, 5), "obj": rng.integers(1, 5, n), "ctx": rng.integers(8, 12, n)}
    masks = {"R1": np.arange(V) < 25, "R2": np.arange(V) < 20, "R3": np.arange(V) >= 10, "EARLY": np.arange(V) < 8}
    res = run(Y, F, counts, masks, B=40, seed=1, skip=("ctx_V3",))
    c = res["contrasts"]["P1"]
    print(json.dumps(c["ctx_unique_beyond_obj_adj"]), json.dumps(res["ctx_unique_claim"], ensure_ascii=False))
    assert c["ctx_above_shuffle"]["diff"] > 0.05           # 신호가 있는 맥락은 셔플보다 높다
    assert abs(res["P"]["ctx_P1_shuf"]["P"]) < 0.08        # 셔플은 0 근처
    assert c["ctx_unique_beyond_obj_adj"]["diff"] > 0.02   # 사물 너머의 고유 기여
    assert res["ctx_unique_claim"]["verdict"] == "확인됨(탐색)"
    # 맥락에 고유 신호가 없는 경우
    F2 = dict(F); F2["ctx_P1"] = noise(); F2["ctx_V2"] = noise()
    res2 = run(Y, F2, counts, masks, B=40, seed=1, skip=("ctx_V3",))
    assert res2["ctx_unique_claim"]["verdict"] != "확인됨(탐색)", res2["ctx_unique_claim"]
    # 확인: 학습/평가를 나눠도 같은 판정
    tr, te = slice(0, 350), slice(350, 500)
    sl = lambda D, i: {k: v[i] for k, v in D.items()}
    rc = confirm(Y[tr], sl(F, tr), sl(counts, tr), Y[te], sl(F, te), sl(counts, te), masks, B=40, seed=1, skip=("ctx_V3",), log=lambda s: None)
    assert rc["contrasts"]["P1"]["ctx_above_shuffle"]["diff"] > 0.03, rc["contrasts"]["P1"]["ctx_above_shuffle"]
    print("confirm 시험:", json.dumps(rc["ctx_unique_claim"], ensure_ascii=False))
    print("selftest 통과")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmri-dir")
    ap.add_argument("--emb-dir")
    ap.add_argument("--bundle-dir")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--model", default="MPNet")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--skip", default="", help="건너뛸 세트 이름(쉼표). 시간이 걸리는 ctx_V3를 빼려면 ctx_V3")
    ap.add_argument("--confirm", action="store_true", help="shared1000에서 확인(탐색 표본으로 적합). 탐색을 끝낸 뒤에만")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import gzip
    import text_variants as tv  # noqa: F401
    cap = {r["cocoId"]: r for r in map(json.loads, gzip.open(Path(a.bundle_dir) / "captions.jsonl.gz", "rt"))}
    lab = {r["cocoId"]: r for r in map(json.loads, gzip.open(Path(a.bundle_dir) / "labels_pass0.jsonl.gz", "rt"))}
    ids_all = json.load(open(Path(a.emb_dir) / "ctx_ids.json"))
    train, Ytr, test, Yte, masks, nkeep, nvox = load_brain(a.fmri_dir, ids_all)
    print(f"학습용(탐색) 이미지 {len(train)}장, 평가용 shared1000 중 라벨 있는 {len(test)}장, 복셀 {nkeep}/{nvox}, 영역 { {g: int(m.sum()) for g, m in masks.items()} }", flush=True)
    if a.confirm:
        res = confirm(Ytr, load_features(a.emb_dir, a.model, ids_all, train), item_counts(cap, lab, train),
                      Yte, load_features(a.emb_dir, a.model, ids_all, test), item_counts(cap, lab, test),
                      masks, B=a.boot, log=lambda s: print(s, flush=True), skip=tuple(x for x in a.skip.split(",") if x))
        res.update(model=a.model, n_reliable_voxels=nkeep, n_voxels_total=nvox)
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(out / f"text_compare_{a.model}_confirm.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
        print("\n맥락의 고유 기여(확인, shared1000):", res["ctx_unique_claim"])
        print(json.dumps(res["contrasts"], ensure_ascii=False, indent=1, default=float))
        return
    F = load_features(a.emb_dir, a.model, ids_all, train)
    counts = item_counts(cap, lab, train)
    res = run(Ytr, F, counts, masks, B=a.boot, log=lambda s: print(s, flush=True), skip=tuple(x for x in a.skip.split(",") if x))
    res.update(model=a.model, n_reliable_voxels=nkeep, n_voxels_total=nvox, n_test_labeled=len(test))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / f"text_compare_{a.model}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print("\n맥락의 고유 기여(탐색):", res["ctx_unique_claim"])
    print(json.dumps(res["contrasts"], ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
