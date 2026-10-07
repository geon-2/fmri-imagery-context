"""라벨 차원 하나를 뺀 임베딩으로 encoding 점수 P가 바뀌는지(ablation). 기준은 계산 전에 정했다(2026-10-08, Notion 기록).

만드는 법: ctx_P1은 구(phrase) 임베딩의 평균이므로 V3의 차원별 블록 평균(항목별 평균)과 항목별 구 개수 n_d로
  ctx_P1 ≈ Σ n_d·block_d / Σ n_d 로 복원할 수 있다. 복원 코사인이 저장된 ctx_P1과 0.99 미만이면 중단한다.
  빼는 차원: scale(주), implied_event(대조: 구 개수 비중이 비슷한 다른 차원).
점수: full(전체 차원 ridge)와 RRR k=12에서의 P(R2·R3 median r 평균, 5폴드, axes_k_curve와 같은 폴드·점수). 같은 폴드·같은 부트스트랩 가중치로 짝지은 차이.
판정 단위: ΔP = P(scale 뺀 것) − P(전체 ctx), full과 k=12 각각, 피험자 4 × 모델 2 = 8 조합(대조 차원은 판정에 안 씀).
  영향 없음      : 8조합 중 6조합 이상에서 full과 k=12 둘 다 |ΔP| < 0.005 (P의 약 1%)
  scale 기여     : 6조합 이상에서 ΔP ≤ −0.005이고 부트스트랩 95% 구간 상한 < 0 (full과 k=12 둘 다)
  scale 제거 이득: 6조합 이상에서 ΔP ≥ +0.005이고 구간 하한 > 0 (둘 다)
  그 외          : 불확정
해석 한계: 구 개수 비중이 달라(scale ≈ 15%, implied_event ≈ 11%) 뺐을 때 바뀌는 크기가 비중에도 의존한다. 영향 없음이 「scale이 불안정해도 된다」는 근거이면서 「scale이 뇌 예측에 정보를 안 준다」는 뜻이기도 하다.
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

import axes_k_curve as ak
import embedding_compare as ec
import text_compare as tc
import text_variants as tv

DIMS = tv.DIMS  # place_type, environment, activity, implied_event, scale, notes


def build(emb_dir, model, bundle_dir, ids_all):
    z = np.load(Path(emb_dir) / f"{model}__ctx_V3.npz")
    blocks, mask = z["blocks"].astype(np.float32), z["mask"]
    lab = {r["cocoId"]: r for r in map(json.loads, gzip.open(Path(bundle_dir) / "labels_pass0.jsonl.gz", "rt"))}
    n = np.array([[len(tv.context_by_dim(lab[c]["labels"])[d]) for d in DIMS] for c in ids_all], np.float32)
    n = n * mask  # 블록이 없는 항목은 개수 0
    stored = np.load(Path(emb_dir) / f"{model}__ctx_P1.npy").astype(np.float32)

    def comb(drop=()):
        w = n.copy()
        for d in drop:
            w[:, DIMS.index(d)] = 0
        return (w[:, :, None] * blocks).sum(1) / np.maximum(w.sum(1, keepdims=True), 1e-9)
    rec = comb()
    cos = float(np.mean((rec * stored).sum(1) / (np.linalg.norm(rec, axis=1) * np.linalg.norm(stored, axis=1) + 1e-9)))
    share = {d: float(n[:, i].sum() / n.sum()) for i, d in enumerate(DIMS)}
    return {"full": stored, "noscale": comb(("scale",)), "noimpl": comb(("implied_event",))}, cos, share


def one(reps, Y, masks, B, seed=0, k=12, log=print):
    n = len(Y)
    folds = ec.outer_folds(n, seed)
    W = ec.boot_weights(n, B, seed + 7)
    sc = ak.Scorer(Y, folds, masks, W)
    out = {}
    for name, X in reps.items():
        rng = np.random.default_rng(seed + 1)
        of, ok = np.empty(Y.shape, np.float32), np.empty(Y.shape, np.float32)
        for f in folds:
            tr = np.setdiff1d(np.arange(n), f)
            p = ak.prep(X[tr], Y[tr], X[f], rng)
            of[f] = p["Xt"] @ p["B"] + p["ym"]
            ok[f] = ak.predict_dir(p["Xa"], p["Xt"], p["Yc"], p["ym"], p["D"]["RRR"], k)
        out[name] = {"full": sc.score(of), f"rrr{k}": sc.score(ok)}
        log(f"  {name:8s} full P={out[name]['full']['P']:.4f}  RRR{k} P={out[name][f'rrr{k}']['P']:.4f}")
    res = {}
    for name in ("noscale", "noimpl"):
        for m in ("full", f"rrr{k}"):
            d = out[name][m]["boot"] - out["full"][m]["boot"]
            res[f"{name}_{m}"] = {"dP": out[name][m]["P"] - out["full"][m]["P"], "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    res["P"] = {n_: {m: v[m]["P"] for m in v} for n_, v in out.items()}
    return res


def verdict(rows, thr=0.005, need=6):
    none = gain = loss = 0
    for r in rows:
        a, b = r["noscale_full"], r["noscale_rrr12"]
        none += abs(a["dP"]) < thr and abs(b["dP"]) < thr
        loss += all(x["dP"] <= -thr and x["ci"][1] < 0 for x in (a, b))
        gain += all(x["dP"] >= thr and x["ci"][0] > 0 for x in (a, b))
    v = "영향 없음" if none >= need else "scale 기여" if loss >= need else "scale 제거 이득" if gain >= need else "불확정"
    return v, {"none": int(none), "loss": int(loss), "gain": int(gain), "n": len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-dir", default="data/stage")
    ap.add_argument("--emb-dir", default="data/embeddings_v2")
    ap.add_argument("--bundle-dir", default="data/colab_bundle")
    ap.add_argument("--out", default="data/results/ablate_scale.json")
    ap.add_argument("--subjects", default="subj01,subj02,subj05,subj07")
    ap.add_argument("--models", default="MPNet,CLIP")
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    ids_all = json.load(open(Path(a.emb_dir) / "ctx_ids.json"))
    row = {c: i for i, c in enumerate(ids_all)}
    res = json.load(open(a.out)) if Path(a.out).exists() else {"combos": {}}
    for model in a.models.split(","):
        embs, cos, share = build(a.emb_dir, model, a.bundle_dir, ids_all)
        print(f"{model}: 복원 코사인 {cos:.4f}, 구 비중 { {k: round(v, 3) for k, v in share.items()} }", flush=True)
        if cos < 0.99:
            raise SystemExit("복원이 저장된 ctx_P1과 다르다. 중단.")
        res["recon_cos_" + model], res["share_" + model] = cos, share
        for s in a.subjects.split(","):
            key = f"{s}_{model}"
            if key in res["combos"]:
                print("건너뜀", key)
                continue
            train, Ytr, test, Yte, masks, nk, nv = tc.load_brain(f"{a.stage_dir}/fmri_{s}", ids_all)
            keep = masks["R2"] | masks["R3"]
            Y, mk = Ytr[:, keep], {g: m[keep] for g, m in masks.items()}
            idx = [row[c] for c in train]
            print(key, f"{len(train)}장 복셀 {int(keep.sum())}", flush=True)
            res["combos"][key] = one({k: v[idx] for k, v in embs.items()}, Y, mk, a.boot)
            json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    rows = list(res["combos"].values())
    v, cnt = verdict(rows)
    res["verdict"], res["counts"] = v, cnt
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    for k, r in res["combos"].items():
        print(k, "scale 뺌 full dP %+.4f %s | k12 dP %+.4f %s | (대조 implied 뺌: full %+.4f, k12 %+.4f)" % (
            r["noscale_full"]["dP"], np.round(r["noscale_full"]["ci"], 4), r["noscale_rrr12"]["dP"], np.round(r["noscale_rrr12"]["ci"], 4),
            r["noimpl_full"]["dP"], r["noimpl_rrr12"]["dP"]))
    print("판정:", v, cnt)


if __name__ == "__main__":
    main()
