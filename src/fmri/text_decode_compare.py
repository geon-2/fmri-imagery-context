"""1단계 텍스트 표현 비교(decoding 주 지표). 기준은 실행 전에 docs/stage1_text_repr_criteria.md와 Notion(1단계 텍스트 표현 비교)에 기록했다.

복셀(전처리 a, R2∪R3) → 텍스트 임베딩(전체 차원) ridge, 보류 폴드 이미지끼리 1장 찾기(top-1 주, top-5 보조, 동점은 부분 점수).
표현: orig_P1, orig_V2, obj_P1, obj_V2, ctx_P1, ctx_V2, objctx_P1(= obj_P1 + ctx_P1 붙임), ctx_P1_shuf(셔플 대조). 인코더 CLIP, MPNet.
사용: python3 src/fmri/text_decode_compare.py --fmri-dir data/stage/fmri_subj01 --emb-dir data/embeddings_v2 --bundle-dir data/colab_bundle --out-dir data/results/text_decode_compare
시험: python3 src/fmri/text_decode_compare.py --selftest
"""
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labeling"))
import axes_decode as ad  # noqa: E402
import embedding_compare as ec  # noqa: E402

REPS = ("orig_P1", "orig_V2", "obj_P1", "obj_V2", "ctx_P1", "ctx_V2", "obj_P1+ctx_P1", "ctx_P1_shuf")
ENCODERS = ("CLIP", "MPNet")
KEY3 = ("orig_P1", "ctx_P1", "obj_P1+ctx_P1")
CONTRASTS = {"K1_ctx_minus_obj": ("ctx_P1", "obj_P1"), "K2_ctx_minus_orig": ("ctx_P1", "orig_P1"),
             "K3_ctxV2_minus_ctxP1": ("ctx_V2", "ctx_P1"), "K4_objctx_minus_orig": ("obj_P1+ctx_P1", "orig_P1"),
             "K5_T10_minus_ctxP1": ("ctx_T10", "ctx_P1"), "K6_T10_minus_ctxV2": ("ctx_T10", "ctx_V2")}
MIN_EFFECT = 0.02


def credit(Sm):
    """정답 후보(대각선)의 순위를 동점 부분 점수로 계산한다. 반환 (top1 점수, top5 점수) 길이 n."""
    d = np.diag(Sm)
    n = len(Sm)
    g = (Sm > d[:, None] + 1e-6).sum(1)                        # 정답보다 확실히 높은 후보 수
    e = (np.abs(Sm - d[:, None]) <= 1e-6).sum(1) - 1             # 정답과 같은 점수의 다른 후보 수
    c1 = np.where(g == 0, 1.0 / (1 + e), 0.0)
    c5 = np.clip((5 - g) / (1 + e), 0.0, 1.0)
    return c1.astype(np.float32), c5.astype(np.float32)


def run(Y, S_by_enc, B=1000, seed=0, log=print):
    """Y: n×V, S_by_enc: {인코더: {표현: n×D}}. 반환 {인코더: {표현: {c1: (n,), c5: (n,)}}}."""
    n = len(Y)
    folds = ec.outer_folds(n, seed)
    out = {e: {r: {"c1": np.zeros(n, np.float32), "c5": np.zeros(n, np.float32)} for r in S_by_enc[e]} for e in S_by_enc}
    t0 = time.time()
    for fi, f in enumerate(folds):
        tr = np.setdiff1d(np.arange(n), f)
        for enc, S in S_by_enc.items():
            names = list(S)
            dims = [S[r].shape[1] for r in names]
            Ztr = np.concatenate([S[r][tr] for r in names], axis=1).astype(np.float32)
            Zte = np.concatenate([S[r][f] for r in names], axis=1).astype(np.float32)
            Zhat = ad.decode(Y[tr], Ztr, Y[f], seed=seed + fi)
            c = 0
            for r, d in zip(names, dims):
                sl = slice(c, c + d)
                c += d
                c1, c5 = credit(ad.sim_matrix(Zhat[:, sl], Zte[:, sl]))
                out[enc][r]["c1"][f], out[enc][r]["c5"][f] = c1, c5
            log(f"  폴드 {fi + 1}/{len(folds)} {enc} 완료 ({time.time() - t0:.0f}s)")
    return out


def summarize(out, B=1000, seed=0):
    n = len(next(iter(next(iter(out.values())).values()))["c1"])
    rng = np.random.default_rng(seed + 5)
    idx = np.stack([rng.integers(0, n, n) for _ in range(B)])

    def bs(v):
        m = v[idx].mean(1)
        return float(v.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

    res = {"n_images": n, "reps": {}, "contrasts": {}, "encoder_rule": {}}
    for enc in out:
        res["reps"][enc] = {}
        for r, d in out[enc].items():
            m1, ci1 = bs(d["c1"])
            m5, ci5 = bs(d["c5"])
            res["reps"][enc][r] = {"top1": m1, "top1_ci": ci1, "top5": m5, "top5_ci": ci5}
        res["contrasts"][enc] = {}
        for k, (a, b) in CONTRASTS.items():
            if a in out[enc] and b in out[enc]:
                diff, ci = bs(out[enc][a]["c1"] - out[enc][b]["c1"])
                res["contrasts"][enc][k] = {"diff": diff, "ci": ci, "verdict": "차이 있음" if (ci[0] > 0 or ci[1] < 0) and abs(diff) >= MIN_EFFECT else "동등"}
    if "CLIP" in out and "MPNet" in out:
        wins = []
        for r in KEY3:
            diff, ci = bs(out["MPNet"][r]["c1"] - out["CLIP"][r]["c1"])
            ok = bool(ci[0] > 0 and diff >= MIN_EFFECT)
            wins.append(ok)
            res["encoder_rule"][r] = {"mpnet_minus_clip": diff, "ci": ci, "mpnet_clearly_higher": ok}
        res["encoder_rule"]["verdict"] = "MPNet 선정 논의" if sum(wins) >= 2 else "CLIP 유지"
    return res


def selftest():
    rng = np.random.default_rng(0)
    n, d, V = 300, 12, 60
    sig = rng.standard_normal((n, d))
    Y = (sig @ rng.standard_normal((d, V)) * 0.6 + rng.standard_normal((n, V))).astype(np.float32)
    dup = np.repeat(rng.standard_normal((n // 10, d)), 10, axis=0)           # 같은 임베딩이 10번 반복(동점)
    S = {"CLIP": {"real": (sig + 0.3 * rng.standard_normal((n, d))).astype(np.float32), "dup": dup.astype(np.float32),
                  "shuf": sig[rng.permutation(n)].astype(np.float32)}}
    out = run(Y, S, log=lambda s: None)
    res = summarize(out, B=100)
    r = res["reps"]["CLIP"]
    print({k: round(v["top1"], 3) for k, v in r.items()})
    assert r["real"]["top1"] > 5 * r["shuf"]["top1"] + 0.02 and r["shuf"]["top1"] < 0.03, r
    assert r["dup"]["top1"] < 0.2                                            # 동점은 부분 점수만
    print("selftest 통과")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmri-dir")
    ap.add_argument("--emb-dir")
    ap.add_argument("--bundle-dir")
    ap.add_argument("--out-dir", default="data/results/text_decode_compare")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import prep_sensitivity as ps
    from text_compare import build_sets, item_counts, load_features
    cap = {r["cocoId"]: r for r in map(json.loads, gzip.open(Path(a.bundle_dir) / "captions.jsonl.gz", "rt"))}
    lab = {r["cocoId"]: r for r in map(json.loads, gzip.open(Path(a.bundle_dir) / "labels_pass0.jsonl.gz", "rt"))}
    ids_all = json.load(open(Path(a.emb_dir) / "ctx_ids.json"))
    train, Y, masks, nkeep, nvox = ps.load_variant(a.fmri_dir, ids_all, True, "rel", log=lambda s: print(s, flush=True))
    counts = item_counts(cap, lab, train)
    S_by_enc = {}
    for enc in ENCODERS:
        F = load_features(a.emb_dir, enc, ids_all, train)
        F.pop("ctx_V3", None)
        S = build_sets(F, counts, 0)
        S_by_enc[enc] = {r: S[r].astype(np.float32) for r in REPS}
    print(f"1단계: 이미지 {len(train)}장, 복셀 {Y.shape[1]}, 인코더 {list(S_by_enc)}, 표현 {list(REPS)}", flush=True)
    out = run(Y, S_by_enc, B=a.boot, log=lambda s: print(s, flush=True))
    res = summarize(out, B=a.boot)
    res["n_voxels"] = int(Y.shape[1])
    o = Path(a.out_dir)
    o.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(o / "text_decode_compare.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    np.savez_compressed(o / "text_decode_compare_scores.npz", **{f"{e}__{r}__{k}": v[k] for e in out for r, v in out[e].items() for k in ("c1", "c5")})
    print("저장:", o)
    for enc in res["reps"]:
        print(enc, {r: round(v["top1"], 4) for r, v in res["reps"][enc].items()})
    print(json.dumps(res["encoder_rule"], ensure_ascii=False))


if __name__ == "__main__":
    main()
