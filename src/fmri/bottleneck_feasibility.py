"""병목 encoder–decoder 가능성 점검(사전 점검, 복원 성능 비교 아님). 기준은 실행 전에 docs/bottleneck_feasibility_criteria.md와 Notion에 기록했다.
perception으로만 학습한 지도형 병목(encoder)의 부분공간에 imagery(세트 A·C)가 정렬되는지, 정보가 유지되는지 본다. 세트 B는 읽지 않는다.
사용: python3 src/fmri/bottleneck_feasibility.py --emb-dir data/embeddings_v2 --bundle-dir data/colab_bundle --out-dir data/results/bottleneck_feasibility
"""
import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compress_compare as cc  # noqa: E402
import denoise_decode as dd  # noqa: E402
import imagery_roi as ir  # noqa: E402

LAM = 1e4
KS = (16, 32, 64)
KMAIN = 32
TARGETS = ("ctx_P1", "orig_P1", "clip_img")
NRAND = 500


def orth(M):
    return np.linalg.qr(M)[0]


def ident(P1, P2):
    return ir.ident_accuracy(P1, P2)[0]


def ratio(Pc, U):
    return float(((Pc @ U) ** 2).sum() / ((Pc ** 2).sum() + 1e-12))


def patterns(B, runs, cols):
    """세트별 imagery run 1·2 패턴과 viewing 홀짝 패턴(각 6 × 복셀)."""
    out = {}
    for s, info in ir.SETS.items():
        pats = []
        for r in info["img_runs"]:
            m = np.isin(cols, list(info["img_cols"])) & (runs == r)
            Xr, cr = B[m], cols[m]
            pats.append(np.stack([Xr[cr == c].mean(0) for c in info["img_cols"]]))
        mv = (runs == info["vis_run"]) & (cols < 6)
        Xv, cv = B[mv], cols[mv]
        h1 = np.stack([Xv[cv == c][0::2].mean(0) for c in range(6)])
        h2 = np.stack([Xv[cv == c][1::2].mean(0) for c in range(6)])
        out[s] = {"img": pats, "vis": (h1, h2)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir")
    ap.add_argument("--bundle-dir")
    ap.add_argument("--out-dir", default="data/results/bottleneck_feasibility")
    ap.add_argument("--subjects", default="01,02,05,07")
    a = ap.parse_args()
    import gzip
    cap = {r["cocoId"]: r for r in map(json.loads, gzip.open(Path(a.bundle_dir) / "captions.jsonl.gz", "rt"))}
    lab = {r["cocoId"]: r for r in map(json.loads, gzip.open(Path(a.bundle_dir) / "labels_pass0.jsonl.gz", "rt"))}
    ids_all = json.load(open(Path(a.emb_dir) / "ctx_ids.json"))
    runs, cols = ir.trial_table()
    rng = np.random.default_rng(0)
    res = {}
    t0 = time.time()
    for sid in a.subjects.split(","):
        stage = f"data/stage/fmri_subj{sid}"
        train, reps, keep, allmask = dd.load_reps(stage, ids_all)
        Y0 = reps.mean(1)[:, keep]
        del reps
        n, V = Y0.shape
        T = cc.load_targets(Namespace(emb_dir=a.emb_dir), ids_all, train, cap, lab)
        # imagery 베타: 같은 복셀 집합, 복셀마다 imagery 실험 전체 시행에서 z-score
        B, _ = ir.load_subject(f"subj{sid}")
        B = B[:, allmask][:, keep]
        B = (B - B.mean(0)) / (B.std(0) + 1e-6)
        P = patterns(B, runs, cols)
        perm = rng.permutation(n)
        tr, te = np.sort(perm[:int(n * 0.8)]), np.sort(perm[int(n * 0.8):])
        mu, sd = Y0[tr].mean(0), Y0[tr].std(0) + 1e-6
        Xs = (Y0[tr] - mu) / sd
        U_, s_, Vt_ = np.linalg.svd(Xs, full_matrices=False)
        pca_dirs = Vt_[:max(KS)].T
        Xte = (Y0[te] - mu) / sd
        Xte_c = Xte - Xte.mean(0)
        enc = {}
        for t in TARGETS:
            Za = (T[t][tr] - T[t][tr].mean(0)) / (T[t][tr].std(0) + 1e-6)
            W = Vt_.T @ ((s_ / (s_ ** 2 + LAM))[:, None] * (U_.T @ (Za - Za.mean(0))))
            Vk = np.linalg.svd(Xs @ W, full_matrices=False)[2].T
            enc[t] = {k: orth(W @ Vk[:, :k]) for k in KS}
        rand = {k: [orth(np.random.default_rng(1000 + i).standard_normal((V, k))) for i in range(NRAND)] for k in KS}
        unit = {}
        for s in ("A", "C"):
            img1, img2 = P[s]["img"]
            v1, v2 = P[s]["vis"]
            Pc = np.concatenate([img1, img2]).reshape(2, 6, V).mean(0)
            Pc = Pc - Pc.mean(0)
            u = {"full": {"img": ident(img1, img2), "vis": ident(v1, v2)}}
            for k in KS:
                sub = {"pca": pca_dirs[:, :k]}
                for t in TARGETS:
                    sub[f"enc_{t}"] = enc[t][k]
                for name, E in sub.items():
                    u[f"{name}_{k}"] = {"img": ident(img1 @ E, img2 @ E), "vis": ident(v1 @ E, v2 @ E),
                                         "ratio_img": ratio(Pc, E), "ratio_perc": ratio(Xte_c, E)}
                rr = [(ident(img1 @ R, img2 @ R), ident(v1 @ R, v2 @ R), ratio(Pc, R), ratio(Xte_c, R)) for R in rand[k][:200]]
                rat = [ratio(Pc, R) for R in rand[k]]
                u[f"rand_{k}"] = {"img": float(np.mean([x[0] for x in rr])), "vis": float(np.mean([x[1] for x in rr])), "ratio_img": float(np.mean(rat)),
                                  "ratio_img_p95": float(np.percentile(rat, 95)), "ratio_perc": float(np.mean([x[3] for x in rr]))}
            u["_pat"] = None
            unit[s] = u
        res[sid] = unit
        print(f"[{sid}] 완료 ({time.time() - t0:.0f}s) 전체 복셀 imagery 식별 A {unit['A']['full']['img']:.2f} C {unit['C']['full']['img']:.2f}, viewing A {unit['A']['full']['vis']:.2f} C {unit['C']['full']['vis']:.2f}", flush=True)
        # 순열 검정용 패턴 저장
        np.savez_compressed(Path(a.out_dir) / f"patterns_{sid}.npz", **{f"{s}_{k}": v for s in P for k, v in (("img1", P[s]["img"][0]), ("img2", P[s]["img"][1]), ("vis1", P[s]["vis"][0]), ("vis2", P[s]["vis"][1]))}) if Path(a.out_dir).mkdir(parents=True, exist_ok=True) is None else None
    json.dump(res, open(Path(a.out_dir) / "bottleneck_feasibility.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=lambda x: None)
    print("완료", flush=True)


if __name__ == "__main__":
    main()
