"""병목 encoder→z→decoder 가능성 점검 2(decoder 포함, 학습은 perception만, imagery는 평가 전용). 기준은 실행 전에 docs/bottleneck_decoder_feasibility_criteria.md와 Notion에 기록했다.
세트 B(자연 장면)는 읽지 않는다. 사용: python3 src/fmri/bottleneck_decoder_pilot.py --emb-dir data/embeddings_v2 --bundle-dir data/colab_bundle --out-dir data/results/bottleneck_decoder_pilot
"""
import argparse
import gzip
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import axes_decode as ad  # noqa: E402
import bottleneck_feasibility as bf  # noqa: E402
import compress_compare as cc  # noqa: E402
import denoise_decode as dd  # noqa: E402
import imagery_roi as ir  # noqa: E402
import text_decode_compare as td  # noqa: E402

LAM = 1e4
KS = (16, 32, 64, 128)
KMAIN = 64
TARGETS = ("clip_img", "ctx_P1", "orig_P1")
WORDS = ["stripes", "zebra", "mammal", "yellow", "banana", "fruit"]
SHAPES = ["bar_000.0deg_450L_43W.png", "bar_045.0deg_450L_43W.png", "bar_090.0deg_450L_43W.png", "bar_135.0deg_450L_43W.png", "crs_000.0deg_450L_43W.png", "crs_045.0deg_450L_43W.png"]
NRAND = 5


def candidates():
    import torch
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer
    name = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(name).eval()
    with torch.no_grad():
        imgs = [Image.open(f"data/raw/nsd_imagery/exp/rawtargetimages/setA/{f}").convert("RGB") for f in SHAPES]
        f = model.get_image_features(**CLIPImageProcessor.from_pretrained(name)(images=imgs, return_tensors="pt"))
        f = f.pooler_output if hasattr(f, "pooler_output") else f
        A = torch.nn.functional.normalize(f, dim=-1).numpy()
        tok = CLIPTokenizer.from_pretrained(name)
        g = model.get_text_features(**tok(WORDS, padding=True, return_tensors="pt"))
        g = g.pooler_output if hasattr(g, "pooler_output") else g
        C = torch.nn.functional.normalize(g, dim=-1).numpy()
    return {"A": A.astype(np.float32), "C": C.astype(np.float32)}


def ident(pred, cand):
    return ir.ident_accuracy(pred, cand)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir")
    ap.add_argument("--bundle-dir")
    ap.add_argument("--out-dir", default="data/results/bottleneck_decoder_pilot")
    ap.add_argument("--subjects", default="01,02,05,07")
    a = ap.parse_args()
    cap = {r["cocoId"]: r for r in map(json.loads, gzip.open(Path(a.bundle_dir) / "captions.jsonl.gz", "rt"))}
    lab = {r["cocoId"]: r for r in map(json.loads, gzip.open(Path(a.bundle_dir) / "labels_pass0.jsonl.gz", "rt"))}
    ids_all = json.load(open(Path(a.emb_dir) / "ctx_ids.json"))
    runs, cols = ir.trial_table()
    cand = candidates()
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    res, t0 = {}, time.time()
    for sid in a.subjects.split(","):
        train, reps, keep, allmask = dd.load_reps(f"data/stage/fmri_subj{sid}", ids_all)
        Y0 = reps.mean(1)[:, keep]
        del reps
        n, V = Y0.shape
        T = cc.load_targets(Namespace(emb_dir=a.emb_dir), ids_all, train, cap, lab)
        B, _ = ir.load_subject(f"subj{sid}")
        B = B[:, allmask][:, keep]
        B = (B - B.mean(0)) / (B.std(0) + 1e-6)
        P = bf.patterns(B, runs, cols)
        perm = np.random.default_rng(10 + int(sid)).permutation(n)
        tr, te = np.sort(perm[:int(n * 0.8)]), np.sort(perm[int(n * 0.8):])
        mu, sd = Y0[tr].mean(0), Y0[tr].std(0) + 1e-6
        Xs = (Y0[tr] - mu) / sd
        Xte = (Y0[te] - mu) / sd
        U_, s_, Vt_ = np.linalg.svd(Xs, full_matrices=False)
        out = {}
        for t in TARGETS:
            tm, ts = T[t][tr].mean(0), T[t][tr].std(0) + 1e-6
            Za, Zte = (T[t][tr] - tm) / ts, (T[t][te] - tm) / ts
            W = Vt_.T @ ((s_ / (s_ ** 2 + LAM))[:, None] * (U_.T @ (Za - Za.mean(0))))
            b0 = Za.mean(0)
            Vk = np.linalg.svd(Xs @ W, full_matrices=False)[2].T
            models = {"full": (lambda X, W=W, b0=b0: X @ W + b0)}

            def lin(E, E_dim):
                z_tr = Xs @ E
                zc = z_tr.mean(0)
                D = np.linalg.lstsq(z_tr - zc, Za - b0, rcond=None)[0]
                return lambda X, E=E, D=D, zc=zc, b0=b0: (X @ E - zc) @ D + b0
            for k in KS:
                models[f"enc_{k}"] = lin(bf.orth(W @ Vk[:, :k]), k)
                models[f"pca_{k}"] = lin(Vt_[:k].T, k)
                for i in range(NRAND):
                    models[f"rand{i}_{k}"] = lin(bf.orth(np.random.default_rng(100 * i + k).standard_normal((V, k))), k)
            res_t = {}
            for name, f in models.items():
                pred = f(Xte)
                c1, c5 = td.credit(ad.sim_matrix(pred, Zte))
                res_t[name] = {"perc_top1": float(c1.mean()), "perc_top5": float(c5.mean())}
                # imagery/viewing: 해당 세트에 맞는 타깃만
                for s in ("A", "C"):
                    if (s == "A" and t != "clip_img") or (s == "C" and t == "clip_img"):
                        continue
                    img1, img2 = P[s]["img"]
                    v1, v2 = P[s]["vis"]
                    Ec = (cand[s] - tm) / ts
                    img = f((img1 + img2) / 2)
                    vis = f((v1 + v2) / 2)
                    ai, Ci = ir.ident_accuracy(img, Ec)
                    av, Cv = ir.ident_accuracy(vis, Ec)
                    res_t[name][f"{s}_img"], res_t[name][f"{s}_vis"] = ai, av
                    if name == "full":
                        res_t[name][f"{s}_img_C"], res_t[name][f"{s}_vis_C"] = Ci.tolist(), Cv.tolist()
            out[t] = res_t
        res[sid] = out
        print(f"[{sid}] 완료 ({time.time() - t0:.0f}s)", flush=True)
        np.savez_compressed(Path(a.out_dir) / f"cand_{sid}.npz", A=cand["A"], C=cand["C"])
    json.dump(res, open(Path(a.out_dir) / "bottleneck_decoder_pilot.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("완료", flush=True)


if __name__ == "__main__":
    main()
