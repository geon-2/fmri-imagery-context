"""A 단계 확장: 타깃 종류(텍스트 대 이미지 표현) 비교. 기준은 실행 전에 docs/stageA_target_type_criteria.md와 Notion에 기록했다.

1) 학습 이미지(4,999장)의 CLIP ViT-L/14 이미지, DINOv2-base [CLS], ResNet-50 풀링 특징을 한 번 계산해 저장한다(이미 있으면 재사용).
2) 전처리 a의 복셀 → 각 타깃(전체 차원) ridge decoding, 보류 폴드 약 1,000장 중 본 이미지 1장 찾기(top-1 주, top-5 함께, 동점 부분 점수).
사용: python3 src/fmri/target_type_compare.py --fmri-dir data/stage/fmri_subj01 --emb-dir data/embeddings_v2 --bundle-dir data/colab_bundle --img-dir data/raw/images --out-dir data/results/target_type_compare
"""
import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prep_sensitivity as ps  # noqa: E402
import text_decode_compare as td  # noqa: E402
from text_compare import build_sets, item_counts, load_features  # noqa: E402

IMG_TARGETS = ("clip_img", "dinov2", "resnet50")
TEXT_TARGETS = ("orig_P1", "obj_P1", "ctx_P1")


def embed_images(ids, img_dir, out_dir, bs=32, prefix="IMG"):
    import torch
    from PIL import Image
    out_dir = Path(out_dir)
    cache = {k: out_dir / f"{prefix}__{k}.npy" for k in IMG_TARGETS}
    if all(p.exists() for p in cache.values()) and np.load(cache["clip_img"]).shape[0] == len(ids):
        return {k: np.load(p) for k, p in cache.items()}
    from transformers import AutoImageProcessor, AutoModel, CLIPImageProcessor, CLIPModel
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    res = {}
    paths = [Path(img_dir) / f"{c}.jpg" for c in ids]

    def batches():
        for i in range(0, len(paths), bs):
            yield [Image.open(p).convert("RGB") for p in paths[i:i + bs]]

    t0 = time.time()
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval()
    cproc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    out = []
    with torch.no_grad():
        for imgs in batches():
            f = clip.get_image_features(**{k: v.to(dev) for k, v in cproc(images=imgs, return_tensors="pt").items()})
            f = f.pooler_output if hasattr(f, "pooler_output") else f
            out.append(torch.nn.functional.normalize(f, dim=-1).float().cpu().numpy())
    res["clip_img"] = np.concatenate(out)
    del clip
    print(f"  CLIP 이미지 완료 ({time.time() - t0:.0f}s)", flush=True)
    dino = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval()
    dproc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    out = []
    with torch.no_grad():
        for imgs in batches():
            h = dino(**{k: v.to(dev) for k, v in dproc(images=imgs, return_tensors="pt").items()}).last_hidden_state[:, 0]
            out.append(h.float().cpu().numpy())
    res["dinov2"] = np.concatenate(out)
    del dino
    print(f"  DINOv2 완료 ({time.time() - t0:.0f}s)", flush=True)
    rn = AutoModel.from_pretrained("microsoft/resnet-50").to(dev).eval()
    rproc = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    out = []
    with torch.no_grad():
        for imgs in batches():
            h = rn(**{k: v.to(dev) for k, v in rproc(images=imgs, return_tensors="pt").items()}).pooler_output
            out.append(h.flatten(1).float().cpu().numpy())
    res["resnet50"] = np.concatenate(out)
    print(f"  ResNet-50 완료 ({time.time() - t0:.0f}s)", flush=True)
    for k, p in cache.items():
        np.save(p, res[k].astype(np.float32))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmri-dir")
    ap.add_argument("--emb-dir")
    ap.add_argument("--bundle-dir")
    ap.add_argument("--img-dir", default="data/raw/images")
    ap.add_argument("--out-dir", default="data/results/target_type_compare")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    cap = {r["cocoId"]: r for r in map(json.loads, gzip.open(Path(a.bundle_dir) / "captions.jsonl.gz", "rt"))}
    lab = {r["cocoId"]: r for r in map(json.loads, gzip.open(Path(a.bundle_dir) / "labels_pass0.jsonl.gz", "rt"))}
    ids_all = json.load(open(Path(a.emb_dir) / "ctx_ids.json"))
    train, Y, masks, nkeep, nvox = ps.load_variant(a.fmri_dir, ids_all, True, "rel", log=lambda s: print(s, flush=True))
    print(f"타깃 종류 비교: 이미지 {len(train)}장, 복셀 {Y.shape[1]}", flush=True)
    emb = embed_images(train, a.img_dir, a.emb_dir)
    counts = item_counts(cap, lab, train)
    F = load_features(a.emb_dir, "CLIP", ids_all, train)
    F.pop("ctx_V3", None)
    S = build_sets(F, counts, 0)
    T = {r: S[r].astype(np.float32) for r in TEXT_TARGETS}
    T.update({k: emb[k].astype(np.float32) for k in IMG_TARGETS})
    perm = np.random.default_rng(11).permutation(len(train))
    for k in IMG_TARGETS:
        T[f"{k}_shuf"] = T[k][perm]
    T["ctx_P1_shuf"] = S["ctx_P1_shuf"].astype(np.float32)
    out = td.run(Y, {"T": T}, B=a.boot, log=lambda s: print(s, flush=True))
    res = td.summarize(out, B=a.boot)
    res["n_voxels"] = int(Y.shape[1])
    o = Path(a.out_dir)
    o.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(o / "target_type_compare.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    np.savez_compressed(o / "target_type_scores.npz", **{f"{r}__{k}": v[k] for r, v in out["T"].items() for k in ("c1", "c5")})
    print({r: (round(v["top1"], 4), round(v["top5"], 4)) for r, v in res["reps"]["T"].items()})


if __name__ == "__main__":
    main()
