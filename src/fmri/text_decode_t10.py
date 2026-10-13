"""T10(맥락 구를 짧은 문장으로 연결, CLIP 75토큰 안)을 1단계 decoding 비교에 추가한다(CLIP만). 기준은 docs/stage1_text_repr_criteria.md의 K5.
text_decode_compare.py와 같은 폴드·같은 평가를 쓰므로 ctx_P1 등 겹치는 표현의 점수는 같아야 한다(확인용으로 함께 계산).
사용: python3 src/fmri/text_decode_t10.py --fmri-dir data/stage/fmri_subj01 --emb-dir data/embeddings_v2 --bundle-dir data/colab_bundle
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prep_sensitivity as ps  # noqa: E402
import text_decode_compare as td  # noqa: E402
from text_compare import build_sets, item_counts, load_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmri-dir")
    ap.add_argument("--emb-dir")
    ap.add_argument("--bundle-dir")
    ap.add_argument("--out-dir", default="data/results/text_decode_compare")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    cap = {r["cocoId"]: r for r in map(json.loads, gzip.open(Path(a.bundle_dir) / "captions.jsonl.gz", "rt"))}
    lab = {r["cocoId"]: r for r in map(json.loads, gzip.open(Path(a.bundle_dir) / "labels_pass0.jsonl.gz", "rt"))}
    ids_all = json.load(open(Path(a.emb_dir) / "ctx_ids.json"))
    train, Y, masks, nkeep, nvox = ps.load_variant(a.fmri_dir, ids_all, True, "rel", log=lambda s: print(s, flush=True))
    counts = item_counts(cap, lab, train)
    F = load_features(a.emb_dir, "CLIP", ids_all, train)
    F.pop("ctx_V3", None)
    S = build_sets(F, counts, 0)
    row = {c: i for i, c in enumerate(ids_all)}
    T10 = np.load(Path(a.emb_dir) / "CLIP__ctx_T10.npy").astype(np.float32)[[row[c] for c in train]]
    reps = {"orig_P1": S["orig_P1"], "obj_P1": S["obj_P1"], "ctx_P1": S["ctx_P1"], "ctx_V2": S["ctx_V2"], "ctx_T10": T10}
    S_by_enc = {"CLIP": {k: v.astype(np.float32) for k, v in reps.items()}}
    out = td.run(Y, S_by_enc, B=a.boot, log=lambda s: print(s, flush=True))
    res = td.summarize(out, B=a.boot)
    o = Path(a.out_dir)
    o.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(o / "text_decode_t10.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    np.savez_compressed(o / "text_decode_t10_scores.npz", **{f"CLIP__{r}__{k}": v[k] for r, v in out["CLIP"].items() for k in ("c1", "c5")})
    print({r: round(v["top1"], 4) for r, v in res["reps"]["CLIP"].items()})
    print(json.dumps(res["contrasts"]["CLIP"], ensure_ascii=False))


if __name__ == "__main__":
    main()
