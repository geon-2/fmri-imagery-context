"""1단계 텍스트 표현 비교를 다른 전처리 복셀 선택(d = 영역별 SNR 상위 20%)으로 반복한다. 기준: docs/stage1_prepvar_criteria.md.
text_decode_compare.py와 같은 폴드·평가. CLIP은 T10까지 9개 표현, MPNet은 8개 표현.
사용: python3 src/fmri/text_decode_prepvar.py --fmri-dir data/stage/fmri_subj01 --emb-dir data/embeddings_v2 --bundle-dir data/colab_bundle --select snr20 --out-dir data/results/text_decode_prepvar_d
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
    ap.add_argument("--select", default="snr20")
    ap.add_argument("--out-dir", default="data/results/text_decode_prepvar_d")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    cap = {r["cocoId"]: r for r in map(json.loads, gzip.open(Path(a.bundle_dir) / "captions.jsonl.gz", "rt"))}
    lab = {r["cocoId"]: r for r in map(json.loads, gzip.open(Path(a.bundle_dir) / "labels_pass0.jsonl.gz", "rt"))}
    ids_all = json.load(open(Path(a.emb_dir) / "ctx_ids.json"))
    train, Y, masks, nkeep, nvox = ps.load_variant(a.fmri_dir, ids_all, True, a.select, log=lambda s: print(s, flush=True))
    counts = item_counts(cap, lab, train)
    row = {c: i for i, c in enumerate(ids_all)}
    S_by_enc = {}
    for enc in td.ENCODERS:
        F = load_features(a.emb_dir, enc, ids_all, train)
        F.pop("ctx_V3", None)
        S = {r: v.astype(np.float32) for r, v in build_sets(F, counts, 0).items() if r in td.REPS}
        if enc == "CLIP":
            S["ctx_T10"] = np.load(Path(a.emb_dir) / "CLIP__ctx_T10.npy").astype(np.float32)[[row[c] for c in train]]
        S_by_enc[enc] = S
    print(f"1단계(전처리 {a.select}): 이미지 {len(train)}장, 복셀 {Y.shape[1]}", flush=True)
    out = td.run(Y, S_by_enc, B=a.boot, log=lambda s: print(s, flush=True))
    res = td.summarize(out, B=a.boot)
    res["n_voxels"] = int(Y.shape[1])
    o = Path(a.out_dir)
    o.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(o / "text_decode_prepvar.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    np.savez_compressed(o / "text_decode_prepvar_scores.npz", **{f"{e}__{r}__{k}": v[k] for e in out for r, v in out[e].items() for k in ("c1", "c5")})
    for enc in res["reps"]:
        print(enc, {r: round(v["top1"], 4) for r, v in res["reps"][enc].items()})
    print(json.dumps(res["encoder_rule"], ensure_ascii=False))


if __name__ == "__main__":
    main()
