"""원본 캡션 + 맥락 라벨 결합 타깃 대 원본 캡션 단독. 기준은 실행 전에 docs/orig_ctx_combo_criteria.md에 기록했다.
사용: python3 src/fmri/orig_ctx_combo.py --emb-dir data/embeddings_v2 --bundle-dir data/colab_bundle --out-dir data/results/orig_ctx_combo
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prep_sensitivity as ps  # noqa: E402
import text_decode_compare as td  # noqa: E402
from text_compare import build_sets, item_counts, load_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir")
    ap.add_argument("--bundle-dir")
    ap.add_argument("--out-dir", default="data/results/orig_ctx_combo")
    ap.add_argument("--subjects", default="01,02,05,07")
    a = ap.parse_args()
    cap = {r["cocoId"]: r for r in map(json.loads, gzip.open(Path(a.bundle_dir) / "captions.jsonl.gz", "rt"))}
    lab = {r["cocoId"]: r for r in map(json.loads, gzip.open(Path(a.bundle_dir) / "labels_pass0.jsonl.gz", "rt"))}
    ids_all = json.load(open(Path(a.emb_dir) / "ctx_ids.json"))
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sid in a.subjects.split(","):
        train, Y, masks, nkeep, nvox = ps.load_variant(f"data/stage/fmri_subj{sid}", ids_all, True, "rel", log=lambda s: print(f"  [{sid}] {s}", flush=True))
        counts = item_counts(cap, lab, train)
        F = load_features(a.emb_dir, "CLIP", ids_all, train)
        F.pop("ctx_V3", None)
        S = build_sets(F, counts, 0)
        o, c, b = (S[k].astype(np.float32) for k in ("orig_P1", "ctx_P1", "obj_P1"))
        m = o + c
        m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
        perm = np.random.default_rng(5).permutation(len(train))
        T = {"orig_P1": o, "ctx_P1": c, "obj_P1": b, "orig+ctx_concat": np.concatenate([o, c], axis=1), "orig+ctx_mean": m.astype(np.float32)}
        T["orig+ctx_concat_shuf"] = T["orig+ctx_concat"][perm]
        out = td.run(Y, {"T": T}, B=1, log=lambda s: print(f"  [{sid}] {s}", flush=True))
        np.savez_compressed(out_dir / f"orig_ctx_combo_{sid}.npz", **{f"{r}__{k}": v[k] for r, v in out["T"].items() for k in ("c1", "c5")})
        print(f"[{sid}] 완료", {r: round(float(v['c1'].mean()), 4) for r, v in out['T'].items()}, flush=True)
    print("완료", flush=True)


if __name__ == "__main__":
    main()
