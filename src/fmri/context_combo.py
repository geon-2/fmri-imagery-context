"""A 단계: 맥락 정보 조합 비교(차원 선택, 라벨 일부 사용), 피험자 1·2·5·7. 기준은 실행 전에 docs/stageA_context_combo_criteria.md와 Notion에 기록했다.
조건: all6(기준), 원본 캡션 P1, 사물 이름 P1, 단일 차원 6, 한 차원 제외 6, 두 차원 조합 15, 라벨 절반(시드 5), 라벨 1개(시드 5), 셔플 대조.
사용: python3 src/fmri/context_combo.py --stage-root data/stage --emb-dir data/embeddings_v2 --bundle-dir data/colab_bundle --out-dir data/results/context_combo [--fast]
"""
import argparse
import gzip
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import axes_decode as ad  # noqa: E402
import embedding_compare as ec  # noqa: E402
import prep_sensitivity as ps  # noqa: E402
import text_decode_compare as td  # noqa: E402
import text_variants as tv  # noqa: E402
from text_compare import build_sets, item_counts, load_features  # noqa: E402

DIMS = tuple(tv.DIMS)
NAME = "openai/clip-vit-large-patch14"
SEEDS = 5


def phrase_table(lab, ids, emb_dir):
    """맥락 구의 고유 문자열별 CLIP 텍스트 임베딩(캐시)."""
    cache = Path(emb_dir) / "CLIP__phrase_unique.npz"
    phrases = sorted({p for c in ids for d in DIMS for p in tv.context_by_dim(lab[c]["labels"]).get(d, [])})
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["phrases"]) == phrases:
            return {p: i for i, p in enumerate(phrases)}, z["emb"]
    import torch
    from transformers import CLIPModel, CLIPTokenizer
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = CLIPTokenizer.from_pretrained(NAME)
    model = CLIPModel.from_pretrained(NAME).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(phrases), 256):
            enc = tok(phrases[i:i + 256], padding=True, truncation=True, max_length=77, return_tensors="pt").to(dev)
            f = model.get_text_features(**enc)
            f = f.pooler_output if hasattr(f, "pooler_output") else f
            out.append(torch.nn.functional.normalize(f, dim=-1).float().cpu().numpy())
    emb = np.concatenate(out).astype(np.float32)
    np.savez(cache, phrases=np.array(phrases, dtype=object), emb=emb)
    print(f"  구 임베딩 {len(phrases)}개 저장", flush=True)
    return {p: i for i, p in enumerate(phrases)}, emb


def build_targets(lab, ids, pidx, pemb, extra):
    """조건 이름 → (n×768 타깃, n 유효 여부). extra는 {이름: n×768}(원본·사물)."""
    n = len(ids)
    by = [tv.context_by_dim(lab[c]["labels"]) for c in ids]
    idsets = [{d: [pidx[p] for p in b.get(d, [])] for d in DIMS} for b in by]
    T, V = {}, {}

    def make(name, pick):
        M = np.zeros((n, pemb.shape[1]), np.float32)
        v = np.zeros(n, bool)
        for i in range(n):
            sel = pick(idsets[i], i)
            if sel:
                m = pemb[sel].mean(0)
                M[i] = m / (np.linalg.norm(m) + 1e-9)
                v[i] = True
        if v.any():
            M[~v] = M[v].mean(0)
        T[name], V[name] = M, v

    make("all6", lambda s, i: [x for d in DIMS for x in s[d]])
    for k, d in enumerate(DIMS):
        make(f"single_{d}", lambda s, i, d=d: list(s[d]))
        make(f"loo_{d}", lambda s, i, d=d: [x for e in DIMS if e != d for x in s[e]])
    for d1, d2 in itertools.combinations(DIMS, 2):
        make(f"pair_{d1}+{d2}", lambda s, i, d1=d1, d2=d2: list(s[d1]) + list(s[d2]))
    for sd in range(SEEDS):
        def half(s, i, sd=sd):
            r = np.random.default_rng(1000 * sd + i)
            out = []
            for d in DIMS:
                L = s[d]
                if L:
                    out += list(r.choice(L, max(1, (len(L) + 1) // 2), replace=False))
            return out

        def one(s, i, sd=sd):
            r = np.random.default_rng(777 + 1000 * sd + i)
            return [int(r.choice(s[d])) for d in DIMS if s[d]]
        make(f"half_s{sd}", half)
        make(f"one_s{sd}", one)
    for k, M in extra.items():
        T[k], V[k] = M.astype(np.float32), np.ones(n, bool)
    perm = np.random.default_rng(3).permutation(n)
    T["all6_shuf"], V["all6_shuf"] = T["all6"][perm], V["all6"][perm]
    return T, V


def run_subject(sid, a, lab, cap, ids_all, fast):
    fdir = Path(a.stage_root) / f"fmri_subj{sid}"
    train, Y, masks, nkeep, nvox = ps.load_variant(fdir, ids_all, True, "rel", log=lambda s: print(f"  [{sid}] {s}", flush=True))
    n0 = len(train)
    pidx, pemb = phrase_table(lab, ids_all, a.emb_dir)
    counts = item_counts(cap, lab, train)
    F = load_features(a.emb_dir, "CLIP", ids_all, train)
    F.pop("ctx_V3", None)
    S = build_sets(F, counts, 0)
    T, V = build_targets(lab, train, pidx, pemb, {"orig_P1": S["orig_P1"], "obj_P1": S["obj_P1"]})
    n = n0
    if fast:
        n = 600
        Y = Y[:n]
        T = {k: v[:n] for k, v in T.items()}
        V = {k: v[:n] for k, v in V.items()}
    names = list(T)
    print(f"[{sid}] 이미지 {n}장, 복셀 {Y.shape[1]}, 조건 {len(names)}개, 유효 이미지 수 예: all6 {int(V['all6'].sum())}, one_s0 {int(V['one_s0'].sum())}", flush=True)
    folds = ec.outer_folds(n, 0)
    if fast:
        folds = folds[:2]
    out = {k: {"c1": np.full(n, np.nan, np.float32), "c5": np.full(n, np.nan, np.float32), "ceil": np.full(n, np.nan, np.float32)} for k in names}
    t0 = time.time()
    for fi, f in enumerate(folds):
        tr = np.setdiff1d(np.arange(n), f)
        Ztr = np.concatenate([T[k][tr] for k in names], axis=1)
        Zte = {k: T[k][f] for k in names}
        Zh = ad.decode(Y[tr], Ztr, Y[f], seed=fi)
        c = 0
        for k in names:
            d = T[k].shape[1]
            v = V[k][f]
            if v.sum() > 20:
                zh, zt = Zh[:, c:c + d][v], Zte[k][v]
                c1, c5 = td.credit(ad.sim_matrix(zh, zt))
                cc1, _ = td.credit(ad.sim_matrix(zt, zt))
                fv = f[v]
                out[k]["c1"][fv], out[k]["c5"][fv], out[k]["ceil"][fv] = c1, c5, cc1
            c += d
        print(f"  [{sid}] 폴드 {fi + 1}/{len(folds)} ({time.time() - t0:.0f}s)", flush=True)
    o = Path(a.out_dir)
    o.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(o / f"context_combo_{sid}{'_fast' if fast else ''}.npz", **{f"{k}__{m}": out[k][m] for k in names for m in ("c1", "c5", "ceil")},
                        **{f"valid__{k}": V[k] for k in names})
    json.dump({"n": n, "n_voxels": int(Y.shape[1]), "conditions": names}, open(o / f"context_combo_{sid}{'_fast' if fast else ''}_meta.json", "w"), ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-root", default="data/stage")
    ap.add_argument("--emb-dir")
    ap.add_argument("--bundle-dir")
    ap.add_argument("--out-dir", default="data/results/context_combo")
    ap.add_argument("--subjects", default="01,02,05,07")
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    cap = {r["cocoId"]: r for r in map(json.loads, gzip.open(Path(a.bundle_dir) / "captions.jsonl.gz", "rt"))}
    lab = {r["cocoId"]: r for r in map(json.loads, gzip.open(Path(a.bundle_dir) / "labels_pass0.jsonl.gz", "rt"))}
    ids_all = json.load(open(Path(a.emb_dir) / "ctx_ids.json"))
    for sid in a.subjects.split(","):
        run_subject(sid, a, lab, cap, ids_all, a.fast)
    print("완료", flush=True)


if __name__ == "__main__":
    main()
