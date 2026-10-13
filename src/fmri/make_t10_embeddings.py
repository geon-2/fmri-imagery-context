"""T10(맥락 구를 새 정보 없이 짧은 문장으로 연결, CLIP 77토큰 안) 임베딩을 만든다. 규칙은 docs/stage1_text_repr_criteria.md.

규칙: 구를 place_type, environment, activity, scale, implied_event, notes 순으로 ", "로 이어 붙이되, 이어 붙인 CLIP 토큰이 75(내용 토큰 한도)를 넘는
      구가 처음 나오면 그 구부터 뒤를 모두 버린다. 새 단어는 더하지 않는다.
검증: 같은 코드로 기존 ctx_P1, ctx_V2를 다시 만들어 저장된 CLIP 임베딩과 코사인을 비교한다(같은 모델·정규화인지 확인).
출력: data/embeddings_v2/CLIP__ctx_T10.npy (ctx_ids.json 순서, L2 정규화), data/results/text_decode_compare/t10_stats.json
사용: python3 src/fmri/make_t10_embeddings.py --emb-dir data/embeddings_v2 --bundle-dir data/colab_bundle
"""
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labeling"))
import text_variants as tv  # noqa: E402

ORDER = ("place_type", "environment", "activity", "scale", "implied_event", "notes")
BUDGET = 75
NAME = "openai/clip-vit-large-patch14"


def t10_text(by_dim, ntok):
    """우선순위 순서로 구를 이어 붙이고, 한도를 넘는 첫 구부터 뒤를 버린다. 반환 (문자열, 사용한 구 수, 전체 구 수)."""
    phrases = [p for d in ORDER for p in by_dim.get(d, [])]
    kept = []
    for p in phrases:
        cand = ", ".join(kept + [p])
        if ntok(cand) > BUDGET:
            break
        kept.append(p)
    return ", ".join(kept), len(kept), len(phrases)


@torch.no_grad()
def embed(model, tok, texts, bs=128):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=77, return_tensors="pt")
        f = model.get_text_features(**enc)
        f = f.pooler_output if hasattr(f, "pooler_output") else f
        out.append(torch.nn.functional.normalize(f, dim=-1).float().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", default="data/embeddings_v2")
    ap.add_argument("--bundle-dir", default="data/colab_bundle")
    ap.add_argument("--out-dir", default="data/results/text_decode_compare")
    a = ap.parse_args()
    t0 = time.time()
    ids = json.load(open(Path(a.emb_dir) / "ctx_ids.json"))
    lab = {r["cocoId"]: r for r in map(json.loads, gzip.open(Path(a.bundle_dir) / "labels_pass0.jsonl.gz", "rt"))}
    tok = CLIPTokenizer.from_pretrained(NAME)
    model = CLIPModel.from_pretrained(NAME).eval()
    ntok = lambda s: len(tok(s, add_special_tokens=False).input_ids)
    print(f"모델 로드 {time.time() - t0:.0f}s, 이미지 {len(ids)}장", flush=True)

    by = [tv.context_by_dim(lab[c]["labels"]) for c in ids]
    # --- 검증: 기존 ctx_P1, ctx_V2 재현
    rng = np.random.default_rng(0)
    pick = rng.choice(len(ids), 40, replace=False)
    ref1 = np.load(Path(a.emb_dir) / "CLIP__ctx_P1.npy").astype(np.float32)
    ref2 = np.load(Path(a.emb_dir) / "CLIP__ctx_V2.npy").astype(np.float32)
    cos1, cos2 = [], []
    for i in pick:
        units = tv.context_units(lab[ids[i]]["labels"])
        e = embed(model, tok, units)
        m = e.mean(0)
        m /= np.linalg.norm(m)
        cos1.append(float(m @ ref1[i] / np.linalg.norm(ref1[i])))
        v2 = tv.context_joined(lab[ids[i]]["labels"])
        e2 = embed(model, tok, [v2])[0]
        cos2.append(float(e2 @ ref2[i] / np.linalg.norm(ref2[i])))
    print(f"검증(40장): ctx_P1 재현 코사인 평균 {np.mean(cos1):.4f} 최소 {np.min(cos1):.4f}, ctx_V2 {np.mean(cos2):.4f} 최소 {np.min(cos2):.4f}", flush=True)

    # --- T10 만들기
    texts, used, total, v2_tokens = [], [], [], []
    for b, c in zip(by, ids):
        s, k, n = t10_text(b, ntok)
        texts.append(s)
        used.append(k)
        total.append(n)
        v2_tokens.append(ntok(tv.context_joined(lab[c]["labels"])))
    used, total, v2_tokens = np.array(used), np.array(total), np.array(v2_tokens)
    dropped = 1 - used.sum() / total.sum()
    stats = {"n_images": len(ids), "model": NAME, "budget": BUDGET, "validation_cos_ctx_P1_mean": float(np.mean(cos1)), "validation_cos_ctx_P1_min": float(np.min(cos1)),
             "validation_cos_ctx_V2_mean": float(np.mean(cos2)), "validation_cos_ctx_V2_min": float(np.min(cos2)),
             "t10_phrase_drop_rate": float(dropped), "t10_images_with_drop": float((used < total).mean()),
             "t10_mean_phrases_kept": float(used.mean()), "t10_mean_phrases_total": float(total.mean()),
             "v2_images_over_75_tokens": float((v2_tokens > BUDGET).mean()), "v2_mean_tokens": float(v2_tokens.mean())}
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    E = embed(model, tok, texts)
    np.save(Path(a.emb_dir) / "CLIP__ctx_T10.npy", E.astype(np.float32))
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    json.dump({**stats, "examples": [{"id": ids[i], "t10": texts[i], "kept": int(used[i]), "total": int(total[i])} for i in pick[:5]]},
              open(Path(a.out_dir) / "t10_stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"저장 CLIP__ctx_T10.npy {E.shape} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
