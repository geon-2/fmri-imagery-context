"""맥락 문장(T10)을 타깃으로 하는 텍스트 릿지를 하나 더 학습한다(서버, mei-env). MIRAGE의 텍스트 릿지(LLaVA 캡션 타깃)는 그대로 두고 별개의 가중치를 만든다.
- 입력: 학습 beta(utils.load_nsd와 동일), NSD id별 맥락 문장 JSON(scripts/mirage_server/ctx/build_ctx_captions.py가 맥에서 만든 파일)
- 타깃: 문장을 Stable Cascade의 bigG 텍스트 인코더(SC_Reconstructor.embed_text)에 통과시킨 77x1280 은닉(원본 Train.ipynb와 같은 함수)
- 릿지: lean_ridge(α=1e5, sklearn Ridge와 같은 해). 출력: ridge_ctx_weights.pkl, 학습 타깃 평균·표준편차(2표본 통계 파일), (선택) 셔플 대조용 ridge_ctxshuf_weights.pkl
- 셔플 대조: 이미지 단위로 문장을 다른 이미지의 문장으로 바꿔(같은 이미지의 3회 반복은 같은 문장) 같은 방식으로 학습. 문장이 없는 이미지의 표본은 학습에서 제외한다(피험자 1: 3장).
환경변수: ROOT, SCRIPTS (env.sh), SUBJ, NUM_SESSIONS, CTX_JSON, BASE_MODEL, CTX_OUT, CTX_VARIANTS(기본 "ctx,shuf")"""
import gc, json, os, pickle, sys, time
import numpy as np


def split_samples(nsd_ids, texts):
    """학습 표본(이미지 id 배열)에서 문장이 있는 표본만 고르고 고유 이미지 목록과 역색인을 만든다. 반환 keep(bool), uniq(정렬된 id), inverse(keep 표본의 uniq 위치)."""
    nsd_ids = np.asarray(nsd_ids).astype(int)
    keep = np.array([str(i) in texts for i in nsd_ids])
    uniq, inverse = np.unique(nsd_ids[keep], return_inverse=True)
    return keep, uniq, inverse


def derangement(n, seed=0):
    """고정점이 없는 순열(모든 이미지가 다른 이미지의 문장을 받게 한다)."""
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            return p


def stats_two_samples(t, chunk=1000):
    """평균·불편 표준편차가 원본과 같은 2표본 텐서(run_nb.py의 _save_stats와 같은 방식)."""
    import torch
    n = t.shape[0]; s = torch.zeros(t.shape[1:], dtype=torch.float64); ss = torch.zeros_like(s)
    for i in range(0, n, chunk):
        c = t[i:i + chunk].double(); s += c.sum(0); ss += (c * c).sum(0)
    mean = s / n; std = ((ss - n * mean * mean) / (n - 1)).clamp_min(0).sqrt(); d = std / (2 ** 0.5)
    return torch.stack([mean + d, mean - d]).float()


def main():
    import torch
    ROOT = os.environ["ROOT"]; SRC = f"{ROOT}/src"
    sys.path.insert(0, SRC); sys.path.insert(0, f"{SRC}/StableCascade"); sys.path.insert(0, os.environ["SCRIPTS"])
    os.chdir(SRC)
    import utils
    from lean_ridge import lean_ridge
    from sc_reconstructor import SC_Reconstructor
    SUBJ = int(os.environ["SUBJ"]); NS = float(os.environ["NUM_SESSIONS"])
    OUT = os.environ["CTX_OUT"]; os.makedirs(OUT, exist_ok=True)
    variants = os.environ.get("CTX_VARIANTS", "ctx,shuf").split(",")
    texts = json.load(open(os.environ["CTX_JSON"]))
    t0 = time.time()
    x_train, ids, _, _ = utils.load_nsd(subject=SUBJ, num_sessions=NS, data_path=f"{ROOT}/dataset")
    keep, uniq, inverse = split_samples(ids, texts)
    print(f"학습 표본 {len(ids)}개(이미지 {len(np.unique(ids))}장), 문장 있는 표본 {int(keep.sum())}개(이미지 {len(uniq)}장), 제외 {int((~keep).sum())}개 | {time.time()-t0:.0f}s", flush=True)
    x = x_train[torch.from_numpy(keep)]; del x_train; gc.collect()

    rec = SC_Reconstructor(compile_models=False, embedder_only=True, device="cuda", cache_dir=f"{ROOT}/cache")
    E = torch.zeros((len(uniq), 77, 1280))
    for i in range(0, len(uniq), 50):
        E[i:i + 50] = rec.embed_text([texts[str(n)] for n in uniq[i:i + 50]]).float().cpu()
        if (i // 50) % 40 == 0: print(f"  문장 임베딩 {i}/{len(uniq)} | {time.time()-t0:.0f}s", flush=True)
    del rec; gc.collect(); torch.cuda.empty_cache()
    assert torch.isfinite(E).all(), "문장 임베딩에 NaN/inf"

    inv = torch.from_numpy(inverse)
    pre = f"{ROOT}/dataset/preprocessed_data/subject{SUBJ}"
    for v in variants:
        order = inv if v == "ctx" else torch.from_numpy(derangement(len(uniq)))[inv]
        Y = E[order].reshape(len(order), -1)                       # (표본, 77*1280)
        if v == "ctx":                                              # 정규화 통계는 정상 타깃에서 한 번만(셔플은 같은 표본 집합이라 같다)
            torch.save(stats_two_samples(Y.reshape(-1, 77, 1280)), f"{pre}/ctxT10_textstats_{int(NS)}sess.pt")
        t1 = time.time(); print(f"[{v}] lean ridge Y={tuple(Y.shape)}", flush=True)
        coef, icpt = lean_ridge(x, Y, 100000, device="cuda")
        name = "ctx" if v == "ctx" else "ctxshuf"
        with open(f"{OUT}/ridge_{name}_weights.pkl", "wb") as f: pickle.dump({"coef": coef, "intercept": icpt}, f)
        print(f"[{v}] saved ridge_{name}_weights.pkl coef{coef.shape} {coef.dtype} | {time.time()-t1:.0f}s", flush=True)
        del coef, icpt, Y; gc.collect()
    print(f"끝: 총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
