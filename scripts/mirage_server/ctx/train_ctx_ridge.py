"""맥락 텍스트를 타깃으로 하는 텍스트 릿지를 하나 더 학습한다(서버, mei-env). MIRAGE의 텍스트 릿지(LLaVA 캡션 타깃)는 그대로 두고 별개의 가중치를 만든다.
- 입력: 학습 beta(utils.load_nsd와 동일), NSD id별 맥락 문장 JSON(T10) 또는 구 목록 JSON(scripts/mirage_server/ctx/build_ctx_captions.py가 맥에서 만든 파일)
- 타깃(변형별): bigG 텍스트 인코더(SC_Reconstructor.embed_text, 원본 Train.ipynb와 같은 함수)의 77x1280 은닉
    ctx    : T10 문장(구를 이어 75토큰으로 자름) 하나를 인코딩
    shuf   : ctx와 같은데 이미지 단위로 문장을 다른 이미지의 것으로 바꿔 학습(대조)
    p1seq  : 잘림 없이 모든 구를 각각 따로 인코딩해서 토큰 위치별로 평균(지각 실험의 P1 평균 방식에 대응, 구는 각각 짧아 77토큰 한도에 걸리지 않음)
- 릿지: lean_ridge(α=1e5, sklearn Ridge와 같은 해). 출력: ridge_<이름>_weights.pkl, 학습 타깃 평균·표준편차(2표본 통계 파일)
- 문장/구가 없는 이미지의 표본은 학습에서 제외한다(피험자 1: 3장).
환경변수: ROOT, SCRIPTS (env.sh), SUBJ, NUM_SESSIONS, CTX_OUT, CTX_VARIANTS(기본 "ctx,shuf"; 예: "p1seq"),
          CTX_JSON(기본 $SCRIPTS/ctx/context_T10_by_nsdid.json), CTX_PHRASES(기본 $SCRIPTS/ctx/context_phrases_by_nsdid.json)"""
import gc, json, os, pickle, sys, time
import numpy as np


def split_samples(nsd_ids, texts):
    """학습 표본(이미지 id 배열)에서 텍스트가 있는 표본만 고르고 고유 이미지 목록과 역색인을 만든다. 반환 keep(bool), uniq(정렬된 id), inverse(keep 표본의 uniq 위치)."""
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


def p1seq_embeddings(embed_fn, phrases_by_id, uniq, batch=64, log=print):
    """이미지마다 구 각각의 시퀀스 임베딩(embed_fn: 문자열 리스트 -> (B,L,D))을 토큰 위치별로 평균한다. 모든 구를 쓴다(잘림 없음)."""
    import torch
    flat, owner = [], []
    for k, n in enumerate(uniq):
        for p in phrases_by_id[str(n)]:
            flat.append(p); owner.append(k)
    owner = torch.tensor(owner); out = None; cnt = torch.zeros(len(uniq))
    for i in range(0, len(flat), batch):
        e = embed_fn(flat[i:i + batch]).float().cpu()
        if out is None: out = torch.zeros((len(uniq),) + tuple(e.shape[1:]))
        idx = owner[i:i + batch]
        out.index_add_(0, idx, e); cnt.index_add_(0, idx, torch.ones(len(idx)))
        if (i // batch) % 200 == 0: log(f"  구 임베딩 {i}/{len(flat)}")
    assert (cnt > 0).all(), "구가 하나도 없는 이미지가 있음"
    return out / cnt.view(-1, *([1] * (out.dim() - 1)))


def main():
    import torch
    ROOT = os.environ["ROOT"]; SRC = f"{ROOT}/src"; SCRIPTS = os.environ["SCRIPTS"]
    sys.path.insert(0, SRC); sys.path.insert(0, f"{SRC}/StableCascade"); sys.path.insert(0, SCRIPTS)
    os.chdir(SRC)
    import utils
    from lean_ridge import lean_ridge
    from sc_reconstructor import SC_Reconstructor
    SUBJ = int(os.environ["SUBJ"]); NS = float(os.environ["NUM_SESSIONS"])
    OUT = os.environ["CTX_OUT"]; os.makedirs(OUT, exist_ok=True)
    variants = os.environ.get("CTX_VARIANTS", "ctx,shuf").split(",")
    assert all(v in ("ctx", "shuf", "p1seq") for v in variants), f"알 수 없는 변형: {variants}"
    need_t10 = any(v in ("ctx", "shuf") for v in variants); need_p1 = "p1seq" in variants
    texts = json.load(open(os.environ.get("CTX_JSON", f"{SCRIPTS}/ctx/context_T10_by_nsdid.json"))) if need_t10 else None
    phrases = json.load(open(os.environ.get("CTX_PHRASES", f"{SCRIPTS}/ctx/context_phrases_by_nsdid.json"))) if need_p1 else None
    avail = texts if texts is not None else phrases                 # 두 파일의 이미지 집합은 같다(같은 라벨에서 만듦)
    t0 = time.time()
    x_train, ids, _, _ = utils.load_nsd(subject=SUBJ, num_sessions=NS, data_path=f"{ROOT}/dataset")
    keep, uniq, inverse = split_samples(ids, avail)
    print(f"학습 표본 {len(ids)}개(이미지 {len(np.unique(ids))}장), 텍스트 있는 표본 {int(keep.sum())}개(이미지 {len(uniq)}장), 제외 {int((~keep).sum())}개 | {time.time()-t0:.0f}s", flush=True)
    x = x_train[torch.from_numpy(keep)]; del x_train; gc.collect()

    rec = SC_Reconstructor(compile_models=False, embedder_only=True, device="cuda", cache_dir=f"{ROOT}/cache")
    E = {}
    if need_t10:
        e = torch.zeros((len(uniq), 77, 1280))
        for i in range(0, len(uniq), 50):
            e[i:i + 50] = rec.embed_text([texts[str(n)] for n in uniq[i:i + 50]]).float().cpu()
            if (i // 50) % 40 == 0: print(f"  문장 임베딩 {i}/{len(uniq)} | {time.time()-t0:.0f}s", flush=True)
        E["t10"] = e
    if need_p1:
        E["p1"] = p1seq_embeddings(lambda b: rec.embed_text(b), phrases, uniq, log=lambda m: print(m, f"| {time.time()-t0:.0f}s", flush=True))
    del rec; gc.collect(); torch.cuda.empty_cache()
    for k, v in E.items(): assert torch.isfinite(v).all(), f"{k} 임베딩에 NaN/inf"

    inv = torch.from_numpy(inverse)
    pre = f"{ROOT}/dataset/preprocessed_data/subject{SUBJ}"
    for v in variants:
        if v == "ctx": base, order, name, stats = E["t10"], inv, "ctx", f"{pre}/ctxT10_textstats_{int(NS)}sess.pt"
        elif v == "shuf": base, order, name, stats = E["t10"], torch.from_numpy(derangement(len(uniq)))[inv], "ctxshuf", None   # 셔플은 같은 표본 집합이라 통계가 같다
        else: base, order, name, stats = E["p1"], inv, "ctxp1seq", f"{pre}/ctxP1seq_textstats_{int(NS)}sess.pt"
        Y = base[order].reshape(len(order), -1)                      # (표본, 77*1280)
        if stats: torch.save(stats_two_samples(Y.reshape(-1, 77, 1280)), stats)
        t1 = time.time(); print(f"[{v}] lean ridge Y={tuple(Y.shape)}", flush=True)
        coef, icpt = lean_ridge(x, Y, 100000, device="cuda")
        with open(f"{OUT}/ridge_{name}_weights.pkl", "wb") as f: pickle.dump({"coef": coef, "intercept": icpt}, f)
        print(f"[{v}] saved ridge_{name}_weights.pkl coef{coef.shape} {coef.dtype} | {time.time()-t1:.0f}s", flush=True)
        del coef, icpt, Y; gc.collect()
    print(f"끝: 총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
