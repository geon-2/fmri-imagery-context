"""Colab 노트북의 patch_notebook/run_nb를 서버용으로 옮긴 것. 사용: python run_nb.py train|recon"""
import json, os, re, subprocess, sys
ROOT=os.environ["ROOT"]; SRC=f"{ROOT}/src"; PY=os.environ["PY"]
SUBJ=os.environ["SUBJ"]; NS=os.environ["NUM_SESSIONS"]; MODEL=os.environ["MODEL_NAME"]
RET="--retrieval" if os.environ["USE_RETRIEVAL"]=="1" else "--no-retrieval"
common=["--dual_guidance","--no-prompt_recon","--blurry_recon",RET]
def patch_notebook(src, dst, repl):
    nb=json.load(open(src)); hit=0
    for c in nb["cells"]:
        if c["cell_type"]!="code": continue
        s="".join(c["source"])
        if "jupyter_args" in s and 'f"' in s:
            s2=re.sub(r'jupyter_args\s*=\s*f"[^"]*"', lambda m:"jupyter_args = "+repr(repl), s, flags=re.S)
            if s2!=s: c["source"]=s2.splitlines(keepends=True); hit+=1
    json.dump(nb,open(dst,"w")); print("patched cells:",hit); assert hit==1
def run_nb(path,out):
    cmd=[f"{os.path.dirname(PY)}/jupyter","nbconvert","--to","notebook","--execute",path,"--output",out,"--allow-errors",
         "--ExecutePreprocessor.kernel_name=mei-env","--ExecutePreprocessor.timeout=-1"]
    r=subprocess.run(cmd,cwd=SRC,capture_output=True,text=True)
    if r.returncode: print("nbconvert 자체 실패:\n"+r.stderr[-3000:])
    outp=os.path.join(SRC,out); bad=0
    if os.path.exists(outp):
        for i,c in enumerate(json.load(open(outp))["cells"]):
            for o in c.get("outputs",[]):
                if o.get("output_type")=="error":
                    bad+=1
                    if bad==1:
                        print(f"❌ 첫 오류: 셀 {i}\n"+"".join(c["source"])[:800]); print("---\n"+o["ename"]+": "+o["evalue"][:600]); print("\n".join(o["traceback"])[-2500:])
        print("오류 셀 수:",bad)
    return bad if os.path.exists(outp) else (r.returncode or 1)
STATS_HELPER = r'''
# [stats] 학습 임베딩 전체(40세션 retrieval이면 약 27GB)를 저장하는 대신, recon이 쓰는 평균/표준편차만 저장한다.
# recon은 torch.mean/torch.std(axis=0)만 쓴다. 표본 2개 x1=mean+std/sqrt2, x2=mean-std/sqrt2 는 평균과 불편 표준편차가 원본과 같다.
def _save_stats(t, path):
    import torch
    n = t.shape[0]; s = torch.zeros(t.shape[1:], dtype=torch.float64); ss = torch.zeros_like(s)
    for i in range(0, n, 1000):
        c = t[i:i + 1000].double(); s += c.sum(0); ss += (c * c).sum(0)
    mean = s / n; std = ((ss - n * mean * mean) / (n - 1)).clamp_min(0).sqrt(); d = std / (2 ** 0.5)
    torch.save(torch.stack([mean + d, mean - d]).float(), path.replace("_embeddings_train_", "_embeddings_stats_"))
'''
def use_stats_embeddings(nb_path):
    nb = json.load(open(nb_path)); hit = 0
    reps = [("torch.save(clip_image_train, file_path)", "_save_stats(clip_image_train, file_path)"),
            ("torch.save(clip_text_train, file_path_txt)", "_save_stats(clip_text_train, file_path_txt)"),
            ("torch.save(vae_image_train, file_path)", "_save_stats(vae_image_train, file_path)"),
            ("torch.save(retrieval_image_train, file_path)", "_save_stats(retrieval_image_train, file_path)")]
    for c in nb["cells"]:
        t = "".join(c["source"])
        if c["cell_type"] == "code" and "Generating Image embeddings!" in t:
            for a, b in reps:
                assert t.count(a) == 1, f"저장 호출을 찾지 못함: {a}"; t = t.replace(a, b)
            c["source"] = (STATS_HELPER + t).splitlines(keepends=True); hit += 1
    assert hit == 1, f"임베딩 셀을 찾지 못함({hit})"
    json.dump(nb, open(nb_path, "w")); print("임베딩을 통계 파일로 저장하도록 패치")
LEAN_CELL = r'''
# [lean ridge] 원본 릿지 셀을 메모리 절약 구현으로 교체(수식은 sklearn Ridge와 같음. scripts/mirage_server/lean_ridge.py)
import sys, os, gc, pickle, time
sys.path.insert(0, os.environ["SCRIPTS"])
from lean_ridge import lean_ridge
del train_images; gc.collect()          # 임베딩을 다 만든 뒤에는 필요 없음(40세션이면 약 18GB)
start = time.time()
def _fit(name, Y, X):
    t0 = time.time(); print(f"lean ridge: {name} Y={tuple(Y.shape)}", flush=True)
    coef, icpt = lean_ridge(X, Y.reshape(len(Y), -1), weight_decay, device="cuda")
    with open(f"{outdir}/ridge_{name}_weights.pkl", "wb") as f: pickle.dump({"coef": coef, "intercept": icpt}, f)
    print(f"  saved ridge_{name}_weights.pkl coef{coef.shape} {coef.dtype}, {time.time()-t0:.0f}s", flush=True)
    del coef, icpt; gc.collect()
_fit("image", clip_image_train, x_train); del clip_image_train
if dual_guidance: _fit("text", clip_text_train, x_train); del clip_text_train
if blurry_recon: _fit("blurry", vae_image_train, x_train); del vae_image_train
if retrieval:
    import torch
    _fit("retrieval", retrieval_image_train, torch.nn.functional.normalize(x_train, p=2, dim=1)); del retrieval_image_train
assert not prompt_recon, "lean ridge는 prompt_recon을 지원하지 않음"
print(f"Elapsed training time for {model_name}: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")
'''
def use_lean_ridge(nb_path):
    nb = json.load(open(nb_path)); hit = 0
    for c in nb["cells"]:
        if c["cell_type"] == "code" and "Training Ridge Image model" in "".join(c["source"]):
            c["source"] = LEAN_CELL.strip().splitlines(keepends=True); hit += 1
    assert hit == 1, f"릿지 셀을 찾지 못함({hit})"
    json.dump(nb, open(nb_path, "w")); print("lean ridge 셀로 교체")
CTX_CELL = r'''
# [ctx blend] 맥락 릿지(ridge_ctx_weights.pkl)의 예측을 LLaVA 텍스트 예측(pred_clip_text)과 임베딩 공간에서 가중 평균한다.
# pred = (1-a)*LLaVA + a*ctx. 맥락 예측도 원본 텍스트 예측과 똑같이 테스트 표본끼리 표준화한 뒤 학습 평균·표준편차로 재스케일한다.
# 실패하면 LLaVA만 쓴 결과가 다른 조건 이름으로 저장되고 생성에 몇 시간을 쓰게 되므로, 예외가 나면 커널을 즉시 종료해 작업을 바로 실패시킨다.
try:
    import pickle, os, torch
    from sklearn.linear_model import Ridge as _R
    _w = pickle.load(open(os.environ["CTX_WEIGHTS"], "rb"))
    _m = _R(alpha=100000, max_iter=50000, random_state=42); _m.coef_ = _w["coef"]; _m.intercept_ = _w["intercept"]
    pred_ctx = torch.from_numpy(_m.predict(voxels[:,0]).reshape(-1, clip_text_seq_dim, clip_text_emb_dim)); del _w, _m
    _st = torch.load(os.environ["CTX_STATS"])
    if normalize_preds:
        pred_ctx = (pred_ctx - torch.mean(pred_ctx,axis=0)) / (torch.std(pred_ctx,axis=0) + 1e-6) * torch.std(_st,axis=0) + torch.mean(_st,axis=0)
    _a = float(os.environ["CTX_ALPHA"]); assert 0.0 <= _a <= 1.0 and dual_guidance
    _before = pred_clip_text.clone()
    pred_clip_text = (1 - _a) * pred_clip_text + _a * pred_ctx
    assert pred_clip_text.shape == _before.shape and torch.isfinite(pred_clip_text).all(), "맥락 합치기 결과가 비정상"
    _cos = torch.nn.functional.cosine_similarity(_before.reshape(len(_before),-1), pred_ctx.reshape(len(pred_ctx),-1), dim=1)
    print(f"[ctx blend] alpha={_a} pred_ctx{tuple(pred_ctx.shape)} LLaVA-ctx 코사인(자극별 평균) {_cos.mean():.4f}, 합친 뒤 LLaVA 대비 변화 {(pred_clip_text-_before).norm()/_before.norm():.4f}", flush=True)
    del _before, pred_ctx, _st
except BaseException:
    import traceback, os, sys
    traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); os._exit(3)
'''
def use_ctx_blend(nb_path):
    nb = json.load(open(nb_path)); idx = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and "std_pred_clip_text" in "".join(c["source"])]
    assert len(idx) == 1, f"텍스트 예측 정규화 셀을 찾지 못함({len(idx)})"
    cell = {"cell_type": "code", "execution_count": None, "id": "ctxblend01", "metadata": {}, "outputs": [], "source": CTX_CELL.strip().splitlines(keepends=True)}
    nb["cells"].insert(idx[0] + 1, cell); json.dump(nb, open(nb_path, "w")); print("맥락 합치기 셀 삽입(셀", idx[0] + 1, ")")
which=sys.argv[1]
if which=="train":
    args=(f"--data_path=../dataset/ --cache_dir=../cache/ --model_name={MODEL} --subj={SUBJ} --num_sessions={NS} --caption_type {os.environ['CAPTION_TYPE']} "+" ".join(common))
    patch_notebook(f"{SRC}/Train.ipynb",f"{SRC}/Train_srv.ipynb",args)
    if os.environ.get("LEAN_RIDGE")=="1": use_lean_ridge(f"{SRC}/Train_srv.ipynb")
    if os.environ.get("STATS_EMB","1")=="1": use_stats_embeddings(f"{SRC}/Train_srv.ipynb")
    rc=run_nb("Train_srv.ipynb","Train_srv_out.ipynb")
else:
    args=(f"--data_path=../dataset --cache_dir=../cache --model_name={MODEL} --subj={SUBJ} --mode imagery --gen_rep={os.environ['GEN_REP']} "
          f"--num_images_per_sample={os.environ['N_IMG']} --num_trial_reps=16 --no-compile_models "+(f"--seed={os.environ['SEED']} " if os.environ.get("SEED") else "")+" ".join(common))
    patch_notebook(f"{SRC}/recon_inference_mi.ipynb",f"{SRC}/recon_srv.ipynb",args)
    # retrieval pool: 원본은 8장씩 한 번에 생성(n_samples=8)한다. 12GB GPU에서는 배치 8이 메모리를 넘길 수 있어 RECON_BS(기본 1)장씩 나눠 생성(후보 수는 그대로)
    BS=os.environ.get("RECON_BS","1"); nb=json.load(open(f"{SRC}/recon_srv.ipynb")); hit=0
    for c in nb["cells"]:
        t="".join(c["source"])
        if "num_images_per_sample // 8" in t:
            t=t.replace("num_images_per_sample // 8",f"num_images_per_sample // {BS}").replace("min(8, num_images_per_sample)",f"{BS}")
            c["source"]=t.splitlines(keepends=True); hit+=1
    assert hit==1, f"생성 셀을 찾지 못함({hit})"; json.dump(nb,open(f"{SRC}/recon_srv.ipynb","w")); print("recon 배치 크기:",BS)
    if os.environ.get("CTX_WEIGHTS"): use_ctx_blend(f"{SRC}/recon_srv.ipynb")   # 맥락 릿지를 합치는 조건(CTX_WEIGHTS, CTX_STATS, CTX_ALPHA)
    rc=run_nb("recon_srv.ipynb","recon_srv_out.ipynb")
print("오류 셀 수(rc):",rc); sys.exit(1 if rc else 0)
