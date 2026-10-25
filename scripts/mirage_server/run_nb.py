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
which=sys.argv[1]
if which=="train":
    args=(f"--data_path=../dataset/ --cache_dir=../cache/ --model_name={MODEL} --subj={SUBJ} --num_sessions={NS} --caption_type {os.environ['CAPTION_TYPE']} "+" ".join(common))
    patch_notebook(f"{SRC}/Train.ipynb",f"{SRC}/Train_srv.ipynb",args)
    if os.environ.get("LEAN_RIDGE")=="1": use_lean_ridge(f"{SRC}/Train_srv.ipynb")
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
    rc=run_nb("recon_srv.ipynb","recon_srv_out.ipynb")
print("오류 셀 수(rc):",rc); sys.exit(1 if rc else 0)
