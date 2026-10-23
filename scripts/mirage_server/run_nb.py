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
which=sys.argv[1]
if which=="train":
    args=(f"--data_path=../dataset/ --cache_dir=../cache/ --model_name={MODEL} --subj={SUBJ} --num_sessions={NS} --caption_type {os.environ['CAPTION_TYPE']} "+" ".join(common))
    patch_notebook(f"{SRC}/Train.ipynb",f"{SRC}/Train_srv.ipynb",args); rc=run_nb("Train_srv.ipynb","Train_srv_out.ipynb")
else:
    args=(f"--data_path=../dataset --cache_dir=../cache --model_name={MODEL} --subj={SUBJ} --mode imagery --gen_rep={os.environ['GEN_REP']} "
          f"--num_images_per_sample={os.environ['N_IMG']} --num_trial_reps=16 --no-compile_models "+" ".join(common))
    patch_notebook(f"{SRC}/recon_inference_mi.ipynb",f"{SRC}/recon_srv.ipynb",args); rc=run_nb("recon_srv.ipynb","recon_srv_out.ipynb")
print("오류 셀 수(rc):",rc); sys.exit(1 if rc else 0)
