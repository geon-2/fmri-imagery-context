"""MIRAGE/StableCascade가 import하는 모듈을 실제로 불러 보고, 없는 모듈이 나오면 알려진 방법으로 설치한다(Colab 노트북 7a 셀과 같은 방식)."""
import os, re, subprocess, sys
PY=os.environ["PY"]; ROOT=os.environ["ROOT"]; SRC=f"{ROOT}/src"
CONS=f"/data/{os.environ['USER']}/envs/constraints.txt"
SPEC={"warmup_scheduler":["git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git"],"torchtools":["git+https://github.com/pabloppp/pytorch-tools"],
 "clip":["--no-deps","git+https://github.com/openai/CLIP.git"],"wandb":["wandb==0.16.2"],"dalle2_pytorch":["dalle2-pytorch"],"cv2":["opencv-python==4.8.1.78"],
 "insightface":["insightface==0.7.3","onnxruntime==1.16.3","onnx2torch==1.5.13"],"onnx2torch":["onnx2torch==1.5.13"],"onnxruntime":["onnxruntime==1.16.3"],
 "PIL":["pillow"],"sklearn":["scikit-learn"],"skimage":["scikit-image==0.22.0"],"yaml":["pyyaml"]}
probe=f'''
import sys, os
sys.path.insert(0, "{SRC}"); sys.path.insert(0, "{SRC}/StableCascade"); os.chdir("{SRC}")
import torch
import utils, vdvae, modeling_git, sc_reconstructor
print("imports ok")
'''
open("/tmp/_probe_srv.py","w").write(probe)
for _ in range(15):
    r=subprocess.run([PY,"/tmp/_probe_srv.py"],capture_output=True,text=True,cwd=SRC)
    if r.returncode==0: print("✅",r.stdout.strip()); sys.exit(0)
    m=re.search(r"No module named '([^'.]+)",r.stderr)
    if not m: print("⚠️ 모듈 누락이 아닌 오류:\n"+r.stderr[-1800:]); sys.exit(1)
    mod=m.group(1); print("누락:",mod)
    ins=subprocess.run([PY,"-m","uv","pip","install","--python",PY,"-c",CONS]+SPEC.get(mod,[mod.replace("_","-")]),capture_output=True,text=True)
    print(("OK   " if ins.returncode==0 else "FAIL ")+mod)
    if ins.returncode: print(ins.stderr[-600:]); sys.exit(1)
print("⚠️ 15회 안에 해결 못 함"); sys.exit(1)
