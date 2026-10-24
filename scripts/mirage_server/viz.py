"""생성 결과 그림 + CLIP 식별 정확도(18개 중 정답 찾기, 우연 1/18). 노드(GPU)에서 실행."""
import os, torch, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from transformers import CLIPModel, CLIPProcessor
ROOT=os.environ["ROOT"]; SRC=f"{ROOT}/src"; MODEL=os.environ["MODEL_NAME"]; OUT=os.environ["RESULTS"]; os.makedirs(OUT,exist_ok=True)
EV=f"{SRC}/evals/{MODEL}"
recons=torch.load(f"{EV}/{MODEL}_all_recons_imagery.pt",map_location="cpu")
bp=f"{EV}/{MODEL}_all_blurryrecons_imagery.pt"; blurry=torch.load(bp,map_location="cpu") if os.path.exists(bp) else None
gt=torch.load(f"{ROOT}/dataset/nsddata_stimuli/stimuli/imagery_stimuli_18.pt",map_location="cpu")
if recons.dim()==5: recons=recons[:,0]
if blurry is not None:
    # 저수준(blurry) 재구성은 recon 노트북이 vstack(dim 0)으로 쌓아 (N*3, H, W)로 저장됨(N_IMG=1 경로). (N,gen_rep,3,H,W)인 경우도 처리
    if blurry.dim()==5: blurry=blurry[:,0]
    elif blurry.dim()==3: blurry=blurry.reshape(-1,3,*blurry.shape[-2:])
    elif blurry.dim()==4 and blurry.shape[1]==1: blurry=blurry[:,0]
    print("blurry",tuple(blurry.shape))
print("recons",tuple(recons.shape),"gt",tuple(gt.shape))
def to_img(t):
    t=t.float()
    assert t.dim()==3 and t.shape[0]==3, f"예상 밖 형태 {tuple(t.shape)}"
    if t.min()<0: t=(t+1)/2
    return t.clamp(0,1).permute(1,2,0).numpy()
n=min(len(recons),len(gt)); rows=3 if blurry is not None else 2
fig,axs=plt.subplots(rows,n,figsize=(1.6*n,1.8*rows))
for i in range(n):
    axs[0,i].imshow(to_img(gt[i])); axs[1,i].imshow(to_img(recons[i]))
    if blurry is not None: axs[2,i].imshow(to_img(blurry[i]))
    for r in range(rows): axs[r,i].axis("off")
plt.tight_layout(); plt.savefig(f"{OUT}/recon_grid_{MODEL}.png",dpi=150)
dev="cuda"; cm=CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval(); pr=CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
@torch.no_grad()
def emb(ts):
    ims=[(to_img(t)*255).astype(np.uint8) for t in ts]
    f=cm.get_image_features(**pr(images=ims,return_tensors="pt").to(dev))
    return torch.nn.functional.normalize(f,dim=-1).cpu()
S=emb(recons[:n])@emb(gt[:n]).T
print(f"재구성→정답 top-1 {(S.argmax(1)==torch.arange(n)).float().mean():.3f}, 정답→재구성 {(S.argmax(0)==torch.arange(n)).float().mean():.3f}, 우연 {1/n:.3f}")
