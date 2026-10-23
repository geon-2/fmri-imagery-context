"""Train.ipynb가 저장한 *_embeddings_train_{N}sess.pt 를 recon 노트북이 읽는 접미사 없는 이름으로 연결(업스트림 불일치, Colab 실측)."""
import glob, os, re
ROOT=os.environ["ROOT"]; SUBJ=os.environ["SUBJ"]
pre=f"{ROOT}/dataset/preprocessed_data/subject{SUBJ}"
for f in sorted(glob.glob(f"{pre}/*_embeddings_train_*sess.pt")):
    dst=re.sub(r"_train_[\d.]+sess\.pt$","_train.pt",f)
    if os.path.lexists(dst): os.remove(dst)
    os.symlink(f,dst); print("link:",os.path.basename(dst))
need=["*_image_embeddings_train.pt","*_latent_embeddings_train.pt","*_text_embeddings_train.pt"]
if os.environ["USE_RETRIEVAL"]=="1": need.append("*_retrieval_embeddings_train.pt")
print("없는 것:",[n for n in need if not glob.glob(f"{pre}/{n}")] or "없음")
