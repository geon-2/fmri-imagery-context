"""Train.ipynb가 저장한 *_embeddings_train_{N}sess.pt 를 recon 노트북이 읽는 접미사 없는 이름으로 연결(업스트림 불일치, Colab 실측)."""
import glob, os, re
ROOT=os.environ["ROOT"]; SUBJ=os.environ["SUBJ"]
pre=f"{ROOT}/dataset/preprocessed_data/subject{SUBJ}"
NS=float(os.environ["NUM_SESSIONS"])
for f in sorted(glob.glob(f"{pre}/*_embeddings_train_*sess.pt")):
    m=re.search(r"_train_([\d.]+)sess\.pt$",f)
    if float(m.group(1))!=NS: continue          # 다른 세션 수의 임베딩은 연결하지 않음(여러 설정이 공존할 때 잘못 연결되는 것을 막음)
    dst=re.sub(r"_train_[\d.]+sess\.pt$","_train.pt",f)
    if os.path.lexists(dst): os.remove(dst)
    os.symlink(f,dst); print("link:",os.path.basename(dst))
need=["*_image_embeddings_train.pt","*_latent_embeddings_train.pt","*_text_embeddings_train.pt"]
if os.environ["USE_RETRIEVAL"]=="1": need.append("*_retrieval_embeddings_train.pt")
print("없는 것:",[n for n in need if not glob.glob(f"{pre}/{n}")] or "없음")
