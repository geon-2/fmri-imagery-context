"""Train.ipynb가 저장한 *_embeddings_train_{N}sess.pt 를 recon 노트북이 읽는 접미사 없는 이름으로 연결(업스트림 불일치, Colab 실측)."""
import glob, os, re
ROOT=os.environ["ROOT"]; SUBJ=os.environ["SUBJ"]
pre=f"{ROOT}/dataset/preprocessed_data/subject{SUBJ}"
NS=float(os.environ["NUM_SESSIONS"])
for kind in ("stats","train"):                 # stats(작은 파일) 우선, 없으면 전체 임베딩 파일
    for f in sorted(glob.glob(f"{pre}/*_embeddings_{kind}_*sess.pt")):
        m=re.search(r"_embeddings_%s_([\d.]+)sess\.pt$"%kind,f)
        if float(m.group(1))!=NS: continue          # 다른 세션 수의 임베딩은 연결하지 않음
        dst=re.sub(r"_embeddings_%s_[\d.]+sess\.pt$"%kind,"_embeddings_train.pt",f)
        if kind=="train" and os.path.lexists(dst) and os.path.islink(dst) and "_stats_" in os.readlink(dst): continue
        if os.path.lexists(dst): os.remove(dst)
        os.symlink(f,dst); print("link:",os.path.basename(dst),"->",os.path.basename(f))
need=["*_image_embeddings_train.pt","*_latent_embeddings_train.pt","*_text_embeddings_train.pt"]
if os.environ["USE_RETRIEVAL"]=="1": need.append("*_retrieval_embeddings_train.pt")
print("없는 것:",[n for n in need if not glob.glob(f"{pre}/{n}")] or "없음")
