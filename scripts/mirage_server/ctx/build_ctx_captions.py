"""MIRAGE의 텍스트 가지에 넣을 맥락 문장(T10)을 NSD id별 JSON으로 만든다(맥에서 실행, 서버로 올릴 작은 파일).
T10 규칙(src/fmri/make_t10_embeddings.py): 맥락 구를 place_type, environment, activity, scale, implied_event, notes 순으로 ", "로 이어 붙이고
CLIP 토큰이 75를 넘는 첫 구부터 뒤를 버린다. 새 단어를 더하지 않는다. 문장은 CLIP 토큰 기준 75 이하(bigG도 같은 BPE).
출력: data/results/mirage_ctx/context_T10_by_nsdid.json  ({nsdId(문자열): 문장}), ctx_T10_check.json (검증 요약)
검증: (1) 모든 라벨 cocoId가 stim_info에 있고 nsdId가 유일한지, (2) 피험자 1,2,5,7의 비-shared1000 학습 이미지가 전부 포함되는지,
      (3) 토큰 수가 75 이하인지, (4) 원본 COCO 캡션과 나란히 몇 장을 출력해 대응을 눈으로 확인할 수 있게 한다.
사용(맥, 저장소 루트에서): python3 scripts/mirage_server/ctx/build_ctx_captions.py"""
import gzip, json, sys
from pathlib import Path
import csv
import numpy as np
sys.path.insert(0, "src/labeling"); sys.path.insert(0, "src/fmri")
import text_variants as tv
from make_t10_embeddings import t10_text, BUDGET, NAME
from transformers import CLIPTokenizer

OUT = Path("data/results/mirage_ctx"); OUT.mkdir(parents=True, exist_ok=True)
import glob
# 공식 라벨(pass 0, 36,977장). 형식 {dim: [{value, confidence, ...}]} 을 text_variants가 받는 [[value, confidence], ...]로 바꾼다.
lab = {}
for f in sorted(glob.glob("data/labels/classify_scale/accepted/*_p0_*.json")):
    for it in json.load(open(f))["items"]:
        c = str(it["image_id"])
        assert c not in lab, f"중복 이미지 {c}"
        lab[c] = {"labels": {d: [[x["value"], x["confidence"]] for x in v] for d, v in it["labels"].items()}}
ids = list(lab)
old = {str(r["cocoId"]): r for r in map(json.loads, gzip.open("data/colab_bundle/labels_pass0.jsonl.gz", "rt"))}   # 이전 20,877장 번들(일치 검증용)
stim = list(csv.DictReader(open("data/raw/nsd_stim_info_merged.csv")))
assert len({r["nsdId"] for r in stim}) == len(stim) and len({r["cocoId"] for r in stim}) == len(stim)
coco2nsd = {r["cocoId"]: int(r["nsdId"]) for r in stim}
nsd2coco = {int(r["nsdId"]): r["cocoId"] for r in stim}
tok = CLIPTokenizer.from_pretrained(NAME)
ntok = lambda s: len(tok(s, add_special_tokens=False).input_ids)

texts, miss, same_as_old = {}, 0, []
for c in ids:
    n = coco2nsd.get(str(c))
    if n is None: miss += 1; continue
    s, k, tot = t10_text(tv.context_by_dim(lab[c]["labels"]), ntok)
    assert ntok(s) <= BUDGET
    texts[str(n)] = s
    if c in old:
        s_old, _, _ = t10_text(tv.context_by_dim(old[c]["labels"]), ntok); same_as_old.append(s == s_old)
train_need = set()
for s in (1, 2, 5, 7):
    train_need |= {int(r["nsdId"]) for r in stim if r[f"subject{s}"] != "0" and r["shared1000"] != "True"}
absent = [n for n in train_need if str(n) not in texts]
lens = np.array([ntok(t) for t in texts.values()])
check = {"n_labeled": len(ids), "n_written": len(texts), "cocoId_not_in_stim_info": miss,
         "train_images_subj1257_nonshared1000": len(train_need), "of_which_missing_text": len(absent),
         "tokens_mean": float(lens.mean()), "tokens_max": int(lens.max()), "tokens_over_75": int((lens > BUDGET).sum()),
         "empty_text": int(sum(1 for t in texts.values() if not t.strip())),
         "overlap_with_old_bundle": len(same_as_old), "overlap_identical_text": int(sum(same_as_old))}
cap = {r["cocoId"]: r for r in map(json.loads, gzip.open("data/colab_bundle/captions.jsonl.gz", "rt"))} if Path("data/colab_bundle/captions.jsonl.gz").exists() else {}
ex = []
for n in list(texts)[:4]:
    cid = nsd2coco[int(n)]
    ex.append({"nsdId": n, "cocoId": cid, "context_T10": texts[n], "coco_caption": str(cap.get(int(cid), cap.get(cid, "")))[:300]})
check["examples"] = ex
json.dump(texts, open(OUT / "context_T10_by_nsdid.json", "w"), ensure_ascii=False)
json.dump(check, open(OUT / "ctx_T10_check.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in check.items() if k != "examples"}, ensure_ascii=False, indent=1))
for e in ex: print(e)
