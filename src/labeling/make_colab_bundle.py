"""Colab에서 쓸 자료 묶음을 만든다(data/colab_bundle/, git 추적 안 함). 사용자의 비공개 Drive에만 올린다.

  captions.jsonl.gz     후보 풀 전체 + shared1000: cocoId, nsdId, 원본 캡션(이미지당 5개), object-only 텍스트(COCO 사물 범주명 중복 제거·알파벳순 쉼표 나열)
  labels_pass0.jsonl.gz 공식 라벨(accepted pass 0): cocoId, prompt_version, 6항목 [value, confidence]
  labels_pass9.jsonl.gz 재현성 측정의 두 번째 라벨(같은 이미지 300장, 같은 v10)

사용: python3 src/labeling/make_colab_bundle.py
"""
import csv
import glob
import gzip
import json
from collections import defaultdict
from pathlib import Path

D = Path("data/labels/classify_scale")
OUT = Path("data/colab_bundle")
DIMS = ("place_type", "environment", "activity", "implied_event", "scale", "notes")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pool = {}
    for r in csv.DictReader(open(D / "queue_order.csv", encoding="utf-8")):
        pool[r["cocoId"]] = {"cocoId": r["cocoId"], "nsdId": r["nsdId"], "captions": json.loads(r["captions"])}
    # 평가용 shared1000도 캡션·object-only가 필요하다(라벨은 아직 없음). nsdId가 같은 항목은 pool에 없다.
    caps_all = defaultdict(list)
    for split in ("train", "val"):
        for a in json.load(open(f"data/raw/annotations/captions_{split}2017.json", encoding="utf-8"))["annotations"]:
            caps_all[str(a["image_id"])].append(a["caption"].strip())
    for r in csv.DictReader(open("data/raw/nsd_stim_info_merged.csv", encoding="utf-8")):
        if r["shared1000"] == "True" and r["cocoId"] not in pool:
            pool[r["cocoId"]] = {"cocoId": r["cocoId"], "nsdId": r["nsdId"], "captions": caps_all[r["cocoId"]], "shared1000": True}
    cats = defaultdict(set)
    for split in ("train", "val"):
        d = json.load(open(f"data/raw/annotations/instances_{split}2017.json", encoding="utf-8"))
        name = {c["id"]: c["name"] for c in d["categories"]}
        for a in d["annotations"]:
            k = str(a["image_id"])
            if k in pool:
                cats[k].add(name[a["category_id"]])
        del d
    with gzip.open(OUT / "captions.jsonl.gz", "wt", encoding="utf-8") as f:
        for k, r in pool.items():
            r["object_only"] = ", ".join(sorted(cats.get(k, [])))
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = 0
    with gzip.open(OUT / "labels_pass0.jsonl.gz", "wt", encoding="utf-8") as f:
        for p in sorted(glob.glob(str(D / "accepted" / "u*_p0_*.json"))):
            d = json.load(open(p, encoding="utf-8"))
            for it in d["items"]:
                lab = {k: [[v["value"], v["confidence"]] for v in it["labels"][k]] for k in DIMS}
                f.write(json.dumps({"cocoId": str(it["image_id"]), "prompt_version": d["meta"]["prompt_version"], "labels": lab}, ensure_ascii=False) + "\n")
                n += 1
    n9 = 0
    with gzip.open(OUT / "labels_pass9.jsonl.gz", "wt", encoding="utf-8") as f:  # 재현성 측정용 두 번째 라벨(같은 이미지, 같은 v10)
        for p in sorted(glob.glob(str(D / "accepted" / "u*_p9_*.json"))):
            d = json.load(open(p, encoding="utf-8"))
            for it in d["items"]:
                lab = {k: [[v["value"], v["confidence"]] for v in it["labels"][k]] for k in DIMS}
                f.write(json.dumps({"cocoId": str(it["image_id"]), "prompt_version": d["meta"]["prompt_version"], "labels": lab}, ensure_ascii=False) + "\n")
                n9 += 1
    print(f"재현성 두 번째 라벨 {n9}장")
    empty = sum(1 for r in pool.values() if not cats.get(r["cocoId"]))
    print(f"후보 {len(pool)}장, 라벨 {n}장, 사물 주석이 없는 이미지 {empty}장")


if __name__ == "__main__":
    main()
