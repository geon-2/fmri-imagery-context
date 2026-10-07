"""이미지마다 주요 COCO 범주 = 마스크 면적(annotation area 합)이 가장 큰 범주(동률이면 category id가 작은 쪽).
출력: data/results/coco_primary.json {cocoId: 범주 이름}. 범주 단위 분할과 같은 범주 안 순위 평가에 쓴다."""
import gzip
import json
from collections import defaultdict
from pathlib import Path

R = Path("data")


def main():
    stim = {r["cocoId"]: r["cocoSplit"] for r in map(json.loads, gzip.open(R / "colab_bundle/stim_info.jsonl.gz", "rt"))}
    ids = set(json.load(open(R / "embeddings_v2/ctx_ids.json")))
    area = defaultdict(lambda: defaultdict(float))
    names = {}
    for split in ("train2017", "val2017"):
        d = json.load(open(R / f"raw/annotations/instances_{split}.json"))
        for c in d["categories"]:
            names[c["id"]] = c["name"]
        want = {int(i) for i in ids if stim.get(i, "").startswith(split[:3])}
        for a in d["annotations"]:
            if a["image_id"] in want:
                area[str(a["image_id"])][a["category_id"]] += a["area"]
        del d
    out = {}
    for i in ids:
        a = area.get(i)
        if a:
            out[i] = names[sorted(a.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]]
    Path("data/results").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("data/results/coco_primary.json", "w"), ensure_ascii=False)
    from collections import Counter
    c = Counter(out.values())
    print(f"{len(out)}/{len(ids)}장, 범주 {len(c)}개, 상위: {c.most_common(8)}")


if __name__ == "__main__":
    main()
