"""classify 샘플들을 이미지별로 라벨 support/mean_confidence로 집계한다.

값 문자열은 대소문자·앞뒤 공백만 무시하고 비교한다. 동의어는 합치지 않는다(어휘 정리는
representation 단계 이전에 사람이 한다). 라벨은 삭제하지 않는다.

두 가지 사용법:
  단일 이미지: classify 출력(그 이미지의 샘플 리스트) JSON을 표준입력으로 넘긴다.
    python3 aggregate.py < single_image_samples.json
  배치(여러 이미지, sample_idx별 배치 응답): classify v2가 이미지 여러 장을 한 번에 반환하는
  배치 호출을 sample_idx 개수만큼(N번) 모은 리스트를 표준입력으로 넘긴다. 각 원소는
  {"sample_idx": 0, "items": [{"image_id": ..., "labels": {...}}, ...]} 형태다.
    python3 aggregate.py --batch < sample_batches.json
"""
import argparse
import json
import sys

DIMENSIONS = ("place_type", "environment", "activity", "implied_event", "scale", "notes")


def aggregate(samples):
    if not samples:
        raise ValueError("샘플이 없다")
    image_ids = {s.get("image_id") for s in samples}
    if len(image_ids) != 1:
        raise ValueError(f"서로 다른 image_id가 섞여 있다: {image_ids}")
    n = len(samples)
    out = {"image_id": image_ids.pop(), "n_samples": n, "labels": {d: [] for d in DIMENSIONS}}
    for dim in DIMENSIONS:
        acc = {}
        for s in samples:
            seen = set()  # 한 샘플 안에서 같은 값이 중복돼도 한 번으로 센다
            for item in (s.get("labels") or {}).get(dim, []):
                key = str(item["value"]).strip().casefold()
                if key in seen:
                    continue
                seen.add(key)
                a = acc.setdefault(key, {"value": str(item["value"]).strip(), "count": 0,
                                         "conf": [], "proposed": False})
                a["count"] += 1
                if item.get("confidence") is not None:
                    a["conf"].append(float(item["confidence"]))
                a["proposed"] = a["proposed"] or bool(item.get("proposed"))
        rows = [{"value": a["value"], "support": round(a["count"] / n, 4),
                 "mean_confidence": round(sum(a["conf"]) / len(a["conf"]), 4) if a["conf"] else None,
                 "proposed": a["proposed"]} for a in acc.values()]
        out["labels"][dim] = sorted(rows, key=lambda r: (-r["support"], r["value"]))
    return out


def aggregate_batch(sample_batches):
    """sample_idx별 배치 응답 리스트를 이미지별로 묶어 각각 aggregate()한다."""
    by_image = {}
    for batch in sample_batches:
        for item in batch.get("items", []):
            by_image.setdefault(item["image_id"], []).append(item)
    return {"items": [aggregate(samples) for samples in by_image.values()]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", action="store_true", help="여러 이미지 배치 모드")
    args = ap.parse_args()
    data = json.load(sys.stdin)
    if args.batch:
        print(json.dumps(aggregate_batch(data), ensure_ascii=False))
    else:
        print(json.dumps(aggregate(data), ensure_ascii=False))
