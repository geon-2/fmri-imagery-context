"""이미지 하나에 대한 classify 샘플들을 라벨별 support/mean_confidence로 집계한다.

값 문자열은 대소문자·앞뒤 공백만 무시하고 비교한다. 동의어는 합치지 않는다(어휘 정리는
representation 단계 이전에 사람이 한다). 라벨은 삭제하지 않는다.

사용: classify 출력 JSON 리스트를 표준입력으로 넘기면 집계 JSON을 표준출력으로 낸다.
"""
import json
import sys

DIMENSIONS = ("place_type", "environment", "activity", "scale")


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


if __name__ == "__main__":
    print(json.dumps(aggregate(json.load(sys.stdin)), ensure_ascii=False))
