"""classify_v8/g01~g50(각 20장, sample 0/1/2)을 집계하고, classify.jsonl에 로그한 뒤,
50장씩 묶어(20세트) critique-a+adjudicate/critique-b용 입력을 만든다.

v7에서 발견된 배치 내 문구 복사 오염(사이sample_idx 한 이미지의 서사형 라벨이 다른
이미지에 그대로 복사되는 사고)을 이번엔 사전에 걸러내기 위해, 각 sample 파일 안에서
implied_event/notes 값이 서로 다른 place_type을 가진 이미지 사이에 그대로 중복되면
경고로 출력한다(자동 삭제는 하지 않음 — 사람이 확인).
"""
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CDIR = REPO / "data" / "labels" / "external_labeling" / "classify_v8"
JDIR = REPO / "data" / "labels" / "external_labeling" / "judge_v8"
JDIR.mkdir(parents=True, exist_ok=True)

N_GROUPS = 50
BATCH_SIZE = 50  # judge_v8 입력 배치 크기 (256KB Read 제한 대응, v7에서 확인된 안전 크기)


def check_cross_contamination(gg, s, items):
    place = {it["image_id"]: set(v["value"] for v in it["labels"].get("place_type", [])) for it in items}
    warnings = []
    for dim in ("implied_event", "notes"):
        val_to_images = defaultdict(list)
        for it in items:
            for v in it["labels"].get(dim, []):
                val_to_images[v["value"]].append(it["image_id"])
        for val, imgs in val_to_images.items():
            if len(imgs) < 2:
                continue
            for i in range(len(imgs)):
                for j in range(i + 1, len(imgs)):
                    p1, p2 = place.get(imgs[i], set()), place.get(imgs[j], set())
                    if p1 and p2 and not (p1 & p2):
                        warnings.append(f"  [경고] g{gg} s{s} [{dim}] {imgs[i]}({p1}) / {imgs[j]}({p2}) 문구 동일: {val!r}")
    return warnings


def aggregate_group(g):
    gg = f"{g:02d}"
    samples = [json.load(open(CDIR / f"g{gg}_classify_out_{s}.json", encoding="utf-8")) for s in range(3)]
    by_image = {}
    for s, batch in enumerate(samples):
        warnings = check_cross_contamination(gg, s, batch.get("items", []))
        for w in warnings:
            print(w)
        for item in batch.get("items", []):
            by_image.setdefault(item["image_id"], []).append(item)

    sys.path.insert(0, str(REPO / "src" / "labeling"))
    import aggregate as agg_mod

    agg_items = [agg_mod.aggregate(v) for v in by_image.values()]
    out = {"items": agg_items}
    json.dump(out, open(CDIR / f"g{gg}_agg.json", "w"), ensure_ascii=False, indent=1)

    log_items = []
    for s in range(3):
        d = samples[s]
        for it in d["items"]:
            log_items.append({
                "image_id": it["image_id"], "agent_name": "context-classify",
                "agent_model": "codex-gpt5.5-high", "prompt_version": "v8",
                "input": {"sample_idx": s, "group": gg},
                "raw_output": json.dumps(it, ensure_ascii=False),
                "parsed_label": it, "flags": [],
            })
    log_path = CDIR / f"g{gg}_classify_log.json"
    json.dump({"items": log_items}, open(log_path, "w"), ensure_ascii=False)
    subprocess.run(
        [sys.executable, str(REPO / "src" / "labeling" / "log_stage.py"), "--stage", "classify"],
        stdin=open(log_path, encoding="utf-8"), check=True, cwd=REPO,
    )
    return out


def main():
    all_agg_items = []
    for g in range(1, N_GROUPS + 1):
        out = aggregate_group(g)
        all_agg_items.extend(out["items"])
        print(f"g{g:02d} 집계 완료 ({len(out['items'])}장)")

    print(f"\n총 집계된 이미지: {len(all_agg_items)}")

    n_batches = (len(all_agg_items) + BATCH_SIZE - 1) // BATCH_SIZE
    for j in range(n_batches):
        chunk = all_agg_items[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
        items = [{"image_id": it["image_id"], "labels": it["labels"]} for it in chunk]
        jj = f"{j + 1:02d}"
        json.dump({"items": items}, open(JDIR / f"j{jj}_crit_in.json", "w"), ensure_ascii=False, indent=1)
        print(f"j{jj}_crit_in.json 생성 ({len(items)}장)")


if __name__ == "__main__":
    main()
