"""g01~g50(각 20장, sample 0/1/2)을 전부 집계하고, classify.jsonl에 로그한 뒤,
100장씩 묶어(5개 그룹) critique-a+adjudicate/critique-b용 입력 10세트를 만든다.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CDIR = REPO / "data" / "labels" / "external_labeling" / "classify_v7"
JDIR = REPO / "data" / "labels" / "external_labeling" / "judge_v7"
JDIR.mkdir(parents=True, exist_ok=True)

N_GROUPS = 50
GROUPS_PER_SUPER = 5  # 5 x 20 = 100장씩


def aggregate_group(g):
    gg = f"{g:02d}"
    samples = [json.load(open(CDIR / f"g{gg}_classify_out_{s}.json", encoding="utf-8")) for s in range(3)]
    by_image = {}
    for batch in samples:
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
                "agent_model": "codex-gpt5.5-high", "prompt_version": "v7",
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

    n_super = N_GROUPS // GROUPS_PER_SUPER
    for j in range(n_super):
        chunk = all_agg_items[j * 100:(j + 1) * 100]
        items = [{"image_id": it["image_id"], "labels": it["labels"]} for it in chunk]
        jj = f"{j + 1:02d}"
        json.dump({"items": items}, open(JDIR / f"j{jj}_crit_in.json", "w"), ensure_ascii=False, indent=1)
        print(f"j{jj}_crit_in.json 생성 ({len(items)}장)")


if __name__ == "__main__":
    main()
