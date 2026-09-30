"""classify_v8용 체크포인트 상태 파일을 sample_idx별로 만든다.
50개 그룹(g01~g50)을 5개씩 묶어 10개 체크포인트로 나눈다(체크포인트당 100장).
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CDIR = REPO / "data" / "labels" / "external_labeling" / "classify_v8"

CHECKPOINT_SIZE = 5  # groups per checkpoint (100 images)
N_GROUPS = 50


def build(sample_idx):
    groups = [f"g{g:02d}" for g in range(1, N_GROUPS + 1)]
    checkpoints = []
    for i in range(0, len(groups), CHECKPOINT_SIZE):
        chunk = groups[i:i + CHECKPOINT_SIZE]
        checkpoints.append({
            "checkpoint": i // CHECKPOINT_SIZE + 1,
            "groups": chunk,
            "status": "pending",  # pending(아직 시작 안함) -> done_unverified(제출, 검증 대기)
                                   # -> verified(통과, 다음 체크포인트 진행 가능) / rejected(반려, 재작업 필요)
            "output_written": False,
            "verified": None,  # null=미검증, true=통과, false=반려
            "notes": "",
            "redo_groups": [],  # 반려 시 이 그룹들만 다시 하면 됨(전체 재작업 아님)
            "redo_count": 0,
        })
    doc = {"sample_idx": sample_idx, "checkpoint_size": CHECKPOINT_SIZE, "checkpoints": checkpoints}
    path = CDIR / f"checkpoints_{sample_idx}.json"
    json.dump(doc, open(path, "w"), ensure_ascii=False, indent=1)
    print(f"{path} 생성 ({len(checkpoints)}개 체크포인트)")


if __name__ == "__main__":
    for s in (0, 1, 2):
        build(s)
