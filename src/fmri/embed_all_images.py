"""ctx_ids.json의 모든 이미지(20,877장)에 대해 CLIP 이미지·DINOv2·ResNet-50 임베딩을 한 번 계산해 IMGALL__*.npy로 저장한다(ctx_ids.json 순서)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import target_type_compare as tt  # noqa: E402

ids = json.load(open("data/embeddings_v2/ctx_ids.json"))
tt.embed_images(ids, "data/raw/images", "data/embeddings_v2", prefix="IMGALL")
print("완료", len(ids))
