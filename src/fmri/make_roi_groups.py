"""영역 묶음(R1 nsdgeneral, R2 범주 선택 floc, R3 고수준 streams 5~7, EARLY streams 1)을 voxels.npz의 복셀 순서에 맞춰 저장한다.

streams 라벨(NSD의 streams.mgz.ctab 확인): 1 early, 2 midventral, 3 midlateral, 4 midparietal, 5 ventral, 6 lateral, 7 parietal.
출력: data/raw/nsd_fmri/{subj}/roi_betas/roi_groups.npz
사용: python3 src/fmri/make_roi_groups.py [subj02]
"""
import sys

import nibabel as nib
import numpy as np

R = f"data/raw/nsd_fmri/{sys.argv[1] if len(sys.argv) > 1 else 'subj01'}/"
vox = np.load(R + "roi_betas/voxels.npz")
idx = vox["flat_index"]
load = lambda n: np.asanyarray(nib.load(R + f"roi/{n}.nii.gz").dataobj).ravel()[idx]
streams = load("streams")
g = {
    "R1": load("nsdgeneral") > 0,
    "R2": (load("floc-places") > 0) | (load("floc-faces") > 0) | (load("floc-bodies") > 0) | (load("floc-words") > 0),
    "R3": np.isin(streams, [5, 6, 7]),
    "EARLY": streams == 1,
}
# 세부 영역(영역별 분석용): 범주 선택 영역 각각과 streams 단계별
for n in ("places", "faces", "bodies", "words"):
    g[n] = load(f"floc-{n}") > 0
for i, n in enumerate(("S_early", "S_midventral", "S_midlateral", "S_midparietal", "S_ventral", "S_lateral", "S_parietal"), 1):
    g[n] = streams == i
np.savez(R + "roi_betas/roi_groups.npz", **g)
print({k: int(v.sum()) for k, v in g.items()}, "복셀 총", len(idx))
