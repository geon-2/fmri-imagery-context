"""NSD 1.8mm betas(GLMdenoise_RR, session별 .nii.gz)를 받아 고수준 영역 복셀만 잘라 저장한다.

세션 하나씩 받아서 자르고 원본은 지우므로 디스크는 세션 하나(약 0.5GB)만 임시로 쓴다.
저장: data/raw/nsd_fmri/{subj}/roi_betas/session{NN}.npy  (trial x voxel, float16, beta/300)
      data/raw/nsd_fmri/{subj}/roi_betas/voxels.npz       (복셀 인덱스와 영역별 소속 마스크)
영역 합집합: nsdgeneral, floc-places/faces/bodies/words, streams. MTL(기억 영역)은 따로 저장해 둔다(복셀 수가 적음).
공개 S3(natural-scenes-dataset)에서 받는다. NSD 데이터 사용 약관에 따라 재배포하지 않는다.

사용: python3 src/fmri/extract_roi_betas.py --subj subj01 --sessions 1:41
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

S3 = "s3://natural-scenes-dataset"
ROOT = Path("data/raw/nsd_fmri")
HIGH_ROIS = ["nsdgeneral", "floc-places", "floc-faces", "floc-bodies", "floc-words", "streams"]
MEMORY_ROIS = ["MTL"]


def aws_cp(src, dst):
    subprocess.run(["aws", "s3", "cp", "--no-sign-request", "--only-show-errors", src, str(dst)], check=True)


def load_masks(subj):
    d = ROOT / subj / "roi"
    d.mkdir(parents=True, exist_ok=True)
    masks = {}
    for name in HIGH_ROIS + MEMORY_ROIS:
        f = d / f"{name}.nii.gz"
        if not f.exists():
            aws_cp(f"{S3}/nsddata/ppdata/{subj}/func1pt8mm/roi/{name}.nii.gz", f)
        masks[name] = np.asanyarray(nib.load(f).dataobj) > 0
    return masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subj", default="subj01")
    ap.add_argument("--sessions", default="1:41")
    a = ap.parse_args()
    lo, hi = map(int, a.sessions.split(":"))
    out = ROOT / a.subj / "roi_betas"
    out.mkdir(parents=True, exist_ok=True)

    masks = load_masks(a.subj)
    high = np.zeros_like(masks["nsdgeneral"])
    for n in HIGH_ROIS:
        high |= masks[n]
    mem = masks["MTL"] & ~high
    union = high | mem
    idx = np.flatnonzero(union.ravel())
    np.savez(out / "voxels.npz", flat_index=idx, shape=np.array(union.shape),
             **{f"in_{n}": masks[n].ravel()[idx] for n in HIGH_ROIS + MEMORY_ROIS})
    print(f"{a.subj}: 고수준 영역 {int(high.sum())}복셀, 기억 영역(MTL, 겹침 제외) {int(mem.sum())}복셀", flush=True)

    for s in range(lo, hi):
        dst = out / f"session{s:02d}.npy"
        if dst.exists():
            continue
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            f = Path(td) / "b.nii.gz"
            aws_cp(f"{S3}/nsddata_betas/ppdata/{a.subj}/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session{s:02d}.nii.gz", f)
            arr = np.asanyarray(nib.load(f).dataobj)  # x,y,z,trial (int16)
            flat = arr.reshape(-1, arr.shape[-1])[idx].T  # trial x voxel
            np.save(dst, (flat.astype(np.float32) / 300.0).astype(np.float16))
        print(f"session {s:02d} 저장 {dst.name} {flat.shape}", flush=True)


if __name__ == "__main__":
    main()
