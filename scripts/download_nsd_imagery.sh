#!/bin/bash
# NSD-Imagery(공개 S3, natural-scenes-dataset): 피험자 1·2·5·7의 1.8mm GLMdenoise_RR 베타(nii.gz)와 실험 설계·자극 파일.
# hdf5(같은 내용의 복제본, 약 1GB)와 화면 녹화 mp4는 받지 않는다. NSD 이용약관에 따라 재배포하지 않는다.
cd /Users/gunlee/Development/fmri-imagery-context || exit 1
S3=s3://natural-scenes-dataset
OUT=data/raw/nsd_imagery
for s in subj01 subj02 subj05 subj07; do
  mkdir -p $OUT/$s
  for f in betas_nsdimagery.nii.gz R2_nsdimagery.nii.gz FRACvalue_nsdimagery.nii.gz meanbeta_nsdimagery.nii.gz; do
    [ -f $OUT/$s/$f ] || aws s3 cp --no-sign-request --only-show-errors $S3/nsddata_betas/ppdata/$s/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/$f $OUT/$s/$f
  done
  echo "[$(date '+%F %T')] $s 베타 완료"
done
aws s3 sync --no-sign-request --only-show-errors --exclude "*.mp4" --exclude "*.pdf" $S3/nsddata/experiments/nsdimagery/ $OUT/exp/
echo "[$(date '+%F %T')] 실험 설계·자극 완료"
echo "전체 완료"
