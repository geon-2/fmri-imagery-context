# archive

파일럿과 초기 대규모 시도(v7, v8)에서 쓰던 스크립트다. 방법을 바꾸면서 더 이상 쓰지 않지만, 어떤 과정을 거쳤는지 보이도록 남겨 둔다.
현재 파이프라인은 상위 폴더의 `scale_queue.py`, `run_scale.py`, `validate_unit.py`를 쓴다(`docs/labeling_pipeline.md` 참고).

| 파일 | 하던 일 | 왜 그만뒀나 |
|---|---|---|
| `aggregate_and_prepare.py`, `aggregate_and_prepare_v8.py` | 20장 배치 50개의 결과를 집계하고 검증용 입력을 만듦 | 한 세션에서 배치를 이어 돌리는 방식에서 문장 복사와 날림 작업이 발생해 폐기 |
| `setup_checkpoints.py`, `verify_checkpoint.py` | 100장마다 사람이 통과시키는 체크포인트 방식 | 단위별 격리 실행과 자동 검사로 대체 |
| `watch_and_log_judge.py` | critique-a/adjudicate 결과를 감시해 기록 | 대규모 실행에서는 이 단계를 하지 않음(비용) |
| `compare_pipelines.py`, `compare_all_versions.py` | 파일럿의 파이프라인 버전별 결과 비교표 | 파일럿 분석용 |
| `download_images.py` | 1,000장 이미지 다운로드 | `download_pool.py`로 대체 |
