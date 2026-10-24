# MIRAGE 서버(moana-u4) 실행 기록

논문: MIRAGE (arXiv 2605.17198). 스크립트: `scripts/mirage_server/`. 이 문서는 실행하면서 측정한 값만 적는다. 추정은 "추정"으로 표시했다.
**논문 결과의 재현이 아니라, 원본 코드를 줄인 설정으로 끝까지 실행한 기록이다** (아래 "논문과 다른 점").

## 환경
- 서버: Seraph/moana, 계정 `ugrad_advisor_x`(지도교수 없는 학부생), 파티션 `debug_ugrad_advisor_x`(srun, 4h), `batch_ugrad_advisor_x`(sbatch, 최대 1일). 한도: GPU 1장, 동시 작업 2개. 메모리는 110GB 요청도 허용(`--test-only`)됐으나 u4 여유 때문에 대기.
- 노드 u4: RTX A2000 12GB (compute capability 8.6, bf16 지원), RAM 119GB(공유), 로컬 디스크 `/local_datasets`(공유, 97% 사용 중).
- 환경: conda(Private Anaconda, `/data/sslmyo/envs/mei-env`), Python 3.10, torch 2.1.0+cu121.

## 단계별 측정
| 단계 | 설정 | 결과 |
|---|---|---|
| 스모크(정답 CLIP 임베딩으로 생성) | 10+5단계 | GPU 최대 9.6GB(패치 전 12.6~13.9GB), NaN 없음, 이미지당 74~78초 |
| 학습 1 | 10세션, retrieval 없음, sklearn Ridge, 64GB | 6분 24초, MaxRSS 48.9GB. (32GB는 OOM으로 실패) |
| 학습 2 | 40세션, retrieval 없음, `lean_ridge`, 64GB | 50분 53초, MaxRSS 46.1GB |
| 생성 1 | 10세션 모델, 20+10단계, 샘플 1장, retrieval 없음 | CLIP 식별 top-1: 재구성→정답 0.222, 정답→재구성 0.167 (우연 0.056) |
| 생성 2 | 40세션 모델, 같은 설정 | CLIP 식별 top-1: 재구성→정답 0.167, 정답→재구성 0.222 |

- 자극 18개라 값 하나가 0.056씩 움직인다. 우연히 4/18 이상 맞을 확률은 약 1.6%, 3/18 이상은 약 7.5%(이항분포 계산). 10세션과 40세션은 이 지표로 구분되지 않는다.
- 정성(10세션 그림): 서핑, 음식, 사람 등은 일부 맞고, 바다와 보트, 새, 강은 틀림. 도형 6개는 구조만 맞고 내용은 무작위 일상 사진. 얼룩말→기린, mammal→개.

## 원본 코드에 가한 변경 (모두 `scripts/mirage_server/`)
- `patch_offload.py`: Stage C/B 생성기를 번갈아 GPU에 올림(12GB GPU용). 스모크로만 검증.
- `lean_ridge.py`(옵션 `LEAN_RIDGE=1`): sklearn Ridge와 같은 수식을 float32 청크로 풂. 소규모 대조 상대오차 약 1e-6(CPU/GPU 모두). 40세션 실제 데이터에서 sklearn과 직접 비교한 것은 아님.
- `link_embeddings.py`: Train.ipynb가 `_train_{N}.0sess.pt`로 저장하는 임베딩을 recon이 읽는 `_train.pt`로 연결(업스트림 파일명 불일치).
- `run_nb.py`: 업스트림 노트북의 인자 문자열 교체. 업스트림의 `--batch_size`, `--no-multi_subject`는 파서에 없어서 제거.
- `RECON_BS`: retrieval 후보를 한 번에 8장이 아니라 1장씩 생성.

## 논문과 다른 점
세션 10/40(논문 40), retrieval 없음(논문은 후보 16장 중 CLIP 유사도로 선택), 샘플 1장(논문은 10장 평균), 피험자 1명(논문 1, 2, 5, 7), 평가 지표가 다름(논문 Table 1의 2-way, 특징 지표는 미구현). 12GB 메모리 패치가 수치에 영향이 없는지는 확인하지 않았다.

## 다음 계획
- 같은 10세션에서 retrieval(후보 16장)만 켠 통제 비교.
- 논문의 특징 지표(CLIP 2-way, AlexNet, Inception 등) 계산 스크립트.
- 샘플 수를 늘려 평균.
