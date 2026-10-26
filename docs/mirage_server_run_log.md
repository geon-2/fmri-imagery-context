# MIRAGE 서버(moana-u4) 실행 기록

논문: MIRAGE (arXiv 2605.17198). 스크립트: `scripts/mirage_server/`. 이 문서는 실행하면서 측정한 값만 적는다. 추정은 "추정"으로 표시했다.
초기에는 줄인 설정으로 끝까지 실행한 기록이었고, 이후 논문 설정에 가깝게 맞춘 최종 실행(A+B)을 진행했다. 현재 상태는 위 "인계 요약"을 본다.

## 인계 요약 (2026-10-26 기준, 미팅 준비 세션용)
이 저장소에 남긴 기록이 전부이고 **Notion에는 아무것도 쓰지 않았다**(`CLAUDE.md`: Notion 쓰기는 사람이 결과를 확인한 뒤 명시적으로 요청했을 때만). 목표는 논문 Table 1(MIRAGE, imagery)의 NSD-Imagery 벤치마크 수치가 재현되는지 확인하는 것이다. 사람 평가(논문 Table 2, 평가자 500명)는 하지 않는다.

**현재 상태**
- 원본 코드로 NSD-Imagery 18개 자극 복원을 끝까지 실행함(서버 moana-u4, RTX A2000 12GB, 학부 계정 한도: GPU 1장, 메모리 64GB 요청은 허용).
- 최종 실행 계획(A+B)을 2026-10-25에 승인받아 제출함. B: 피험자 2, 5, 7, 40세션, retrieval 없음, 반복 5회 — **완료**. A: 피험자 1, 40세션, retrieval 후보 16, 반복 3회 — 학습 완료(17분), 생성 시드 1 실행 중(작업 150497, 2026-10-26 08:33 시작), 시드 2, 3 대기(150498, 150499). 시드당 약 10시간(10세션 retrieval 실측 9시간 40분에서 추정)이라 시드 3은 2026-10-27(화) 오후쯤 끝날 것으로 어림.
- 피험자 1의 retrieval 없는 40세션 반복 5개(원본 시드 42 + 시드 1~4)도 있음.

**중간 결과(공식 평가 방식 `official`, 피험자 4명 평균, 반복 5회, retrieval 없음, 자극 12개)** — 최종이 아님(피험자 1을 retrieval 포함 결과로 바꿔야 함)

| 지표 | 우리 평균 | 논문 Table 1 | 상대 차이 |
|---|---|---|---|
| PixCorr | 0.106 | 0.104 | +2.2% |
| SSIM | 0.390 | 0.398 | −2.0% |
| Alex(2) | 0.616 | 0.639 | −3.7% |
| Alex(5) | 0.607 | 0.625 | −2.8% |
| Incep | 0.562 | 0.523 | +7.7% |
| CLIP | 0.552 | 0.575 | −4.0% |
| EffNet-B | 0.919 | 0.914 | +0.6% |
| SwAV | 미계산 | 0.575 | — (`torch.hub`가 외부 저장소 코드를 실행해 보안 확인에서 거부됨. 7개 지표로 판정할지 사용자 결정 대기) |

피험자별(반복 5회 평균)은 값이 꽤 흩어진다. 예: Incep 피험자 1 0.708, 2 0.632, 5 0.489, 7 0.421. Alex(2)는 피험자 7이 0.514로 낮음. 7개 지표 모두 4명 평균에서 논문 값과 상대 차이 8% 이내. 판정 규칙(15% 이내가 75% 이상)에서 7/7이지만, 이 규칙은 피험자 2의 결과를 본 뒤에 정했다(`docs/mirage_repro_criteria.md`에 경위 기록). 논문 Table 1이 피험자 평균인지는 확인하지 못했다.

**주의할 점(발표 때 같이 말해야 함)**
- 판정 규칙은 결과를 본 뒤에 사용자 지시로 정함. 규칙 자체에 근거 있는 임계값이 없음(논문이 오차를 보고하지 않음).
- 95% 구간은 자극 12개의 변동만 반영하고 구분력이 약함(주 규칙이 아님).
- 공식 코드의 2-way 계산에 `-1`이 한 번 더 들어가 값이 약 0.091 낮게 나옴(`official`). 이걸 뺀 `corrected` 값은 논문보다 약 0.09 높음. 논문 표가 이 코드로 나왔는지는 추정.
- 12GB GPU용 메모리 패치(`patch_offload.py`), 메모리 절약 릿지(`lean_ridge.py`), 임베딩 통계 저장(`run_nb.py`의 stats 패치)은 원본 코드에 없는 변경이다. 수치에 영향이 없다는 직접 대조는 하지 않았다(소규모 합성 데이터 대조만).

**결과 파일 위치**
- 서버 NAS: `/data/sslmyo/mirage_results/<모델이름>/` (예: `subj02_40sess_server_ridge_sc2/..._all_recons_imagery_seed1.pt`). 로그: `/data/sslmyo/mirage_server/logs/`.
- 맥: `data/results/mirage_server/`(git 무시). recons만 받음. 평가 표: `results/tables/imagery_eval_*`(git 무시).
- 다시 계산: `python -m src.eval.nsd_imagery_eval --gt data/results/mirage_server/imagery_stimuli_18.pt --recons s1=<파일들,쉼표> ... --combine all4=s1,s2,s5,s7 --paper all4 --no-swav`

**남은 일**
1. A 생성 시드 1~3이 끝나면(화요일 오후쯤) 피험자 1을 retrieval 포함 결과로 바꿔 4명 평균 최종 계산.
2. SwAV 지표를 계산할지 정함(7개 지표로 판정하면 불필요).
3. 아직 구현하지 않은 지표: brain correlation(GNet 필요), 캡션 지표.

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
| 학습 3 | 10세션, retrieval 포함, `lean_ridge`, 64GB | 7분 24초, MaxRSS 45.1GB |
| 생성 3 | 10세션 retrieval 후보 16장(1장씩 생성), 20+10단계 | 9시간 40분, MaxRSS 50.6GB. CLIP 식별 top-1: 0.222 / 0.222 |

- 자극 18개라 값 하나가 0.056씩 움직인다. 우연히 4/18 이상 맞을 확률은 약 1.6%, 3/18 이상은 약 7.5%(이항분포 계산). 10세션과 40세션은 이 지표로 구분되지 않는다.
- 정성(10세션 그림): 서핑, 음식, 사람 등은 일부 맞고, 바다와 보트, 새, 강은 틀림. 도형 6개는 구조만 맞고 내용은 무작위 일상 사진. 얼룩말→기린, mammal→개.

### NSD-Imagery 공식 평가 방식의 지표 (`src/eval/nsd_imagery_eval.py`, 자극 12개, 피험자 1, 샘플 1장)
공식 평가 노트북(MedARC-AI/MIRAGE `src/final_evaluations_mi_multi.ipynb`)을 읽고 같은 방식으로 구현했다. 확인한 점: 정답이 있는 앞 12개만 평가, 재구성 512 리사이즈, 2-way의 distractor는 다른 *정답* 이미지(논문 A.6 문장과 다름), 공식 코드의 성공 수 계산에 `-1`이 한 번 더 들어가 값이 1/(N-1)=약 0.091 낮게 나옴(`official`=코드 그대로, `corrected`=-1 제거). 공식 노트북과 같은 데이터로 직접 대조한 것은 아직 아니다.
이전에 쓴 CLIP 18-way 식별(top-1), 그리고 distractor 방향이 달랐던 2-way 표는 공식 방식이 아니라서 논문 비교에 쓰지 않는다.

| 설정 (official) | PixCorr | SSIM | Alex(2) | Alex(5) | Incep | CLIP | EffNet-B↓ | SwAV↓ |
|---|---|---|---|---|---|---|---|---|
| 10세션, retrieval 없음 | 0.151 | 0.414 | 0.674 | 0.545 | 0.553 | 0.576 | 0.928 | 0.574 |
| 40세션, retrieval 없음 | 0.172 | 0.405 | 0.689 | 0.621 | 0.727 | 0.659 | 0.882 | 0.553 |
| 10세션, retrieval 후보 16 | 0.154 | 0.412 | 0.606 | 0.621 | 0.659 | 0.621 | 0.908 | 0.556 |
| (논문 Table 1, MIRAGE imagery) | 0.104 | 0.398 | 0.639 | 0.625 | 0.523 | 0.575 | 0.914 | 0.575 |

- **구현 검증**: 공식 노트북의 함수를 그대로 복사해 같은 재구성(10세션)에 돌린 값과 `src/eval` 구현 값이 PixCorr, SSIM, Alex(2), Alex(5), Incep, CLIP, EffNet-B 7개에서 소수 5자리까지 일치(`scripts/mirage_eval/crosscheck_official.py`). SwAV는 대조하지 못함. 리사이즈의 antialias는 torchvision 기본값(True)을 썼다. 논문 저자 환경의 torchvision 버전은 모르며, 값 차이를 보려면 `--antialias false`로 다시 계산할 수 있다.
- 논문 값은 샘플 10장(MIRAGE는 5회 반복) 평균이고 피험자 평균 여부는 확인 못 함(부록 Table E는 "averaged across all subjects"라고 적음). 우리는 피험자 1명, 샘플 1장이라 직접 비교가 아니다.
- 10세션 대비 차이의 95% 구간(자극 12개 복원추출, 자극 변동만 반영): Alex(5) retrieval +0.076 [+0.030, +0.121]; EffNet-B 40세션 -0.046 [-0.082, -0.009]. 나머지는 구간이 0을 포함. 지표 8개 × 비교 2개를 본 것이라 구간이 0을 벗어나는 일부는 우연일 수 있다.

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

## 최종 실행(A+B) 기록
- **파이프라인**(`scripts/mirage_server/04_pipeline.sh`, `submit_AB.sh`): 준비(데이터 다운로드, 메모리 패치) → 학습 → 시드별 생성 → 정리(그 피험자 임베딩, 가중치, 베타 삭제로 디스크 확보). 모든 작업은 u4에 고정(데이터가 노드 로컬 디스크).
- **임베딩 통계 저장**: 40세션 retrieval이면 학습 임베딩이 약 46GB라 디스크가 모자라서, recon이 쓰는 평균과 불편 표준편차만 저장하도록 학습 노트북을 패치했다(표본 2개로 평균과 표준편차를 재현). 합성 데이터 대조 오차 약 1e-7. 서버에서 recon이 이 파일을 읽어 정상 동작함을 피험자 2에서 확인(결과 지표가 피험자 1과 같은 범위).
- **측정(서버)**: B 피험자 2, 5, 7 각각 3시간 56분~3시간 58분(학습 약 50분 + 시드 5개 × 약 38분 + 정리). A 학습(40세션, retrieval, lean ridge) 17분(이미지/텍스트/VDVAE 임베딩은 기존 파일 재사용).
- **디스크**: u4 로컬 디스크가 다른 사용자와 공유라 여유가 17~55GB 사이에서 움직였다. 10세션 관련 파일을 지워 확보함.
- **시드 반복**: 같은 모델에서 `SEED`만 바꿔 생성. 결과 파일에 시드가 붙음(`..._seed{N}.pt`). 피험자 1의 retrieval 없는 40세션: 원본(시드 42) + 시드 1~4, 각 약 38분.
