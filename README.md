# 떠올린 장면에 남는 맥락

Perception fMRI 기반 저차원 context 축의 mental imagery 전이 분석
소프트웨어융합캡스톤디자인 2026-2 · 경희대학교 국제캠퍼스

## 한 줄 요약

기존 embedding 공간(문장 embedding)을 새로 만들지 않고, 그 공간 안에서
perception fMRI와 공유 분산이 큰 저차원 축(k개)을 뇌 정렬 기준으로 고른 뒤,
그 축이 같은 차원의 뇌 비정렬 축(PCA)보다 mental imagery로 더 잘 전이되는지 검증한다.

## 연구 질문

| ID | 질문 | 역할 |
|---|---|---|
| RQ1 | Object 정보를 넘어, perception fMRI와 context embedding이 공유하는 저차원 축이 있는가 | 사전 확인 |
| RQ2 | 그 뇌 정렬 k축이 같은 차원의 뇌 비정렬 축보다 imagery로 더 잘 전이되는가 | **주 가설** |
| RQ3 | 기억 관련 영역을 더하면 추가 이득이 있는가 | 추가 실험 |

전 단계에 **사전 점검**(context embedding과 뇌 반응의 공유 정보 유무를 RSA·encoding 예측력으로 확인)이 있다.
자세한 배경, 선행연구 비교, 방법론적 함정과 대응은 Notion 연구 노트를 참고한다(`docs/links.md`).

## 현재 진행 상황 (2026-09-30)

지금까지의 작업은 대부분 **맥락 라벨을 만드는 일**이었다. 축을 찾으려면 이미지마다 맥락이 먼저 붙어 있어야 하기 때문이다.

| 단계 | 상태 |
|---|---|
| 후보 이미지 풀 (NSD 35,977장) 확정, 이미지 다운로드 | 완료 |
| 맥락 라벨 기준 (6개 항목, 다중 라벨, 독립 샘플) | 완료 |
| 파일럿 300장, 사람 라벨 200장 | 완료 (자동 라벨과의 비교는 예정) |
| 자동 라벨링 파이프라인 | 완료, **1만 장 라벨링 끝** (이후 추가 진행 중) |
| 라벨 → 문장 embedding, 사전 점검(RQ1) | 시작 전 |
| 축 추출, decoder, imagery 평가(RQ2, RQ3) | 시작 전 (`src/axes`, `src/decoding`, `src/eval`은 비어 있음) |

## 처음 보는 분께: 어디부터 보면 되나

1. **[docs/labeling_pipeline.md](docs/labeling_pipeline.md)** — 라벨을 어떻게 만들었는지, 어떤 시행착오를 거쳤는지, 결과가 어떤지. 가장 먼저 읽으면 된다.
2. **[.claude/agents/context-classify.md](.claude/agents/context-classify.md)** — 라벨링에 쓴 프롬프트(현재 v10). 파일 안 `prompt_version`이 버전이다.
3. **[src/labeling/run_scale.py](src/labeling/run_scale.py)**, **[validate_unit.py](src/labeling/validate_unit.py)** — 단위별로 라벨링을 실행하고, 결과를 자동으로 검사해서 통과분만 저장하는 코드.
4. **[results/tables/](results/tables)** — 결과 요약 표(라벨 값 분포, 파일럿 요약 등).
5. **[CLAUDE.md](CLAUDE.md)** — 작업 규칙(라벨 원칙, 커밋·데이터 규칙). 자동화 도구에게 주는 지침이지만 프로젝트의 약속이 정리돼 있다.

## 데이터

- **NSD core** (Allen et al., Nat Neurosci 2022): perception 학습 데이터. subject 1/2/5/7.
- **NSD-Imagery** (Kneeland et al., CVPR 2025): imagery 평가 데이터.

두 데이터셋 모두 Natural Scenes Dataset Data Use Agreement를 따른다.
**원본·가공 데이터는 이 저장소에 없다** (`.gitignore` 참고). `data/` 아래 폴더는 로컬에서만 채우고,
다른 사람이 재현하려면 각자 DUA에 동의한 뒤 원본에서 내려받아야 한다.

## 저장소 구조

```
.claude/agents/   # 라벨링 프롬프트 (classify가 현재 사용, critique·adjudicate는 파일럿에서 사용)
data/             # 로컬 전용 (git 추적 안 함)
  raw/            #   NSD·COCO 원본, 이미지
  processed/      #   후보 풀, 파일럿 목록 등
  labels/         #   라벨링 결과와 기록
docs/             # 라벨링 파이프라인 설명, 사람 라벨링 가이드, 발표 자료
src/
  labeling/       # 후보 풀 만들기, 라벨링 실행·검사, 결과 집계 (archive/는 폐기한 시도)
  axes/           # (예정) 사전 점검(RSA·encoding) + 축 추출(PLS/RRR), PCA·무작위 기준선
  decoding/       # (예정) ridge decoder 학습, imagery 적용
  eval/           # (예정) 후보 순위 평가, permutation test, k-곡선
results/tables/   # 요약 표(.md만 추적)
configs/          # 실험 설정 (예정)
```

## 기록의 원천

- **연구 노트·의사결정·미팅 기록**: Notion ("떠올린 장면에 남는 맥락" 프로젝트 페이지)
- **일정**: Notion 연구 일정 스케쥴(소캡디 미팅) + Google Calendar(주차별 작업, 준비 마감)
- **코드·실험 이력**: 이 저장소의 git 로그
- **발표 자료**: `docs/slides/`에 HTML 슬라이드·대본 보관 (버전 관리)

셋의 역할을 섞지 않는다 — 코드가 아닌 논의·결정은 Notion에, 실행 가능한 것만 여기에 둔다.

## 환경

```
conda env create -f environment.yml
```

라벨링 실행에는 Codex CLI 로그인(`codex exec`)도 필요하다.

## 라벨링 재현 순서

NSD·COCO 데이터를 `data/`에 준비했다는 전제다. 자세한 설명은 `docs/labeling_pipeline.md`.

```
python3 src/labeling/build_pool.py            # 후보 풀 (35,977장)
python3 src/labeling/sample_pilot.py          # 파일럿 300장
python3 src/labeling/extend_pilot.py          # 코어 1,000장
python3 src/labeling/scale_queue.py --unit-size 50 --seed 1
python3 src/labeling/download_pool.py
python3 src/labeling/run_scale.py --pass 0 --units 0:20 --workers 3 --model gpt-5.5 --effort medium
python3 src/labeling/report_stats.py          # 결과 수치
```

이후 단계(사전 점검 → 축 추출 → decoder → 평가)의 명령은 `scripts/`에 스크립트가 추가되는 대로 채운다.
