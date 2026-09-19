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
| RQ2 | 그 뇌 정렬 k축이 같은 차원의 뇌 비정렬 k축보다 imagery로 더 잘 전이되는가 | **주 가설** |
| RQ3 | 기억 관련 영역을 더하면 추가 이득이 있는가 | 추가 실험 |

전 단계에 **사전 점검**(context embedding과 뇌 반응의 공유 정보 유무를 RSA·encoding 예측력으로 확인)이 있다.
자세한 배경, 선행연구 비교, 방법론적 함정과 대응은 Notion 연구 노트를 참고한다.

## 데이터

- **NSD core** (Allen et al., Nat Neurosci 2022): perception 학습 데이터. subject 1/2/5/7.
- **NSD-Imagery** (Kneeland et al., CVPR 2025): imagery 평가 데이터.

두 데이터셋 모두 Natural Scenes Dataset Data Use Agreement를 따른다.
**원본·가공 데이터는 이 저장소에 절대 커밋하지 않는다** (`.gitignore` 참고). `data/` 아래 폴더는
로컬에서만 채우고, 다른 사람이 재현하려면 각자 DUA에 동의 후 원본에서 내려받아야 한다.

## 저장소 구조

```
data/
  raw/         # NSD 원본 (git 추적 안 함, 로컬 전용)
  processed/   # beta, ROI 마스킹 결과 등 (git 추적 안 함)
  labels/      # object/context 라벨·캡션 변형 (git 추적 안 함, 원본은 아래 참고)
notebooks/     # 탐색적 분석
src/
  labeling/    # context 4차원 정의, 닫힌 어휘 생성, LLM 라벨링, 검증(사람 200장, COCO-Stuff·Places365 대조)
  axes/        # 사전 점검(RSA·encoding) + 축 추출(PLS/RRR), PCA·무작위 기준선
  decoding/    # ridge decoder 학습 (perception → k개 축 점수), imagery 적용
  eval/        # 후보 순위 평가, permutation test, k-곡선
configs/       # 실험 설정 (yaml)
results/
  figures/     # 파이프라인 산출 그림 (git 추적 안 함)
  tables/      # 파이프라인 산출 표 (git 추적 안 함)
docs/          # 설계 메모 등 이 저장소에 둘 문서 (긴 논의는 Notion이 원본)
scripts/       # 실행 스크립트 (예: run_axes.py, run_eval.py)
```

## 기록의 원천

- **연구 노트·의사결정·미팅 기록**: Notion ("떠올린 장면에 남는 맥락" 프로젝트 페이지)
- **일정**: Notion 연구 일정 스케쥴(소캡디 미팅) + Google Calendar(주차별 작업, 준비 마감)
- **코드·실험 이력**: 이 저장소의 git 로그
- **발표 자료**: `docs/slides/`에 HTML 슬라이드·대본 보관 (버전 관리)

셋의 역할을 섞지 않는다 — 코드가 아닌 논의·결정은 Notion에, 실행 가능한 것만 여기에 둔다.

## 환경

```
conda env create -f environment.yml   # 또는
pip install -r requirements.txt
```

## 재현 순서

1. `src/labeling/` — context 라벨 생성·검증
2. `src/axes/` — 사전 점검 → 축 추출 (PLS/RRR, PCA·무작위 기준선 포함)
3. `src/decoding/` — decoder 학습 (NSD core만 사용, shared1000 제외) → NSD-Imagery 적용
4. `src/eval/` — 후보 순위, permutation test, k-곡선

각 단계의 정확한 커맨드는 `scripts/`에 스크립트가 추가되는 대로 이 섹션에 채운다.
