# CLAUDE.md — 이 저장소에서 Claude Code가 지켜야 할 것

## 프로젝트
"떠올린 장면에 남는 맥락" — perception fMRI 기반 저차원 context 축의 mental imagery 전이 분석.
연구 설계·결정의 원본은 Notion(`docs/links.md` 참고)이다. 이 파일은 **실행** 지침이다.

## 최초 실행 시 준비 (한 번만)

1. **Notion MCP 연결** — 아직 `/mcp`로 notion이 연결돼 있지 않으면:
   ```
   claude mcp add --transport http notion https://mcp.notion.com/mcp
   ```
   실행 후 `/mcp`로 OAuth를 완료한다. **주의: Notion 쓰기는 메인 세션(orchestrator)만
   사용한다.** 아래에서 만드는 라벨링 서브에이전트들에는 Notion MCP 도구를 허용하지
   않는다(`tools:` 목록에서 제외) — 텍스트 판단만 하면 되는 작업에 쓰기 권한을 줄
   이유가 없다.

2. **서브에이전트 파일 생성** — `.claude/agents/`에 아래 5개 파일이 없으면 만든다.
   (`/agents` 명령으로 대화형으로 만들어도 되고, 이 명세대로 직접 파일을 써도 된다.)
   각 파일의 `tools`는 `Read, Write`만 허용한다(코드 실행·Notion·웹 접근 없음 —
   순수 텍스트 판단 작업이기 때문). 단, critique-b는 로컬 래퍼 스크립트를 부르므로
   아래에 적은 예외를 따른다.

   모델 배정(단계별 역할에 맞춰 나눔 — 판단이 어려운 단계에 큰 모델을 쓴다):
   extract=`haiku`, classify=`opus`, critique-a=`sonnet`, adjudicate=`opus`,
   critique-b=Ollama `qwen2.5:14b`(Claude가 아닌 로컬 오픈소스 모델). critique-a와
   adjudicate가 같은 계열이 되지 않도록 critique-a는 `sonnet`으로 유지한다.

   - **`.claude/agents/context-extract.md`**
     (`model: haiku`) 역할: 캡션에서 맥락 관련 키워드 후보를 최대한 뽑는다(누락보다
     과다 추출 선호). 사물 이름·행동 자체는 후보에서 걸러내지 않아도 된다 — 그 판단은 다음 단계 몫이다.

   - **`.claude/agents/context-classify.md`**
     (`model: opus`) 역할: extract가 넘긴 후보를 4차원(장소 유형 / 환경 속성 / 활동·기능 / 공간 규모)에
     배정하거나 "제외"로 표시한다. 사물 이름·관계 술어는 반드시 제외한다.
     정해진 닫힌 어휘가 있으면 그 안에서만 고른다(파일럿 초반엔 어휘가 아직 없을
     수 있다 — 그 경우 자유롭게 제안하되 "제안 어휘"로 표시한다).

   - **`.claude/agents/context-critique-a.md`** (`model: sonnet`)
   - **`.claude/agents/context-critique-b.md`** (Ollama `qwen2.5:14b` 래퍼)
     역할: 둘 다 같은 프롬프트 — classify 결과를 보고, 이 라벨이 특정 COCO object
     category를 강하게 암시하는지(누수) 판정한다. 왜 서로 다른 모델을 쓰는지: 한
     모델의 편향으로 결과가 좌우되지 않게 하려는 것이므로, critique-a는 Claude
     (Sonnet) 서브에이전트, critique-b는 Ollama로 돌리는 로컬 오픈소스 모델을
     감싼 래퍼 스크립트(`src/labeling/critique_b_ollama.py`)로 한다.
     critique-b의 `.md` 파일은 래퍼 호출 방법을 안내하는 문서이고, 프롬프트 본문은
     `context-critique-a.md`의 본문을 래퍼가 읽어와 그대로 쓴다(두 벌로 관리하지
     않는다). 이 파일만 `tools`에 `Bash`가 필요하다(래퍼 실행용) — 그 외 권한은
     주지 않는다. 로그의 `agent_model`에는 실제 Ollama 모델명을 남긴다. API 키가
     필요 없는 구성이므로 별도 벤더 키는 전제하지 않는다.

   - **`.claude/agents/context-adjudicate.md`**
     (`model: opus`) 역할: 집계된 라벨 분포와 critique-a/critique-b 판정을 받아 라벨별
     지지도·누수 플래그를 붙인다. **라벨을 삭제하거나 하나로 확정하지 않는다.** critique
     둘의 판단이 갈리면 자동으로 밀어붙이지 않고 "review" 상태로 표시한다.

## 라벨 구조 원칙 — 해석의 여지를 라벨링 단계에서 닫지 않는다

같은 장면도 사람마다 다르게 해석·표현할 수 있으므로, 라벨링 단계는 여러 해석을 **보존**하고
해석을 어떻게 합칠지는 representation 추론 단계에서 정한다.

- 다중 라벨 허용: 한 차원에 값이 여러 개여도 된다. 동의어 표현을 라벨링 단계에서 억지로
  통일하지 않는다.
- 분포 기반: 이미지당 classify를 N회 독립 샘플링(파일럿 기본 N=3)하고, 메인 세션이 값별
  `support`(나온 비율)와 `mean_confidence`로 집계한다. 라벨은 삭제하지 않는다.
- 누수(critique strong)는 삭제 사유가 아니라 **플래그**다. 누수 라벨을 빼고/빼지 않고
  분석하는 것은 이후 단계의 선택이다.
- 어휘는 단계적으로: (1) 어휘 없이(`vocab: null`) 샘플링해 표현 분포를 본다 → (2) 사람이 표현을
  정리해 어휘를 만든다 → (3) 어휘가 생기면 어휘 전체에 대한 dense 점수와 임베딩 기반 보완을
  검토한다. 어휘는 정답 목록이 아니라 표현 통일용 사전이다.
- 이 원칙은 Notion에 아직 기록되지 않은 설계 결정이다. 사용자에게 Notion 기록을 요청했는지
  확인하고 진행한다.

## 파일럿 단계 (200~500장) — 서브에이전트로 직접 실행

메인 세션이 파일럿 이미지를 순회하며 위 5개 서브에이전트를 이미지당
`extract → classify × N(독립 샘플) → 집계(메인 세션) → (critique-a, critique-b 동시) → adjudicate`
순서로 호출한다. 파일럿은 300장이다(`data/processed/pilot_300.csv`, 시드 0, 후보 풀은
`src/labeling/build_pool.py`).
집계는 `src/labeling/aggregate.py`, 단계별 로그 기록은 `src/labeling/log_stage.py`를 쓴다
(메인 세션이 손으로 합치거나 로그를 쓰지 않는다).
사람이 지켜보며 이상한 사례를 바로 확인할 수 있는 규모이므로, 이 단계는 스크립트로
자동화하지 않고 서브에이전트 호출을 그대로 쓴다.

## 스케일업 단계 (전체 데이터) — 동적 워크플로우로 전환

파일럿 검증(아래 "검증 기준")을 통과하고 프롬프트가 안정되면, **같은 프롬프트 텍스트를
그대로** `claude_agent_sdk`의 `AgentDefinition`으로 옮겨 스크립트가 전체 이미지를
순회하게 한다. 서브에이전트 파일과 SDK 정의의 프롬프트가 갈라지지 않도록, SDK
스크립트는 `.claude/agents/*.md`의 프롬프트 본문을 파일에서 읽어와 그대로 쓴다
(복붙해서 두 벌로 관리하지 않는다).

## 로그 — 모든 단계, 모든 이미지

`data/labels/agent_logs/{stage}.jsonl` 에 한 줄씩 append (git 추적 안 함 —
`.gitignore` 참고. 대신 **요약 통계**는 커밋 대상인 `results/tables/labeling_summary.md`
에 남긴다):

```json
{
  "image_id": "...", "stage": "extract|classify|critique_a|critique_b|adjudicate",
  "agent_name": "...", "agent_model": "...", "prompt_version": "v1",
  "input": "...", "raw_output": "...", "parsed_label": {...},
  "flags": ["leakage_suspected", "disagreement", ...],
  "timestamp": "..."
}
```

프롬프트 자체(서브에이전트 파일 본문)를 고치면 파일을 바로 덮어쓰지 말고 파일 안
주석으로 버전을 올린다(`<!-- prompt_version: v2 -->`). 어떤 버전이 어떤 결과를 냈는지
로그의 `prompt_version` 필드로 재구성 가능해야 한다.

## 검증 기준 — 숫자를 먼저 정하지 않는다

파일럿 결과를 `results/tables/`에 표로 남긴 뒤, 그 분포를 보고 Notion 연구 노트에
기준(누수 허용치, kappa 임계값 등)을 확정한다. **기준이 아직 확정되지 않았다면,
코드에 하드코딩하지 말고 파일럿부터 실행하라고 사용자에게 알린다.**

파일럿에서 반드시 확인할 것:
- 누수: context 라벨만으로 COCO object category를 예측하는 선형 분류기 정확도
  (object-only 상한, 무작위 하한과 함께 보고)
- critique-a와 critique-b 간 일치도 (라벨 단위 Cohen's kappa; 다중 라벨이므로 라벨 집합 Jaccard도 함께) — review로 얼마나 많이
  넘어갔는지도 함께 집계
- 사람 라벨(200장, 아직 미작성 — 이 저장소에 없으면 먼저 요청)과의 일치도
- 환경 속성 ↔ COCO-Stuff, 장소 유형 ↔ Places365 상관
- 4차원 각각의 값 분포 (한 값에 쏠리면 어휘 재정의 필요 신호)

## 절대 하지 않는 것

- `data/raw/`, `data/processed/`의 실제 NSD 파생물을 git에 커밋하지 않는다
  (`git add -f`로 강제 추가하지 않는다).
- shared1000(및 NSD-Imagery 자연장면 5장에 대응하는 이미지)을 라벨링 파일럿이나
  축 추출 학습에 포함하지 않는다 — 반드시 학습 분할에서 제외를 먼저 확인한다.
- 검증 기준을 결과가 나온 뒤에 맞춰 정하지 않는다.
- 라벨링 서브에이전트(extract/classify/critique/adjudicate)에 Notion MCP 도구를
  주지 않는다. Notion 쓰기는 메인 세션이, 그것도 사람이 결과를 확인한 뒤 명시적으로
  요청했을 때만 한다 — 파일럿 도중 자동으로 Notion에 기록하지 않는다.
- Notion에 없는 새로운 설계 결정을 코드만으로 조용히 바꾸지 않는다 — 방향이
  바뀌면 먼저 Notion에 기록하도록 사용자에게 요청한다.

## 커밋 규율

작은 단위로 자주 커밋한다. 커밋 메시지에 "무엇을, 왜"를 적고, 설계 결정이
관련되면 어떤 Notion 노트에 해당하는지 한 줄 남긴다(`docs/links.md` 형식 참고).
