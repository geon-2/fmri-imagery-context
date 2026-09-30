# 라벨링 파일럿 요약 (300장)

- 관련 Notion 노트: "라벨링 구조 결정 — 해석을 라벨링 단계에서 닫지 않는다"
  (https://app.notion.com/p/3e224ea96d58814e8819e895821348e8)
- 파이프라인: extract(caption, haiku, v2) → classify(vision, opus, v3, N=3 독립 샘플) →
  집계(`aggregate.py`) → critique-a(sonnet, v2) → adjudicate(opus, v2)
- **critique-b(Ollama qwen2.5:14b)는 이번 파일럿에서 제외했다.** 배치 처리 시 스키마
  불안정·응답 누락이 반복됐고(로그 5건만 정상 완료), RAM 제약으로 순차 실행만 가능해
  병목이 됐기 때문이다. adjudicate에는 `critique_b: {"items": [], "overall": "not_available"}`로
  전달해 "critique 누락 → 해당 라벨 leak_b=null"만 적용되고 "critique 에러 → disagreement"
  규칙은 적용되지 않게 했다. 이 설계 변경은 코드로만 반영됐고, 아직 Notion에 기록되지
  않았다 — CLAUDE.md 규칙("Notion에 없는 새로운 설계 결정을 코드만으로 조용히 바꾸지
  않는다")에 따라 사용자 확인 후 Notion에 남겨야 한다.

## 완료 현황

| 단계 | 기록 수 | 이미지 수 | 비고 |
|---|---|---|---|
| extract | 300 | 300 | 전부 v2(캡션 기반) |
| classify | 960 | 300 | v3(비전 기반) 900건(=300×3), v2(캡션 기반, 배치1 재작업 이전 잔존분) 60건 |
| critique-a | 320 | 300 | v2, 일부 이미지(배치1의 구버전 캡션 기반 실행분)는 중복 로그 존재 |
| critique-b | 5 | 5 | 배치1 첫 5장에서 한 번 성공한 것 외 사용 안 함(위 사유로 제외) |
| adjudicate | 300 | 300 | 전부 v2, status 전원 "ok" |

파일럿 300장 전 이미지가 extract→classify(3샘플)→집계→critique-a→adjudicate까지
빠짐없이 완료됐다.

## 누수(leakage) 신호

- `leakage_suspected` 플래그가 붙은 이미지: **196/300 (65.3%)**
  (critique-a가 어떤 라벨이든 하나라도 `strong`으로 판정한 경우)
- `low_support` 플래그(모든 라벨의 support < 0.5): **0/300**
- adjudicate `status`는 300장 전부 "ok" — 이는 critique-b가 빠져 disagreement 판정이
  발생할 수 없기 때문이며, "review로 얼마나 넘어갔는지" 같은 검증 기준 항목은 이번
  파일럿에서는 계측할 수 없다.
- 반복적으로 strong 누수를 유발한 패턴: place_type이 활동 도구를 직접 암시하는 경우
  (예: "bathroom"→toilet/sink, "bedroom"→bed, "tennis court"→tennis racket), 활동 표현이
  COCO 카테고리명을 문자 그대로 포함하는 경우(예: "watching tv", "dining table"),
  scale/구도 표현("portrait", "medium shot of person")이 person 존재를 강하게 암시하는
  경우.

## 4차원 값 분포 (전체 300장, casefold 기준)

| 차원 | 고유 값 수 | 상위 값(빈도) |
|---|---|---|
| place_type | 853 | home interior(32), sidewalk(26), street(26), park(22), city street(22), living room(20) |
| environment | 1040 | outdoor(188), daytime(135), indoor(111), sunny(78), uncrowded(51), overcast(48) |
| activity | 1037 | resting(25), leisure(20), recreation(19), walking(17), commuting(15), transportation(14) |
| scale | 505 | medium shot(112), close-up(83), single room(39), street-level view(31) |

어휘가 아직 없는(`vocab: null`) 자유 제안 단계라 표현이 매우 분산돼 있다(예: place_type만
853개 고유값). 한 값에 크게 쏠리지는 않았지만, 동의어 파편화가 뚜렷해 다음 단계(사람이
표현을 정리해 어휘를 만드는 단계)가 필요하다는 신호로 해석된다.

## 아직 계측하지 못한 검증 기준 (critique-b 부재로 인해)

CLAUDE.md의 파일럿 검증 기준 중 아래 항목은 critique-b가 빠져 있어 이번 파일럿
결과로는 계측할 수 없다. 재계측하려면 critique-b를 다른 방식(예: 이미지 1장씩 호출,
또는 다른 로컬 모델)으로 다시 설계해 이 300장 각각을 재실행해야 한다.

- critique-a와 critique-b 간 일치도(라벨 단위 Cohen's kappa, 라벨 집합 Jaccard)
- review로 넘어간 비율(critique 간 불일치 비율)

## 아직 하지 않은 것

- 사람 라벨(200장, `data/labels/human_labels_200_en.csv`)과의 일치도 비교
- context 라벨만으로 COCO object category를 예측하는 선형 분류기(object-only 상한 대비)
- 환경 속성 ↔ COCO-Stuff, 장소 유형 ↔ Places365 상관 분석
- critique-b 제외 결정의 Notion 기록

---

# 파일럿 재실행 요약 (5차원, 300장) — v4/v3

- 관련 Notion 노트: "라벨링 구조 결정 — 해석을 라벨링 단계에서 닫지 않는다"
  (https://app.notion.com/p/3e224ea96d58814e8819e895821348e8)
- 이 절은 위 4차원 파일럿을 재설계해 다시 돌린 결과다. 이전 결과는 지우지 않고 위에 그대로
  남겨 두었다(버전별 재구성 가능해야 한다는 CLAUDE.md 원칙에 따름).
- **파이프라인 변경점**: extract 단계를 완전히 없앴다 — classify가 이미지를 직접 보고 처음부터
  라벨을 생성하므로 캡션 후보를 미리 추리는 중간 단계가 중복이었기 때문이다. 새 파이프라인은
  classify(vision, opus, **v4**, N=3 독립 샘플) → 집계(`aggregate.py`) → critique-a(sonnet,
  **v3**) → adjudicate(opus, **v3**)이다.
- **차원 변경점**: `implied_event`(암묵적 사건·의도) 차원을 새로 추가해 4차원을 5차원으로
  늘렸다. `activity`는 "지금 화면에 보이는 동작·용도"만 담도록 범위를 좁히고, 화면 밖
  서사(이전/의도/다음)는 `implied_event`로 분리했다. 근거: Bar(2004)의 하향식 context
  frame, VisualCOMET(Park et al., ECCV 2020)의 사건 추론 프레임워크, 스크립트 이론(Schank &
  Abelson).
- **critique 판정 기준 변경점**: critique-a/b 모두 "라벨 문자열에 사물 이름이 그대로 있으면
  무조건 strong"이라는 기계적 규칙을 없애고, 모델의 종합 판단에 맡기도록 프롬프트를 다시
  썼다 — 이 판정 자체가 정량적 정답이 없는 문제라서 애초에 두 모델로 이중화한 것이기
  때문이다.
- **critique-b는 이번 재실행 전체에서 한 번도 실행하지 못했다**(Ollama 배치 불안정 문제가
  이전 파일럿과 동일하게 남아 있음). adjudicate에는 이전과 같이
  `critique_b: {"items": [], "overall": "not_available"}`를 전달해 "critique 누락 →
  해당 라벨 leak_b=null"만 적용하고 "critique 에러 → disagreement"는 적용하지 않았다.

## 완료 현황

| 단계 | 기록 수 | 이미지 수 | 비고 |
|---|---|---|---|
| classify | 900 | 300 | 전부 v4(비전 기반, extract 없음), 이미지당 3개 독립 샘플 |
| critique-a | 300 | 300 | 전부 v3(종합 판단, 고정 규칙 없음) |
| critique-b | 0 | 0 | 이번 재실행에서 전혀 실행하지 못함(위 사유) |
| adjudicate | 300 | 300 | 전부 v3, status 전원 "ok"(critique-b 부재로 disagreement 계산 불가) |

파일럿 300장 전 이미지가 classify(3샘플)→집계→critique-a→adjudicate까지 새 파이프라인으로
빠짐없이 완료됐다.

## 누수(leakage) 신호

- `leakage_suspected` 플래그가 붙은 이미지: **252/300 (84.0%)** — 이전 4차원 파일럿의
  65.3%보다 높다. `implied_event` 차원이 새로 추가되면서 서사적 추론("about to eat",
  "just arrived")이 특정 COCO 사물을 직접 언급하는 경우가 늘어난 것이 주된 원인으로
  보인다(예: "diner photographing food before eating"→food 관련 카테고리, "train about to
  depart"→train). 이 차이가 실제로 `implied_event` 때문인지, 아니면 critique 판정 기준을
  완화한 효과인지는 이 집계만으로는 분리할 수 없다 — 사람 재검토가 필요하다.
- `low_support` 플래그: **0/300**
- adjudicate `status`는 300장 전부 "ok" — critique-b가 단 한 번도 실행되지 않아
  disagreement 판정 자체가 발생할 수 없었다. 즉 이번 재실행의 모든 "ok"는 critique-a
  단일 모델의 판정일 뿐, 두 모델이 일치했다는 뜻이 아니다.
- 배치별로 반복 관찰된 판단 경계(critique-a 서브에이전트들이 스스로 보고한 내용):
  - "person/people"이라는 단어가 라벨 문자열에 문자 그대로 들어가면 strong, "player/officer/
    pedestrian" 같은 역할 명사나 "walking/standing" 같은 동작 동사만 있으면 weak/none —
    이 경계가 배치마다 다소 흔들렸다(예: "full-body person"은 strong, "full-body player"는
    none으로 처리된 사례가 배치30에서 보고됨).
  - "kitchen→refrigerator/oven", "living room→couch/tv", "bathroom→toilet/sink" 같은
    place_type→가구/가전 강한 연상은 화장실 예시(CLAUDE.md 원 지침)를 다른 방 유형까지
    유추 적용한 것으로, 이 확장이 과도한지는 사람 검토가 필요하다는 지적이 배치19, 24에서
    반복됐다.
  - "tabletop"/"table" 단어가 들어간 scale·activity 라벨을 dining table로 strong 처리하는
    관행이 여러 배치(13, 16, 21)에서 나타났다 — COCO의 dining table 카테고리가 식탁뿐 아니라
    거실 테이블 등 모든 테이블류를 포괄한다는 점을 근거로 확장 적용한 것이다.

## 5차원 값 분포 (전체 300장, casefold 기준)

| 차원 | 고유 값 수 | 상위 값(빈도) |
|---|---|---|
| place_type | 1124 | home interior(30), city street(22), sidewalk(20), living room(20), park(17) |
| environment | 1592 | outdoor(186), indoor(107), daytime(105), sunny(58), open(36) |
| activity | 1343 | standing(24), walking(15), posing for a photo(14), sitting(13), resting(13) |
| implied_event | 2380 | 최빈값도 빈도 3 이하로 극도로 분산됨(예: "recent snowfall", "train about to depart") |
| scale | 958 | close-up(61), medium shot(53), near distance(33), single room(30), medium distance(29) |

이전 4차원 파일럿과 비교하면 place_type(853→1124)·environment(1040→1592)·activity(1037→1343)
모두 고유값 수가 늘었다. classify 프롬프트를 다시 쓰면서(v4) 표현이 더 세분화된 영향과,
어휘가 여전히 없는(`vocab: null`) 자유 제안 단계라는 점이 함께 작용한 것으로 보인다.
`implied_event`는 예상대로 가장 파편화된 차원이다(2380개 고유값, 최빈값도 3회뿐) — 서사
추론은 본질적으로 장면마다 고유하기 때문일 수 있으나, 어휘 정리 단계에서 가장 많은 통합
작업이 필요한 차원이라는 신호이기도 하다.

## 아직 계측하지 못한 검증 기준 (critique-b 부재로 인해)

이전 파일럿과 동일한 사유로 아래 항목은 이번 재실행에서도 전혀 계측하지 못했다.

- critique-a와 critique-b 간 일치도(라벨 단위 Cohen's kappa, 라벨 집합 Jaccard)
- review로 넘어간 비율(critique 간 불일치 비율)

## 아직 하지 않은 것 (재실행 기준)

- 새 5차원 라벨과 사람 라벨(200장) 간 일치도 비교
- 새 파이프라인의 84.0% 누수율이 이전 파이프라인의 65.3%보다 높아진 원인 분석(implied_event
  추가 vs. critique 기준 완화의 기여도 분리)
- critique-b 재설계(다른 로컬 모델 또는 호출 방식 변경) 및 이 300장 재판정
- place_type→가구/가전 강한 연상 확장 적용이 적절한지, place_type/activity(범주 연상이
  강함)와 environment(구조적으로 더 깨끗함)를 leak 허용 기준에서 다르게 취급할지 결정
- `scale` 차원을 공간 규모와 카메라 구도로 분리할지 결정(OPA의 openness/distance 관련
  문헌 근거로 이전에 제기된 질문, 아직 미결)
