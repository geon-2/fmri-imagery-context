# 맥락 decoder 추가 실험(야간 실행)과 전체 계획 목록 (2026-10-27, 실행 전에 작성)

## 1. 계획 목록 (논의한 것 전부, 상태 표시)

| # | 계획 | 상태 | 비고 |
|---|---|---|---|
| 0 | MIRAGE 재현 기준선(공식 7지표 7/7, 사람 평가 2AFC 1명) | 완료 | `docs/mirage_server_run_log.md` |
| 1 | **맥락 decoder를 하나 더 두고 텍스트 조건에 합치기**(이 문서의 실험) | 오늘 밤 실행 | 피험자 1, retrieval 없음 |
| 2 | 합치는 규칙 변형(가중치 α 탐색, 맥락+캡션 이어 붙여 한 문장으로 학습, 맥락만 교체) | 1번 결과 후 | `ctxonly100` 조건이 교체 방식의 첫 확인 |
| 3 | ROI별 decoder 분리(방향 1) | 미정 | 지각 shared1000에서 임베딩 식별 지표로 값싸게 먼저 거른 뒤 생성 |
| 4 | 입출력 차원 축소(방향 3) | 미정 | 비열등성 + 자원 측정. 생성 시간은 안 줄어듦 |
| 5 | 생성 효율(retrieval 후보 수 N, 샘플링 단계 수) | 보류 | retrieval은 당분간 원본 그대로 유지(사용자 결정) |
| 6 | 평가 보강: (피험자, 자극) 짝지은 부트스트랩, shared1000 비전 지표, 사람 평가(2AFC) 도구 재사용 | 필요 시 | 12자극은 검정력이 낮음 |
| 7 | 확인 항목: LLaVA 캡션 파일 검토(허깅페이스 다운로드 필요), 디코더 학습에 imagery 자극이 없는지(확인됨: shared1000 제외) | 일부 완료 | |
| 8 | **Long-CLIP(248토큰) 활용**(사용자 결정 2026-10-28). 자리: (a) 지각 decoding에서 인코더 비교(잘림 없는 전체 맥락 문장, ViT-L, MPNet과 비교, 12번의 선행 게이트), (b) 후보 선택(retrieval)용 결합 공간, (c) **12번: Long-CLIP 타깃 decoder + Stage C 수정** | 후보 | 수정하지 않은 Stage C의 기존 입구(bigG 77×1280, 1280 pooled)에는 Long-CLIP을 못 넣지만, Stage C에 입구를 추가해 학습하면(12번) 쓸 수 있다. 설계 결정이라 비교 규칙을 정해 Notion에 먼저 기록(사용자 요청 후). 체크포인트 다운로드 필요 |
| 9 | 생성기에 새 입구(학습 없음) B: `clip_text`를 [LLaVA 77 ; 맥락 77] = 154토큰으로 이어 붙임(맥락은 bigG의 T10 또는 P1 평균) | 후보(오늘 밤 결과 후) | Stage C는 조건을 하나의 시퀀스로 이어 붙여 어텐션에 넣으므로 길이에는 형식상 열려 있음. 77에서만 학습된 모델이라 동작은 실험 필요 |
| 10 | 생성기에 새 입구(학습 없음) C: MIRAGE가 무조건 임베딩으로 중립화한 `clip_text_pooled`(1280→4토큰)를 맥락 요약 벡터(1280차원 릿지)로 채움 | 후보 | 가장 가벼움. bigG 풀링 벡터라야 하므로 Long-CLIP은 못 씀 |
| 11 | 생성기 쪽 D: 조건별 노이즈 예측을 합성(composable guidance) | 보류 | 샘플러 수정과 Stage C 시간 1.5~2배. 학습 불필요 |
| 12 | **목표 방향(사용자 결정 2026-10-28)**: 맥락 전용 Long-CLIP 임베딩을 타깃으로 하는 decoder를 따로 학습하고, **Stage C에 맥락 전용 입구를 추가해 학습**한다(IP-Adapter식). 설계와 게이트는 9절 | 게이트 통과 후 | 오늘 밤 결과(1번), 8-(a) 지각 decoding 비교, 학습 가능성 시험(9절)이 선행 |

## 2. 오늘 밤 실험 설계

**질문**: MIRAGE의 텍스트 조건(Stable Cascade Stage C의 77×1280)에, 우리 맥락 문장에서 학습한 decoder의 예측을 더하면 imagery 재구성이 달라지는가.

**구성**
- 기존 텍스트 릿지(LLaVA 캡션 타깃)는 그대로 둔다. 맥락 문장(T10: 맥락 구를 place_type, environment, activity, scale, implied_event, notes 순으로 잇고 CLIP 토큰 75 이하로 자름) 을 bigG 텍스트 인코더에 통과시킨 77×1280을 타깃으로 하는 **릿지를 하나 더 학습**한다(α=1e5, 같은 40세션 beta, 문장이 없는 이미지 3장은 학습에서 제외).
- 추론에서 두 예측을 임베딩 공간에서 가중 평균한다: `pred_text = (1−α)·LLaVA 예측 + α·맥락 예측`. 맥락 예측도 원본 텍스트 예측과 같은 방식으로 테스트 표본끼리 표준화한 뒤 학습 평균·표준편차로 재스케일한다(`normalize_preds`).
- 다른 모든 것(CLS 릿지, VDVAE 릿지와 필터, Stable Cascade 설정, retrieval 없음, 시드 방식)은 기준선과 같다. 이미지 CLS, 텍스트(LLaVA), VDVAE 릿지 가중치는 기준선과 같은 파일을 심볼릭 링크로 쓴다.

**조건** (피험자 1, 40세션, retrieval 없음)
| 조건 | α | 맥락 릿지 | 시드 | 역할 |
|---|---|---|---|---|
| 기준선 | 0 | - | orig + 1~4 (이미 있음, 맥에 저장) | LLaVA만 |
| `ctxblend050` | 0.5 | 정상 | 1~5 | 주 조건 |
| `shufblend050` | 0.5 | **이미지 단위로 문장을 섞어 학습** | 1~3 | 대조: 맥락 내용 때문인지, 합치기 자체 때문인지 |
| `ctxonly100` | 1.0 | 정상 | 1~3 | LLaVA 없이 맥락만(교체 방식 첫 확인) |
| `p1seqblend050` | 0.5 | **구를 하나씩 bigG로 인코딩해 토큰 위치별로 평균(잘림 없음)** | 1~3 | 잘림의 영향 확인: T10과 같은 α에서 모든 구를 쓰는 표현 |

**소요**: J0(맥락 릿지 학습 2개 + 모델 폴더)는 **실측 5분 21초**(문장 임베딩 약 1분, 릿지 각 약 2분; 처음 예상 1시간은 과대). 생성이 시간 대부분이며 시드당 약 38분 예상: T10 세 조건(J1~J4, 시드 11개)은 약 7.5시간. P1 평균 조건(J5~J6, `ctx/submit_ctx_p1.sh`)은 구 임베딩(약 9만 번, 문장 임베딩 속도로 보면 약 10분 안팎) + 릿지 약 2분 + 시드 3개로 약 2.5시간 추가(전체 약 10시간).

**P1 평균 조건의 의미**: 구 각각은 짧아서 77토큰 한도에 걸리지 않는다. 각 구의 bigG 77×1280 시퀀스를 토큰 위치별로 평균하면 지각 실험의 P1(구별 임베딩 평균)에 대응하는 표현이 MIRAGE의 텍스트 입력 형태로 만들어진다. 위치가 다른 토큰이 섞이는 방식이라 의미가 불분명하다는 한계가 있고, 이 방식이 Stage C에서 잘 동작하는지는 모른다.

## 3. 사전 보고 규칙 (결과를 보기 전에 정함)

이번은 **탐색(파일럿)** 이다. 합격/불합격 숫자 기준을 두지 않고 아래만 보고한다(`CLAUDE.md`: 기준 숫자는 파일럿 분포를 보고 확정).
1. 각 조건과 기준선의 공식 7지표 차이와 95% 구간(자극 12개 복원추출, 짝지은 비교). 반복(시드)은 평균.
2. 자극 그룹별(단순 도형 0~5, 자연 장면 6~10, 그림 11)도 같이 표시. 자연 장면이 5개뿐이라 그룹별은 참고용.
3. 해석 문장 규칙: (a) 구간이 0을 포함하면 "차이를 확인하지 못함". (b) `ctxblend050`이 기준선보다 높고 `shufblend050`보다도 높을 때만 "맥락 내용이 기여했을 가능성"이라고 쓴다. (c) `shufblend050`이 `ctxblend050`과 비슷하면 "합치기 자체의 효과일 수 있음"이라고 쓴다. (d) 피험자 1명, 자극 12개이므로 어떤 경우에도 결론이 아니라 탐색 결과로만 쓴다.
4. 사람이 직접 보는 비교 그림(원본, 기준선, 각 조건)을 같이 만든다. 눈으로 본 인상은 지표와 따로 적는다.

## 4. 실행 방법

서버에서 먼저, **기준선 공용 파일 두 개를 백업**하고 실험 폴더를 만든다(덮어써도 되돌릴 수 있게):
```bash
ssh moana 'cd /data/sslmyo/mirage_server && cp -n 04_pipeline.sh 04_pipeline.sh.before_ctx && cp -n run_nb.py run_nb.py.before_ctx && mkdir -p ctx && ls'
```
로컬(맥)에서 파일을 올린다(SFTP 확장은 한 번 파일을 0바이트로 만든 적이 있어 scp 권장). 공용 두 파일은 위쪽 폴더로, 실험 전용 파일은 `ctx/`로:
```bash
cd /Users/gunlee/Development/fmri-imagery-context
scp scripts/mirage_server/04_pipeline.sh scripts/mirage_server/run_nb.py moana:/data/sslmyo/mirage_server/
scp scripts/mirage_server/ctx/train_ctx_ridge.py scripts/mirage_server/ctx/make_ctx_models.sh scripts/mirage_server/ctx/ctx_preflight.sh scripts/mirage_server/ctx/submit_ctx.sh data/results/mirage_ctx/context_T10_by_nsdid.json moana:/data/sslmyo/mirage_server/ctx/
```
서버 moana-u4에서:
```bash
cd /data/sslmyo/mirage_server && bash ctx/ctx_preflight.sh
```
"전부 OK"가 나오고 문장 파일 md5가 `9486eabaee41f30b61d92fd23167705c`(36,977개)로 일치하면:
```bash
bash ctx/submit_ctx.sh dry    # 명령만 확인
bash ctx/submit_ctx.sh        # 제출
squeue -u $USER
```
학습 작업(J0)의 처음 20~30분은 로그를 보는 것이 좋다: `tail -f logs/slurm-<J0 번호>.out`. 문장 임베딩과 첫 릿지가 지나가면 나머지는 자동으로 이어진다.

### 4-2. P1 평균 조건 추가 (T10 조건 제출 뒤 또는 같이)
맥에서 바뀐 파일과 구 목록 파일을 올린다:
```bash
cd /Users/gunlee/Development/fmri-imagery-context && scp -P <포트> scripts/mirage_server/ctx/train_ctx_ridge.py scripts/mirage_server/ctx/make_ctx_models.sh scripts/mirage_server/ctx/submit_ctx_p1.sh data/results/mirage_ctx/context_phrases_by_nsdid.json <계정>@<로그인서버>:/data/<계정>/mirage_server/ctx/
```
서버에서 (T10 조건을 이미 제출했다면 그 마지막 작업 번호 J4를 인자로 준다):
```bash
cd /data/sslmyo/mirage_server && bash ctx/submit_ctx_p1.sh dry <J4 번호>
```
```bash
bash ctx/submit_ctx_p1.sh <J4 번호>
```
주의: T10 학습(J0)이 아직 시작하지 않았다면 새 `train_ctx_ridge.py`가 J0에서 쓰여도 동작은 같다(기본 변형 `ctx,shuf`는 그대로).

## 5. 아침에 확인하는 방법

서버 결과를 맥으로 받는다:
```bash
for m in ctxblend050 shufblend050 ctxonly100; do mkdir -p data/results/mirage_server/subj01_40sess_$m; rsync -av moana:/data/sslmyo/mirage_results/subj01_40sess_$m/ data/results/mirage_server/subj01_40sess_$m/; done
```
평가(기준선은 `subj01_40sess_server_ridge_sc2`의 원본 + 시드 1~4):
```bash
B=data/results/mirage_server/subj01_40sess_server_ridge_sc2; R=data/results/mirage_server
python3 -m src.eval.nsd_imagery_eval --gt $R/imagery_stimuli_18.pt --no-swav --baseline base --out results/tables/ctx_blend_eval \
  --recons base=$B/subj01_40sess_server_ridge_sc2_all_recons_imagery.pt,$B/subj01_40sess_server_ridge_sc2_all_recons_imagery_seed1.pt,$B/subj01_40sess_server_ridge_sc2_all_recons_imagery_seed2.pt,$B/subj01_40sess_server_ridge_sc2_all_recons_imagery_seed3.pt,$B/subj01_40sess_server_ridge_sc2_all_recons_imagery_seed4.pt \
  blend=$(ls $R/subj01_40sess_ctxblend050/*_all_recons_imagery_seed*.pt | paste -sd, -) \
  shuf=$(ls $R/subj01_40sess_shufblend050/*_all_recons_imagery_seed*.pt | paste -sd, -) \
  ctxonly=$(ls $R/subj01_40sess_ctxonly100/*_all_recons_imagery_seed*.pt | paste -sd, -)
```

## 6. 위험과 한계

- 합치기는 임베딩 공간의 선형 평균이다. Stage C가 이런 평균 임베딩에서 자연스럽게 동작하는지는 모른다(프롬프트 보간처럼 되길 기대하지만 확인하지 않음). 이 점은 `ctxonly100`과 `shufblend050`으로 일부 가려낸다.
- 맥락 라벨은 사물 이름을 제외하도록 만든 것이라 텍스트만으로는 장면 내용이 LLaVA 캡션보다 덜 구체적이다. 그래서 α=1은 오히려 불리할 수 있다.
- 피험자 1명, 12자극이라 작은 차이는 구분하지 못한다.
- 서버 디스크(/local_datasets)가 거의 찬 상태일 수 있다. 맥락 릿지 2개(각 6.2GB)가 필요하다. `ctx_preflight.sh`가 여유를 출력한다.
- 학습(J0)이 실패하면 뒤 작업은 모두 취소된다. 생성 작업 하나가 실패해도 다음 조건은 이어서 돈다. 맥락 합치기 셀이 실패하면 즉시 커널을 종료시켜 LLaVA만 쓴 결과가 다른 이름으로 저장되는 일을 막았다.
- 로컬에서 확인한 것: 합치기 수식(합성 데이터), 노트북 패치 위치와 문법, 학습 스크립트의 순수 함수. 서버에서 처음 도는 것: 문장 임베딩(bigG), lean ridge 학습, 합쳐진 조건으로 한 생성.

## 7. 코드와 환경 분리 원칙 (기준선과 개선 실험이 섞이지 않게)

- **코드 위치**: 기준선(재현)은 `scripts/mirage_server/` 맨 위 파일들이고, 이번 실험 전용은 `scripts/mirage_server/ctx/` 아래다(서버도 같은 구조). 공용 파일 두 개(`04_pipeline.sh`, `run_nb.py`)의 변경은 **추가만(34줄, 삭제 0줄)**이고, 맥락 환경변수(`CTX_WEIGHTS` 등)나 새 단계 이름(`ctxtrain`, `ctxmodel`)을 쓰지 않으면 기준선과 동작이 같다. 서버에는 `*.before_ctx` 백업을 남긴다.
- **이름 분리**: 모델 폴더 `subj01_40sess_{ctxblend050,shufblend050,ctxonly100}`, 맥락 릿지 `train_logs/ctx_T10_subj01/`, 맥락 통계 `dataset/preprocessed_data/subject1/ctxT10_textstats_40sess.pt`(기준선 임베딩 연결 스크립트의 파일 패턴과 겹치지 않는 이름), 결과 `/data/sslmyo/mirage_results/<조건 이름>/`와 맥의 `data/results/mirage_server/<조건 이름>/`. 기준선 결과 폴더는 건드리지 않는다.
- **기준선 가중치는 읽기만**: 조건별 모델 폴더의 LLaVA 릿지는 기준선 no-retrieval 모델 폴더 `subj01_40sess_server_ridge_sc2`(기준선 재구성을 만든 바로 그 가중치)의 파일을 심볼릭 링크로 가리킨다. 그 폴더를 지우면 링크가 깨지므로 실험이 끝나기 전에는 지우지 않는다. 이 실험은 `clean` 단계를 쓰지 않는다.
- **패키지 환경**: 새 패키지를 설치하지 않는다(서버는 기존 `mei-env`, 맥은 표준 라이브러리, numpy, torch, transformers만). 서버의 `deps_probe.py`는 기존 동작 그대로다.
- **동시 실행 금지**: `recon_srv.ipynb`, `recon_srv_out.ipynb`는 서버의 `MIRAGE/src`에서 작업마다 덮어쓰는 공용 이름이다. GPU가 1장이라 작업은 직렬이지만, 기준선 재실행을 이 실험과 겹쳐 돌리지 않는다.
- **설계 결정**: 이 실험 결과로 설계가 바뀌면 `CLAUDE.md` 규칙대로 Notion에 먼저 기록하도록 사용자에게 요청한다(Notion 쓰기는 사용자 요청 후).

## 8. 결과를 읽는 방법 (결과를 보기 전에 작성, 2026-10-27)

**잘림 현황(공식 라벨, 피험자 1 학습 이미지 8,997장)**: 맥락 문장(T10, bigG 75토큰 한도)에서 **47.4%의 이미지가 한 구 이상 잘리고**, 잘린 구는 전체의 6.8%다. 모든 구를 이으면 필요한 토큰은 중앙값 75, 90% 93, 최대 약 160이다. 잘림은 문장 뒤쪽 차원에 몰린다: place_type, environment, activity 거의 0%, scale 2.4%, **implied_event 41.9%, notes 76.6%**. 이번 실험의 맥락 decoder는 "장소, 환경, 활동"은 온전히, "암묵적 사건·의도"는 절반가량만 담은 문장을 학습한다.

**부정 결과(개선 없음)를 이렇게 읽는다.** 이번 실험이 검증하는 것은 아래 구체적 구성이다.
> 75토큰으로 자른 맥락 문장(T10)의 bigG 시퀀스를 릿지로 예측해 LLaVA 텍스트 예측과 임베딩 공간에서 평균해 Stable Cascade 텍스트 조건으로 넣었을 때, 피험자 1의 imagery 재구성(12자극)이 달라지는가.

따라서 개선이 없어도 다음은 **주장하지 않는다**: "맥락 정보는 MIRAGE 재구성에 도움이 되지 않는다." 대신 다음만 쓴다: "이 구성에서는 차이를 확인하지 못했다."
가능한 다른 설명(어느 것도 이번 실험이 가려내지 못한다):
1. **잘림**: implied_event 등이 빠졌다. 지각 decoding에서 잘림 없는 평균 표현(P1)은 T10보다 top-1이 약 0.02(상대 약 15%) 높았다. 이 차이가 하한 근거이며, 생성에서의 영향은 측정하지 못했다.
2. **표현 불일치**: 토큰별 은닉 상태 77×1280(98,560차원)은 풀링된 벡터보다 예측이 어렵고, 지각 실험에서도 맥락 decoding은 원본 캡션보다 낮았다(top-1 0.135 대 0.169). 맥락 정보가 없어서가 아니라 읽어내기가 약해서일 수 있다.
3. **합치기 방식**: 임베딩 공간의 선형 평균(α=0.5)이 Stage C에 자연스러운 입력인지 모른다. `ctxonly100`은 α=1 한 점만 본다.
4. **맥락 라벨의 성격**: 사물 이름을 제외하도록 만든 라벨이라 장면 내용 정보는 LLaVA 캡션보다 적다.
5. **검정력**: 피험자 1명, 자극 12개(자연 장면 5개)라 작은 차이를 구분하지 못한다.

**긍정 결과(개선 있음)를 이렇게 읽는다.** 셔플 대조(`shufblend050`)보다 높을 때만 "맥락 내용이 기여했을 가능성"으로 쓴다. 셔플과 비슷하면 합치기 자체의 효과일 수 있다. 어느 경우에도 확인(다른 피험자, 더 많은 자극)이 필요한 탐색 결과다.

**잘림의 영향을 직접 가리려면(이번 실험 밖)**: 지각 데이터에서 잘림 없는 문장(Long-CLIP 248토큰)과 T10의 decoding을 같은 방식으로 비교한다(계획 목록 8번). 생성기 쪽에서는 bigG 한도(75토큰) 안에서 차원별로 핵심 구를 균형 있게 고르는 문장(implied_event 포함)을 다음 후보로 둔다.

**P1 평균 조건(`p1seqblend050`)을 읽는 법(결과를 보기 전에 작성)**: T10과 같은 α=0.5에서 모든 구를 쓰는 표현이다. (a) `p1seqblend050`이 T10 조건보다 눈에 띄게 낫다면 잘림이 영향을 줬을 가능성이 있다(두 표현의 다른 차이, 즉 평균 방식 자체의 효과가 섞여 있어 단정은 못 한다). (b) 둘 다 기준선과 차이가 없으면 "잘림이 유일한 원인은 아닐 가능성"의 근거로 쓰되, 위 2~5번 설명(읽어내기가 약함, 합치기 방식, 라벨 성격, 검정력)이 남는다. (c) 둘 다 좋지 않다고 해서 맥락이 쓸모없다고 쓰지 않는다.

## 9. 목표 방향 12번(Long-CLIP 타깃 decoder + Stage C 수정)의 설계와 게이트 (2026-10-28, 실행 전 작성)

**구조(초안)**
- **decoder**: 복셀 → Long-CLIP 텍스트 임베딩(전체 맥락 문장, 잘림 없음, 768차원 풀링 벡터를 첫 타깃으로)을 예측하는 릿지 하나(기존 릿지들과 같은 방식, α=1e5). 필요하면 이후 토큰 시퀀스 타깃을 추가한다.
- **Stage C 수정**: 현재 입력은 `clip_text`(T×1280), `clip_text_pooled`(1280→4토큰), `clip_img`(768→4토큰)를 이어 붙인 시퀀스다(`modules/stage_c.py`). 여기에 새 매퍼 `Linear(768 → c_cond × 4)`를 추가해 맥락 토큰 4개를 시퀀스에 더한다. 학습하는 것은 새 매퍼(와 필요하면 LoRA)뿐이고 나머지는 얼린다.
- **학습 데이터와 손실**: 학습 이미지와 맥락 라벨이 모두 있는 NSD 이미지(피험자 1~4 합쳐 약 3.6만 장). 정답(실제) 임베딩(EffNet 잠재, LLaVA 캡션의 bigG 시퀀스, CLS, 정답 맥락의 Long-CLIP 임베딩)을 조건으로 Stage C의 기존 확산 손실로 학습한다. 맥락 입구는 일부 확률로 비워(조건 드롭아웃) 기존 동작과 guidance가 유지되게 한다. 테스트에서는 예측 임베딩을 넣는다(MIRAGE와 같은 방식).
- 다른 모든 것(VDVAE 흐린 이미지 시작점, CLS 릿지, 텍스트 릿지)은 그대로이고, 기준선(수정 전 Stage C)과 같은 시드로 짝지어 비교한다.

**게이트(위에서 아래로, 통과해야 다음으로)**
1. 오늘 밤 결과(1번): 이미 있는 입구에 맥락을 섞어 넣은 최소 신호를 본다. 결과가 어떻든 12번을 막지는 않지만 해석의 기준점이 된다.
2. **8-(a)**: 지각 데이터에서 Long-CLIP 맥락 임베딩이 fMRI에서 읽히는가(ViT-L, MPNet 대비). 읽히지 않으면 입구를 만들어도 입력이 잡음이다.
3. **학습 가능성 시험**: A2000 12GB에서 Stage C(3.6B, bf16 가중치 약 7.2GB)의 얼린 가중치 위에 매퍼만 학습하는 한 스텝의 메모리와 시간을 잰다(그래디언트 체크포인팅, 배치 1~2). 대략 추정(검증 안 함): 표본당 순전파+역전파가 1초 안팎이면 3.6만 장 1에폭이 10시간 안팎이라 가능하고, 몇 초면 불가능하다. 불가능하면 더 작은 Stage C 변형(1B, 확인 필요)이나 다른 GPU를 검토한다.
4. 준비물: EffNet 잠재(약 3.6만 장), LLaVA 캡션 파일(허깅페이스 다운로드, 사용자 허락 필요), Long-CLIP 체크포인트(다운로드, 사용자 허락 필요), 정답 맥락의 Long-CLIP 임베딩.

**사전 보고 규칙(초안)**: 오늘 밤 규칙(3절)과 같다. 기준선 대비 짝지은 차이와 구간만 보고하고 합격 기준 숫자는 파일럿 후에 정한다. 맥락 입구를 비운 수정 Stage C(입구에 0을 넣은 경우)가 기준선과 같은 결과를 내는지도 확인해, 수정이 기존 동작을 망가뜨리지 않았는지 본다.

## 10. 결과 (2026-10-28, 피험자 1, 자극 12개: 단순 6, 자연 5, 그림 1)

조건: blend(T10, α=0.5, 시드 5개), shuf(셔플 대조, 시드 3), ctxonly(α=1, 시드 3), p1seq(P1 평균, α=0.5, 시드 3). 기준선은 no-retrieval 5회. 표와 구간은 `results/tables/ctx_blend_eval/`(git 무시)에 있고, 직접 비교는 `scripts/mirage_eval/ctx_blend_paired.py`, 그림은 `scripts/mirage_eval/make_ctx_view.py`(`docs/report_figures/ctx_view_*_seed1.png`)로 다시 만든다.

- **합치기가 적용됐는지**: 같은 시드의 기준선과 이미지가 같지 않다(평균 픽셀 차이 0.07~0.09). 기준선끼리 시드를 바꾼 차이는 0.20이다. 세 조건(blend, shuf, p1seq)의 차이 크기가 비슷해, 이 크기만으로는 맥락 내용의 효과를 말할 수 없다.
- **기준선 대비(공식 7지표, 구간은 자극 12개 복원추출 5000회)**: 구간이 0을 제외한 것은 28개 중 3개뿐이다(ctxonly EffNet-B +0.019, p1seq Alex(5) +0.048, p1seq EffNet-B +0.020). 보정이 없는 탐색이라 우연일 수 있고, 셔플도 EffNet-B에서 같은 방향(+0.013, 구간은 0 포함)이다. blend는 모든 지표에서 0을 포함한다.
- **직접 비교(시드 1~3)**: blend − shuf, p1seq − shuf, ctxonly − shuf는 모든 지표에서 0을 포함한다. p1seq − blend는 EffNet-B만 +0.011 [+0.000, +0.022]로 경계다.
- **읽기**: 3절 (b)를 충족하지 못했고(합치기가 기준선과 셔플 모두보다 높지 않음) (c)에 해당한다. 8절 P1 규칙으로는 (b)다(T10과 P1 모두 기준선과 차이가 없으므로 잘림이 유일한 원인은 아닐 가능성의 근거지만, 8절의 2~5번 설명은 남는다). 쓸 수 있는 문장은 "이 구성에서는 차이를 확인하지 못했고, 틀린 맥락과도 구분되지 않았다"까지다. 맥락 정보 자체의 유무에 대한 결론이 아니다.
- **눈으로 본 인상(시드 1 한 장씩, 지표와 따로)**: 장면의 큰 틀은 모든 조건에서 같고 세부만 바뀐다. 원본에 더 가까워진 조건은 보이지 않는다(자극 6 서퍼, 9 강은 모든 조건에서 반영되지 않음). blend와 shuf는 구분되지 않고, ctxonly가 상대적으로 더 달라진다. 단순 도형 6개는 어느 조건에서도 선을 닮지 않는다(기준선과 같은 현상).
