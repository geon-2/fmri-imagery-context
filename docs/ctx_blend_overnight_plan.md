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

**소요 예상**: J0 학습 약 1시간(문장 임베딩 + 릿지 2개), 생성은 시드당 약 38분. 전체 약 8.5시간.

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
- **기준선 가중치는 읽기만**: 조건별 모델 폴더의 LLaVA 릿지는 기준선 `..._ret16` 폴더의 파일을 심볼릭 링크로 가리킨다. 그 폴더를 지우면 링크가 깨지므로 실험이 끝나기 전에는 지우지 않는다. 이 실험은 `clean` 단계를 쓰지 않는다.
- **패키지 환경**: 새 패키지를 설치하지 않는다(서버는 기존 `mei-env`, 맥은 표준 라이브러리, numpy, torch, transformers만). 서버의 `deps_probe.py`는 기존 동작 그대로다.
- **동시 실행 금지**: `recon_srv.ipynb`, `recon_srv_out.ipynb`는 서버의 `MIRAGE/src`에서 작업마다 덮어쓰는 공용 이름이다. GPU가 1장이라 작업은 직렬이지만, 기준선 재실행을 이 실험과 겹쳐 돌리지 않는다.
- **설계 결정**: 이 실험 결과로 설계가 바뀌면 `CLAUDE.md` 규칙대로 Notion에 먼저 기록하도록 사용자에게 요청한다(Notion 쓰기는 사용자 요청 후).
