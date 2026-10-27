# MIRAGE 아키텍처 검토 (2026-10-27, 개선 플랜 세우기 전)

원본 저장소(MedARC-AI/MIRAGE, 커밋 `7dbc13f`)의 `src/Train.ipynb`, `src/recon_inference_mi.ipynb`, `src/sc_reconstructor.py`, `src/utils.py`를 읽고 정리했다. 코드에서 직접 확인한 것은 파일과 줄 번호를 적었고, 추정은 "추정"이라고 표시했다. 읽기 전용으로만 봤고 실행하지 않았다.

## 한 장 요약

```
fMRI (nsdgeneral 15,724복셀, 단일 벡터)
  ├─ 릿지1 → CLIP 이미지 CLS        768           ─┐ c_i
  ├─ 릿지2 → CLIP 텍스트 시퀀스     77×1280=98,560 ─┤ c_t   (중간 길이 캡션의 텍스트 임베딩을 직접 예측)
  ├─ 릿지3 → VDVAE 잠재             91,168        ──→ VDVAE 디코드 → 흐린 이미지(선명도 20, 대비 1.5 필터)
  ├─ 릿지4 → CLIP 은닉(retrieval)   257×1024=263,168 → 후보 중 하나를 고르는 용도로만 사용(복셀은 L2 정규화 후 입력)
  └─ 릿지5 → GiT 특징               257×1024      → 캡션 문장 생성(캡션 지표 평가용, 이 코드에서는 생성에 쓰이지 않음)

각 예측 → 테스트 샘플 전체로 차원별 표준화 후 학습 통계로 재스케일(normalize_preds)

생성(Stable Cascade):
  흐린 이미지 → EffNet 잠재 → strength 0.7로 노이즈 추가
  Stage C 20×0.7=14단계, 조건: c_i(768), c_t(77×1280), textstrength 0.5로 섞음, cfg 4
  Stage B 10단계, 조건: Stage C 결과 + 빈 캡션
  Stage A 디코드 → 1024×1024
(retrieval 사용 시: 후보 16장 생성 → 예측 hidden과 코사인 최대인 1장 선택, 생성 비용 16배)
```

## 코드에서 확인한 사실

1. **decoder는 독립 릿지 5개다.** 모두 같은 입력(복셀 전체)에서 α=1e5로 따로 학습한다(`Train.ipynb` 릿지 학습 셀). 서로 정보를 공유하지 않는다.
2. **생성에 쓰이는 뇌 예측은 3개**(c_i, c_t, VDVAE 잠재)다. retrieval용 hidden은 후보 선택에만(`utils.pick_best_recon`, `utils.py:429-443`), GiT 예측은 캡션 파일 저장에만 쓰인다(`recon_inference_mi.ipynb`의 predcaptions 셀). 텍스트 조건 c_t는 문장을 만들지 않고 텍스트 임베딩을 곧장 예측한다.
3. **저수준은 VDVAE, 의미는 CLIP이 맡는다.** Stage C의 시작점이 흐린 이미지의 EffNet 잠재(`sc_reconstructor.py:125-136`)이고, c_i와 c_t가 조건이다(`:144-146`). 두 조건은 textstrength로 섞이며 무조건 임베딩이 일부 들어간다(`:156-161`). 도형 자극이 자연 사진으로 복원되는 것은 VDVAE 구조 정보가 약하고 CLIP 쪽이 의미를 만들어 내기 때문일 가능성이 있다(추정, 확인 안 함).
4. **normalize_preds는 예측을 테스트 샘플끼리 표준화한다**(`recon_inference_mi.ipynb`). 예측 차원마다 테스트 샘플 평균·표준편차로 맞춘 뒤 학습 평균·표준편차를 곱한다. imagery에서는 샘플이 18개(개념 자극 6개 포함)다. 릿지가 수축시킨 분산을 복구하는 역할이지만 평가 샘플 집합에 의존하는 단계다. 어떤 decoder를 바꾸든 이 단계를 똑같이 거쳐야 비교가 된다.
5. **학습은 이미지당 3회 반복을 각각 샘플로 쓰고**, shared1000은 학습에서 제외한다(`utils.py:1180-1183`, 세션 수는 `:1200`에서 `num_sessions*750`으로 자름). imagery 테스트는 trial을 평균한 beta를 쓴다(`load_nsd_mental_imagery`, `utils.py:674-743`). 학습 입력(단일 trial)과 테스트 입력(평균)의 잡음 수준이 다르다.
6. **ROI 선택용 인자가 코드에 이미 있다.** `load_nsd_mental_imagery`에 `snr`, `top_n_rois`, `whole_brain`, `nsd_general` 인자가 있고 `get_kastner_masks`(`utils.py:1393`) 등이 있다. 논문 기본 설정은 nsdgeneral 하나다. 이 인자들의 구현 본문은 아직 읽지 않았다.
7. **가중치 크기**(float32, 복셀 15,724 기준 계산): 이미지 CLS 약 48MB, 텍스트 6.2GB, VDVAE 5.7GB, retrieval 16.5GB, GiT 16.5GB. 실측과 일치한다(`docs/mirage_server_run_log.md`).
8. **비용 실측**(A2000 12GB): 학습 약 50분(retrieval 없음), 생성은 이미지당 약 1.3~2분, retrieval 후보 16장이면 시드당 약 9시간 40분.

## 개선 방향이 붙는 위치

| 방향 | 붙는 곳 | 코드에서 건드릴 부분 | 바로 확인할 것 |
|---|---|---|---|
| 1. ROI별 decoder 분리 | 릿지 1~3(필요하면 4)의 입력 복셀 | 릿지 학습 셀의 `x_train` 열을 ROI별로 나눠 학습, 예측을 합침. 마스크는 `get_kastner_masks` 계열 재사용 가능 | ROI를 나눠도 normalize_preds 이후 예측이 어떻게 달라지는지, 합치는 방식(평균, 타깃별 ROI 선택) |
| 2. 맥락 decoder 추가 + 복셀 압축 | c_t 자리(77×1280 텍스트 조건) 또는 별도 가이던스 | `reconstruct()`의 conditions 구성(`sc_reconstructor.py:144-161`) | 맥락 임베딩을 bigG 텍스트 공간으로 옮기는 방법. 기존 단계 실험에서 맥락이 캡션을 넘은 적이 없음 |
| 3. 입출력 차원 축소 | 릿지 출력(98,560 / 91,168 / 263,168)과 입력(15,724) | 릿지 학습 셀. 출력은 k개 축으로 예측 후 복원 | 성능 유지(비열등)와 학습 시간·메모리·가중치 크기. 생성 시간은 거의 안 줄어듦(Stage C/B가 대부분) |

## 평가를 위한 단계 구분(제안)

생성은 이미지당 1~2분이고 (4명 × 18자극 × 5시드)면 조건당 약 12시간이므로, 값싼 단계에서 먼저 거른다.
1. **지각 shared1000**(1,000장): 예측 임베딩의 식별 정확도·코사인 유사도, 생성 없음. 검정력이 높다.
2. **imagery 12자극**: 같은 임베딩 지표를 imagery 베타에서 계산.
3. **생성**: 위에서 개선 신호가 있는 것만 공식 평가 방식으로 평가(피험자 짝지은 비교).

## 아직 확인하지 않은 것

- NSD-Imagery 자연 장면 5개가 shared1000이어서 학습에서 제외되는지(`nsd_stim_info_merged.csv`와 `nsdimagery_stimuli.pkl3`의 `image_map`으로 확인 가능, 서버에 있음).
- `snr`, `top_n_rois` 인자의 구현 본문과 `get_kastner_masks`가 만드는 ROI의 정의.
- 텍스트 조건 c_t가 정확히 어떤 캡션(`mid_length_captions_73K`)에서 오는지의 생성 방식(논문 부록 확인 필요).
