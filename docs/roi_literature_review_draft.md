# ROI별 정보 내용과 imagery 압축 관련 문헌 정리 (초안, 2026-10-09)

목적: 「뇌 쪽(복셀)도 압축해서 보자」는 방향을 정하기 전에, ROI들이 실제로 어떤 정보를 담는지와 imagery에서 perception과 어떻게 다른지를 선행 연구에서 확인한다. 논문이 중심 결과로 다루지 않았지만 본문·방법·보충에 있는 정보도 함께 모았다.

## 0. 읽은 범위와 신뢰도 (중요)
| 자료 | 읽은 범위 | 신뢰도 |
|---|---|---|
| Saha Roy 등 2025 bioRxiv, "A transformation from vision to imagery in the human brain" | PDF 전문을 텍스트로 추출해 초록, 결과, 토의, 방법의 일부를 직접 읽음 | 높음(수치는 원문 인용). 보충 그림(S1~S3)은 못 봄 |
| Saha Roy 등 CCN 2024 포스터 | 초록 페이지 | 초록 수준 |
| Lee, Kravitz, Baker 2012 NeuroImage(온라인 2011) | PMC 본문을 도구가 요약한 것 | 중간(수치는 원문 재확인 필요) |
| Scrivener, Teed, Silson 2025 | PMC 본문 앞 100,000자(전체 138,618자) 요약 | 중간, 뒷부분 미확인 |
| Wang 등 2023 Nature Machine Intelligence(preprint 2022, CLIP, 고차 시각 피질) | 도구 요약 | 중간 |
| Doerig 등 2025 Nature Machine Intelligence | PMC 앞 100,000자(전체 136,693자) 요약 | 중간, ROI별 수치는 그림에만 있다고 함 |
| Dijkstra 등 2019 TiCS 리뷰 | PDF 텍스트 앞부분 | 중간 |
| Spagna 등 2021 메타분석, Shen 등 CCN 2024, Elmoznino & Bonner 2024, 7T aphantasia 연구(2025) | 검색 결과 초록 수준 | 낮음~중간 |

도구가 요약한 수치는 원문과 다를 수 있다. 슬라이드나 논문에 쓰기 전에 원문 확인이 필요하다.

**정정(2026-10-15, `docs/literature_crosscheck_1015.md`):** 위 초안에서 「CCN 2023 「Mental Imagery: Weak Vision or Compressed Vision?」은 검색으로 확인되지 않았다」고 쓴 것은 **틀렸다.** 세 가지가 모두 존재한다: (1) CCN 2023 "Mental Imagery: Weak Vision or Compressed Vision?"(DOI 10.32470/CCN.2023.1693-0, 포스터 P-2B.94), (2) CCN 2024 포스터 B155 "Do vision and imagery share common principal signal components?"(웹 검색으로 포스터 페이지·PDF 확인), (3) 2025 bioRxiv 프리프린트(위 제목, 같은 저자들). 슬라이드·대본·Notion의 CCN 2023 인용은 유지해도 된다. 또 Wang et al.은 Nature Machine Intelligence **2023**(preprint 2022), Lee, Kravitz & Baker는 NeuroImage **2012**(온라인 2011)다.

## 1. 핵심 요약
1. **imagery는 약해진 perception이 아니라 변형된 perception이다(Saha Roy 2025).** 초기 시각 영역(V1 등)에서 imagery 부분공간의 차원 수는 vision의 약 절반이고 방향도 틀어진다. 상위 영역에서는 차원 수가 비슷하고 거의 같은 부분공간이다.
2. **perception과 imagery의 겹침은 상위 영역일수록 크다.** 초기 영역의 겹침은 필요한 시각 세부와 개인의 vividness에 따라 달라진다(Dijkstra 2019, Lee 2011).
3. **imagery 단독으로는 V1에서 디코딩되지 않아도, perception으로 학습한 분류기를 imagery에 적용하면(cross-decoding) V1에서도 유의했다(Lee 2011).**
4. **메타분석은 초기 시각 피질의 imagery 활동 증거를 못 찾았고 좌측 방추이랑(fusiform) 한 부분을 찾았다(Spagna 2021).** 다만 활성화 평균 분석 기반이라 패턴 분석과 결과가 다를 수 있다.
5. **고차 시각 ROI는 의미(언어) 정보를 담는다.** 캡션 임베딩과 CLIP이 FFA, EBA, PPA, OPA, RSC를 예측하고(Wang 2022, Doerig 2025), 사물 명사만 쓴 것은 전체 캡션보다 못하다.
6. **압축과 고차원은 긴장 관계다.** DNN 모델에서는 고차원 표현이 피질을 더 잘 예측했고(Elmoznino & Bonner 2024), 피질 영역과 DNN이 공유하는 표현의 차원 수는 초기에서 후기로 갈수록 줄며 초기는 공간 차원, 후기는 채널 차원에 의존했다(Shen 등 CCN 2024). 이 두 연구는 우리가 쓰는 의미 부분공간 압축과 직접 비교할 수 없는 모델 쪽 결과다.

## 2. ROI별 정리
| ROI | perception에서 담는 정보 | imagery에서의 상태 | 출처 |
|---|---|---|---|
| V1~V4(초기) | 위치·방향·공간주파수 등 저수준. CLIP 이미지 모델의 마지막 층보다 ResNet이 V1에서 더 잘 설명 | imagery 부분공간 차원이 vision의 약 절반이고 방향이 틀어짐. 정렬 비율 V1 25~30%, V4 약 50%. SNR이 낮음. imagery 단독 디코딩은 V1에서 비유의(Lee), cross-decoding은 유의. 개인 vividness와 V1 imagery 차원 수가 상관(r=0.71, 8명) | Saha Roy 2025, Lee 2011, Wang 2022 |
| LO, pFs(물체 선택) | 물체 정체성 | imagery 디코딩이 perception의 약 40%(초기는 약 10%). imagery에서 위계가 반대로(물체 영역이 상대적으로 강함) | Lee 2011 |
| ventral/lateral(상위) | 범주·의미. 캡션 임베딩으로 잘 예측 | 차원 수가 vision과 비슷(패리티). 거의 identity 변환 | Saha Roy 2025, Doerig 2025 |
| FFA, EBA(생물) | 사람·동물(PC1: 생물 대 실내 장면) | 친숙한 사람·장소 상상(12명)에서 VTC 영역들은 각자 선호 범주 안의 자극을 디코딩했으나 FFA1은 예외(FFA2는 fusion 반응이 드문드문) | Wang 2022, Scrivener 2025 |
| PPA, OPA, RSC(장면) | 장소·레이아웃. PC2에서 장소와 음식 영역 구분 | 장소 상상 중 장소 선택 영역이 먼저 반응(약 500 ms, EEG-fMRI fusion). V1은 훨씬 나중(약 1371 ms 시작) | Wang 2022, Scrivener 2025 |
| parietal(IPS 등) | 공간·특징 기반 주의 | imagery에서 vision과 같은 부분공간(정렬 비율 약 100%), 내용 구분(범주 안)은 vision과 비슷 | Saha Roy 2025, Dijkstra 2019 |
| frontal | 과제 관련 자극 구조 | perception과 imagery에서 비슷한 표현 | Dijkstra 2019 |
| 좌측 fusiform(imagery node) | — | 메타분석에서 imagery 일관 활성(46개 fMRI 연구 중 27개가 시각 imagery 대상) | Spagna 2021 |
| TPOJ, 각회 | CLIP이 더 잘 설명(심성화·언어 연결은 저자들의 추측) | — | Wang 2022 |

## 3. 논문이 중심으로 다루지 않았지만 우리에게 중요한 정보
1. **첫 imagery run의 SNR이 낮았다(Saha Roy, 방법).** imgA-1, imgB-1을 SNR이 낮아 분석에서 뺐고 저자들은 훈련 효과(비정상성)로 해석했다. 우리 imagery 분석에서 run 순서에 따른 SNR을 확인해야 한다. NSD-Imagery의 run 구성도 다시 봐야 한다.
2. **복셀을 SNR 상위 2%(데이터셋 1, NSD-Imagery 12자극, n=8) 또는 상위 20%(데이터셋 2, Breedlove 데이터, n=3)로 골랐다(방법; 2026-10-15 교차 검증으로 정정: 이전에는 20%로만 적었음).** 독립 vision 데이터(NSD-core 또는 Set A)에서 SNR이 ROI 내 98 또는 80백분위 이상인 복셀만 사용. 선택 방식이 차원 추정에 영향을 줄 수 있다. 우리는 반복 신뢰도 ≥0.1 기준을 쓰므로 비교할 때 선택 기준을 맞춰야 한다. **정렬비 정의(교차 검증, Claude 전문 읽음)**: a_g = 시각 test 패턴을 imagery 부분공간 d_im차원에 사영한 총분산 ÷ 같은 패턴을 visual 부분공간 첫 d_im차원에 사영한 총분산. 즉 「시각 분산 중 imagery 부분공간이 포착하는 비율」이며, 「imagery 분산을 perception 부분공간이 설명하는 비율」이 아니다. 노이즈 천장 보정은 없다.
3. **자극이 12개뿐이라 차원 수의 상한이 12다.** 저자들은 절대 차원 수가 아니라 상대 차원 수만 해석하라고 명시한다. 우리 imagery 세트(target 6개)는 더 작다.
4. **imagery 차원이 인코딩하는 것은 주로 거친 범주 구분이다.** 단순 도형 대 자연 장면 같은 큰 구분은 robust했고, 같은 범주 안의 세밀한 구분은 거의 없었다(parietal 제외). 우리 RQ2a는 자연 장면 5개를 식별하는데, 거친 범주 차이만으로도 맞출 수 있는지 확인해야 한다.
5. **vividness와의 관계는 V1 차원 수에서만 유의했다.** 다른 ROI와 정렬 정도에서는 없었다(8명, r=0.71, 95% 신뢰구간 0.15~0.98로 매우 넓음).
6. **Lee 2011:** perception 반응이 imagery보다 약 10배 컸고, imagery 반응은 ES에서만 0과 달랐다. 물체 선택 영역 디코딩의 일부는 이미지 간 망막위치 차이 때문일 수 있다고 저자들이 인정. 표본 11명, 자극 10개, 두정엽·측두엽 일부 미촬영.
7. **Scrivener 2025:** V1에서 장소는 디코딩되고 사람은 안 됨(장소 쪽 시각 변동이 커서일 수 있다는 가설, 모델 근거는 약함). 단일 자극 디코딩 정확도 54~60%(보통 수준). 내측 두정엽 ROI는 개인 localizer가 아니라 기존 데이터에서 가져옴. 해마의 장소 선택성은 보충자료에만 있음. 표본 12명.
8. **Spagna 2021:** 후두엽에 국한된 손상 환자에서도 imagery가 보존된 사례를 근거로 초기 시각 피질이 imagery에 필수가 아니라고 주장.
9. **Doerig 2025:** 명사·동사만 쓴 캡션은 전체 캡션보다 정렬이 낮고(초기 시각 피질에서 명사는 예외), 형용사·부사·전치사의 정렬은 매우 낮았다. 단어 순서를 섞어도 MPNet 임베딩 상관이 평균 0.91이라 순서에 둔감하다. 참가자가 연속 재인 과제를 해서 속으로 캡션을 만들었을 가능성이 캡션 임베딩에 유리할 수 있다고 저자들이 언급.
10. **Wang 2022:** CLIP 마지막 층은 텍스트와 정렬된 병목이라 V1에서는 ResNet보다 못하다. 중간층이 전체 경로를 가장 잘 예측. 단어 임베딩 BERT는 CLIP 대비 고유 설명량이 거의 없음. 본문은 ROI별 통계를 거의 보고하지 않고 그림 중심.
11. **Dijkstra 2019:** 초기 시각 영역의 겹침은 과제가 요구하는 시각 세부와 vividness에 의존. 큰 틀의 주장은 imagery에는 상향(bottom-up) 처리가 없다는 것.

## 4. 우리 가설과 설계에 주는 함의 (제안, 확정 아님)
- **가설을 ROI별로 나눠 써야 한다.**
  - 고차 ROI(ventral, lateral, parietal): imagery 패턴이 perception 부분공간 안에 있을 가능성이 높다(Saha Roy: 거의 identity).
  - 초기 ROI: imagery가 다른 부분공간에 놓이므로 perception 부분공간 투영이 잘 안 될 가능성이 높다.
  - 우리가 지금까지 본 「고차 영역이 초기보다 높다」와 같은 방향이다.
- **압축 방법 후보(복셀 쪽):** perception 복셀 PCA, PLS, CCA, 복셀-복셀 RRR(Saha Roy의 vis2vis와 같은 구조), 오토인코더, 무작위 부분공간 기준선. 평가는 정렬 비율(imagery 부분공간이 perception 부분공간 분산을 설명하는 정도)을 무작위 기준선과 비교하는 방식이 선행 연구에 있다.
- **글 쪽 압축과 복셀 쪽 압축을 둘 다 하고, 같은 ROI 단위로 비교한다.**
- **복셀 선택 기준, run별 SNR, 거친 범주 대 세부 구분을 분석 전에 정해 두어야 한다.**
- **Doerig의 결과는 맥락 라벨에 직접 관련이 있다.** 맥락 라벨이 형용사·전치사 비중이 높은 구라면 임베딩 정렬이 낮을 수 있고, 이건 「캡션이 맥락 라벨보다 높음」의 가능한 원인 후보로 보류해 둔다.
- **Elmoznino와 Shen의 고차원 결과는 DNN 특징의 차원이라서 우리 의미 축 압축과 직접 비교할 수 없다.** 다만 「소수 축이 전체와 거의 같다」를 말할 때 반박 문헌으로 인용될 수 있어 대비해 둔다.

## 5. 확인되지 않은 것, 다음 조사
- Saha Roy 2025 보충 그림과 두 번째 데이터셋 세부, eLife 심사 의견.
- Lee 2011, Scrivener 2025 뒷부분, Doerig 2025 ROI별 수치(보충과 저장소 `visuo_llm`)의 원문 확인.
- Horikawa & Kamitani 2017(계층적 특징 decoding과 imagery), Breedlove 2020(imagery의 pRF), Naselaris 2015(기억된 장면의 인코딩 모델 디코딩), NSD-Imagery 데이터 논문(Kneeland 2025)의 ROI별·vividness 결과.
- 7T aphantasia 연구(Cortex 2025)의 ROI별 결과 원문.
- NSD ROI 정의(스트림 ROI, 범주 선택 localizer)와 우리 ROI(EARLY, S_ventral, 고차 400개)의 대응.
