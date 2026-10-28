# 문헌 리서치 교차 검증 (Claude Research · Perplexity · Codex, 2026-10-15)

대상: `docs/literature_research_prompts_1015.md` 프롬프트 A(ROI별 정보). 프롬프트 B(부분공간 정렬·RSA·복셀 압축)는 아직 결과 없음. 숫자는 모두 **원문 재확인 전**이다. 탐색 단계.

## 1. 도구별 신뢰도
| 도구 | 읽은 범위 | 평가 |
|---|---|---|
| Claude Research | Saha Roy 2025 PMC 전문을 읽고 방법·수치를 인용. 나머지는 초록·2차 | 가장 사용 가능. 단, 아래 CCN 2024 항목은 틀림 |
| Perplexity | 대부분 초록 수준, 「not verified」를 정직하게 표기. 보강 논문 다수 | 유용하나 본문 결론과 자기 출처 목록이 충돌한 곳이 있음(CCN 2023) |
| Codex | 「정독 여부」 표기는 있으나 실제 근거 불명 | **낮음.** 존재가 의심되는 인용이 여러 개 있어 단독 근거로 쓰지 않음 |

Codex 인용 중 의심 항목(다른 두 도구에 없고 서지가 기억과 어긋남, 직접 확인 전까지 쓰지 않음): Hajhajate 2020 eLife(pRF), Hong 2018 Nat Neurosci(LO "SOFI"), Dehaene-Lambertz 2010(EBA/FBA), Joseph 2006(VWFA 상상), Xing & Li 2023 eNeuro, Wagstyl 2025 eLife, Mullally 2012 Cortex, Pennock 2017 Curr Biol(음식 영역은 Pennock 2023으로 알려짐), Baldassano 2016 Nat Neurosci. 「NSD-Imagery 초기 시각 영역 r≈0.20」 같은 수치도 출처 불명.

## 2. 도구 간 충돌과 해소
| 항목 | Claude Research | Perplexity | 직접 확인 결과 |
|---|---|---|---|
| CCN 2024 「Do vision and imagery share common principal signal components?」 | 찾지 못함 | 포스터 B155로 존재 | **존재함.** [포스터 페이지](https://2024.ccneuro.org/poster/?id=498), [논문 PDF](https://2024.ccneuro.org/pdf/415_Paper_authored_tiasha_ccn2024_withauthors.pdf) (이 세션의 웹 검색으로 확인). 방법: voxel-to-voxel autoencoding으로 vision/imagery 주성분 추정. 결과: imagery 첫 성분이 vision 첫 성분과 비슷하게 정렬, imagery 성분 위의 vision 분산은 훨씬 낮음(압축 가설과 부합), 예비 결과 |
| CCN 2023 「Mental Imagery: Weak Vision or Compressed Vision?」 | 존재(DOI 10.32470/CCN.2023.1693-0) | 본문은 「확인되지 않음」이라 했으나 출처 목록에 같은 제목의 2023 proceedings PDF가 있음 | **존재로 판단.** Claude가 DOI까지 제시, Perplexity 출처도 일치. 내가 직접 페이지를 열려 했으나 서버 오류로 실패 |
| Saha Roy 정렬 수치 | V1 25~30%, V4 약 50%, parietal 약 100%(전문 읽음) | ROI별 수치 확인 못함 | Claude 쪽 채택(전문 근거). 단 두 번째 데이터셋(Breedlove 2020, n=3)에서는 V1/V2 50~60% |
| Wang et al. 연도 | NMI 2023 | bioRxiv 2022 | **NMI 2023**(preprint 2022). 우리 문서의 「2022」 수정 필요 |
| Lee, Kravitz & Baker | 2012 | 2012 | **NeuroImage 2012**(온라인 2011). 우리 문서의 「2011」 수정 필요 |
| MIRAGE | PLoS CB 2026-05-22, DOI 10.1371/journal.pcbi.1014263 | 같은 DOI | 일치. arXiv 2605.17198, PMC13218618로 본문 접근 가능 → **ROI 처리는 우리가 직접 읽어 확인해야 함**(두 도구 모두 확인 못함) |

## 3. 우리 설계에 영향을 주는 사실 (원문 확인 필요 표시)
1. **복셀 선택 기준 불일치.** Claude는 Saha Roy가 「NSD-core SNR 상위 **2%**」 복셀을 썼다고 쓰고, 우리 기존 문헌 정리(`docs/roi_literature_review_draft.md`)는 「상위 **20%**」(80백분위 이상)라고 적었다. 둘 중 하나가 틀렸다. 우리 전처리(반복 신뢰도 ≥ 0.1)와 비교하려면 방법 절을 직접 확인해야 한다.
2. **자극 수.** Saha Roy는 자극 12개(도형·막대 6 + 자연 장면 6), NSD-Imagery는 18개(단순 6, 복잡 6, 개념 6). 맥락 축을 imagery 데이터만으로 추정할 수 없다는 우리 우려와 일치.
3. **단순 대 복잡 구분이 모든 ROI에서 보존**되고 범주 내 구분은 parietal 외에는 약하다(Saha Roy). 우리의 자연 장면 6개 식별이 이 거친 구분만으로 풀릴 수 있다. 맥락 분석은 장면 자극 안에서만 해야 한다는 Claude의 권고에 동의.
4. **MIRAGE 구성 정보**(Claude): ridge, 텍스트+이미지 multimodal 타깃, Stable Cascade, 저수준 「layout」 이미지를 먼저 재구성. 즉 사용자가 기억한 「CLS 768 + 캡션」과 다르다. ROI 분해는 확인 안 됨.
5. **차원 해석 주의.** Gauthaman 등의 scale-free 표현 연구(arXiv 2409.06843 → PLoS CB)는 지각 표현의 차원이 사실상 무한하다고 주장. Saha Roy는 「차원 수 추정이 아니라 상대 비교」라고 직접 경고. 압축 주장을 쓸 때 함께 인용해야 한다.
6. **place memory areas(LPMA/VPMA/MPMA).** Steel 2025 확률 아틀라스가 공개되어 NSD 분석에 ROI로 추가할 수 있다(Claude 제안, 미확인).
7. **VWFA·EBA/FBA·음식 영역의 imagery 연구는 두 도구 모두 못 찾음.** 「없다」가 아니라 「이번 검색에서 못 찾음」으로 쓴다.

## 4. 새로 확인된 논문 (한 도구에만 나온 것 중 존재 가능성이 높은 것, DOI 확인 전)
- Favila, Kuhl & Winawer 2022 Nat Commun 13:5864 (기억 인출 시 pRF tuning이 위계와 무관하게 넓음, n=9)
- Woodry, Curtis & Winawer 2025 J Neurosci (작업기억·장기기억에서도 tuning 확대)
- Chang et al. 2025 Curr Biol 35:591 (aphantasia에서 V1 decoding은 되지만 지각과 교차 안 됨, 해석 논쟁 있음)
- Wadia et al. 2026 Science (VTC 단일뉴런, 지각-심상 공유 코드, 환자 16명)
- Bainbridge, Hall & Baker 2021 Cereb Cortex (회상은 부호화의 단순 재활성화가 아님)
- Senden et al. 2019 Brain Struct Funct (글자 imagery, pRF 기반 복원, n=6, 7T)
- Harrison & Tong 2009 Nature; Albers et al. 2013 Curr Biol; Weber et al. 2024 (작업기억-imagery 공유)
- Megla et al. 2024/2025 (일란성 쌍둥이 aphantasia, n=2)
- arXiv 2604.15374 「Seeing the imagined」(HCP-MMP1 parcel별 imagery decoding, preprint), NeuroAdapter arXiv 2509.23566
- Shirakawa et al. 2025 (NSD 재구성의 spurious reconstruction 비판), Koide-Majima 2024 재분석 논쟁

## 5. 아직 해야 할 것
1. 프롬프트 B 결과(부분공간 정렬, RSA, 복셀 압축, MIRAGE의 ROI 처리)를 같은 방식으로 교차 검증.
2. 직접 원문 확인 대상: Saha Roy 2025 방법 절(복셀 선택 2% 대 20%, 정렬비 정의, 대조군), MIRAGE 방법 절(ROI·복셀 처리), Lee 2012(수치·장비).
3. `docs/roi_literature_review_draft.md`의 인용 정정: Saha Roy CCN 2023(존재함)과 CCN 2024(존재함) 둘 다 확인됨(「확인되지 않았다」는 문구 삭제), Wang 2023, Lee 2012.

---

# 프롬프트 B 교차 검증 (같은 날 추가)

## 6. 도구별 신뢰도 (B)
- **Claude Research**: Saha Roy 2025와 MIRAGE 본문을 직접 읽고 방법을 재구성. 불명확한 점을 10개로 정리. 가장 사용 가능.
- **Perplexity**: 방법 지도와 분석 설계(식, null, 모델 계층)는 유용. Saha Roy의 세부(분모 정의, 복셀 선택, null)는 「not verified」로 정직하게 표기. MIRAGE 복셀 마스크 문구는 인용.
- **Codex**: 다시 신뢰도 낮음. Saha Roy의 모델을 「2층 선형 오토인코더」, 데이터셋 1을 「COCO 12장 + NSD 1000장 이상」, VVIQ가 「V1 정렬」과 상관이라고 적었으나 Claude(전문 읽음)와 어긋남(실제: reduced-rank ridge, 자극 12개, VVIQ는 V1 **심상 차원 수**와 r=0.71, 정렬비와는 상관 없음). Naselaris 2015를 「Neuron」, Silson 2019를 「Nat Neurosci」, Breedlove 2020을 「Nat Commun」으로 적은 서지 오류도 있음. 단독 근거로 쓰지 않는다.

## 7. 핵심 해소 사항
| 항목 | 결과 | 확인 방식 |
|---|---|---|
| **MIRAGE의 ROI 처리** | **ROI 분할 없음. 전체 `nsdgeneral` 마스크(1.8 mm) 하나.** Methods 4.2.1 원문: 「We applied the provided nsdgeneral voxel mask at a 1.8 mm resolution…」. decoder는 ROI별이 아니라 **특징 세트(VDVAE, CLIP 이미지 CLS 768, CLIP 텍스트 등)별로 병렬 ridge**를 따로 학습(4.2.3). ROI(Early/Higher Vis.)는 Table 1의 평가 지표(뇌 상관)에서만 쓰임 | 세 도구 중 둘이 같은 문구 인용 + 이 세션에서 PMC 본문(PMC13218618) 직접 확인 |
| MIRAGE ridge λ 선택 | 재구성 대상이 아닌 NSD 피험자 4명으로 격자 탐색, λ = 100,000 | 본문 직접 확인. 단 그림 설명에 「imagery 성능」이 y축이라고 되어 있어 imagery 결과가 λ 선택에 쓰였는지 불명확(원문이 다 읽히지 않음: 앞 100,000자만 확인) |
| 사용자 질문 3 (CLS 768 + 캡션 decoder 병렬) | 사실과 가까움: 이미지 CLS 768 decoder와 텍스트(CLIP ViT-bigG/14, LLaVA 합성 긴 캡션 평균 34단어) decoder를 따로 두고 Stable Cascade에 결합. 추가로 저수준 VDVAE 복원 이미지가 구조를 제공. 텍스트+이미지 결합이 특히 imagery에서 큰 향상, 이미지만이 텍스트만보다 우수 | Claude·Perplexity 일치 |
| **Saha Roy alignment ratio 정의** | **우리 정의와 방향이 반대.** 원문: a_g = (시각 test 패턴을 imagery 부분공간 d_im차원에 사영한 총분산) / (같은 패턴을 visual 부분공간 첫 d_im차원에 사영한 총분산). 즉 「**시각 분산 중 imagery 부분공간이 포착하는 비율**」. 우리가 쓴 「imagery 분산을 perception 부분공간이 설명하는 비율」과 다름 | Claude(전문) + Codex 일치. Perplexity는 미확인이라 자체 식을 「권장」으로 제시 |
| Saha Roy 복셀 선택 | 데이터셋 1(NSD-Imagery 12자극, n=8): ROI 내 NSD-core SNR **상위 2%**(98백분위). 데이터셋 2(Breedlove, n=3): **상위 20%**(80백분위). **두 수치가 둘 다 맞고 데이터셋이 다름** → 우리 기존 정리의 「20%」는 데이터셋 2 기준 | Claude + Codex 일치 |
| Saha Roy 모델·null | reduced-rank ridge(vis2img), 차원은 validation 최대 성능(Fig.4는 peak의 99%), null은 시각 공간의 **무작위 singular vector d_im개 100회**. **노이즈 천장 보정 없음.** 첫 imagery run 제외 | Claude(전문) |
| NSD-Imagery 평가 자극과 누출 | complex 자극 6개 중 5개가 shared1000. 벤치마크 decoder는 shared1000 제외 학습. 우리 CLAUDE.md의 「shared1000 및 NSD-Imagery 자연장면 5장 제외」와 일치 | Claude. 우리 규칙과 이미 부합 |
| NSD-Imagery 벤치마크 ROI 결과 | 심상에서 EVC brain correlation 하락이 상위 시각피질보다 가파름(예: MindEye1 EVC 0.180 vs 상위 0.135, vision EVC 0.374) | Claude(2차). 직접 원문 확인 전 |
| 뇌 정렬 축 vs 같은 차원 PCA 축의 imagery 일반화 | **두 도구 모두 직접 연구를 못 찾음.** 가장 가까운 간접 근거는 Semedo 2019(「예측 차원 ≠ 분산 최대 차원」, 원숭이 V1-V2)와 MIRAGE의 저차원 특징 이점 | 「없다」가 아니라 「이번 검색에서 못 찾음」으로 쓴다 |
| ROI별 PCA 대 전역 PCA 직접 비교, 복셀 선택 임계값이 정렬비에 미치는 영향 | 못 찾음 | 위와 같음 |

## 8. 설계에 반영할 것 (제안, 기준 문서에서 확정)
1. alignment ratio는 **원문 방향(시각 분산을 imagery 부분공간이 포착)** 과 **우리가 원래 생각한 방향(imagery 분산을 perception 부분공간이 설명)** 을 둘 다 계산해 보고한다. 비대칭 자체가 정보다.
2. **노이즈 천장 정규화와 시뮬레이션**(「완전 재사용 + 관측 노이즈」의 기대값)을 필수로 둔다. Saha Roy의 EVC 낮은 정렬은 회전과 노이즈를 구분하지 못한 상태다.
3. 모델 계층 비교: 평균만 → 스칼라(약한 vision) → 대각 → 직교 회전 → reduced-rank → full ridge. 모두 held-out 자극에서 평가. 「약한 vision 대 변형된 vision」 구분을 가능하게 한다.
4. null은 한 종류가 아니라 여러 개: 같은 차원 무작위 부분공간, 스펙트럼 보존 회전, PCA 부분공간(같은 차원), 라벨 셔플로 학습한 RRR 부분공간.
5. RSA는 crossnobis(run 단위 교차검증). 6개 조건이면 거리 15개뿐이라 점근 p값을 피하고 정확 순열·기술 통계로 제한. 자극 종류(단순·복잡·개념)는 따로 분석.
6. 복셀 수·신뢰도를 ROI 간에 맞춘 반복 부분표집, 첫 imagery run 포함/제외, GLM 버전, 복셀 임계값을 민감도 분석으로.
7. **imagery 데이터로 하이퍼파라미터·차원·ROI를 고르지 않는다**(MIRAGE의 λ 선택 모호성을 반면교사로).
8. 맥락 축 평가는 복잡 자극 6개(5개는 shared1000)와 개념 6개에 한정. 단순 6개는 음성 대조.

## 9. 아직 해야 할 것 (갱신)
- 직접 원문 확인: Saha Roy 2025 Supplementary와 수식 이미지(alignment ratio, 차원 선택 규칙), NSD-Imagery 논문 Appendix(복셀 마스크, GLM 버전, ROI 결과 표), MIRAGE Table 1 + S1 appendix(λ 선택에 imagery가 쓰였는지).
- 기록 정정: `docs/roi_literature_review_draft.md`의 Saha Roy 「20%」 서술에 「데이터셋 2 기준, 데이터셋 1은 2%」를 추가하고, alignment ratio 정의 방향을 바로잡는다.
- Notion 「새 논의 정리」 페이지의 3번 항목(MIRAGE ROI 처리 「확인 필요」)을 「ROI 분할 없음(nsdgeneral)」으로 갱신. Notion 쓰기는 요청이 있을 때만.

---

# 원문 직접 확인 결과 (같은 날, 이 세션에서 WebFetch)

## 10. Saha Roy 2025 (bioRxiv v1 본문, 수식은 이미지라 텍스트 정의만)
- 정렬비(Methods 「Evaluating geometric alignment」): 「시각 활동의 총분산(모든 자극) 중 [imagery d_im 차원이] 설명하는 양」 ÷ 「시각 부분공간 첫 d_im 차원이 설명하는 시각 분산」. 방향 확인됨(시각 분산 기준).
- d_im 선택: 순위 r을 1부터 min(n, T)까지 훑어 validation 평균 Pearson r 최대. Results의 d_vis는 「각 곡선 peak의 99% 값의 평균」, Fig.2: 4-fold 교차검증 선 탐색. 두 규칙이 본문에 공존.
- null: Fig.5B 「시각 활동 공간의 무작위 d_im개 singular vector로 100회」. 예측 정확도 null은 입력-출력 무작위 짝짓기(Fig.3E). 색 막대 「p < .01」 외에 ROI별 p값 없음.
- 복셀 선택: NSD-Imagery는 ROI 내 SNR 98백분위 초과, Spatial Imagery는 vision 시행에서 SNR 80백분위 초과. SNR = 신호 분산 ÷ 조건당 8회 반복 베타의 분산. **ROI별 복셀 수는 본문에 없음.**
- ROI: ventral/lateral/parietal은 「해부학적 위치」, Spatial Imagery의 IPS는 공개 확률지도.
- **노이즈 천장·신뢰도 보정 없음.** 코드·데이터 공개 문구 없음(NSD-Imagery 데이터 매뉴얼 참조만).
- 확인 못 함: 수식 이미지, Supplementary(뒤 72,471자).

## 11. NSD-Imagery 논문 (arXiv 2506.06898 HTML)
- 참가자 8명 모두 추가 세션을 했으나, 40세션 전체를 마친 피험자는 본문이 「1, 2, 5, 7」(Sec.4.1은 「11, 22, 55, 77」로 표기해 오타로 보임, Appendix A.5.1이 1·2·5·7로 명시). **우리 피험자 1·2·5·7과 일치.**
- 자극: 세트마다 6개, 총 18개. **complex = NSD shared1000의 자연 장면 5개 + 미술작품 1개(Kehinde Wiley의 「The Two Sisters」).**
- 반복: vision 8회, imagery 16회(imagery run 2번). 총 12 run, 피험자당 576시행. 본문 「66」「88」은 오타로 추정.
- GLM: b3(`betas_fithrf_GLMdenoise_RR`), 1.8 mm, **run별로 따로 z-score**. GLMsingle가 작은 데이터셋에서 덜 효과적이라고 저자가 언급.
- 복셀 마스크: 대부분 `nsdgeneral`. Takagi만 streams 아틀라스의 early·higher(ventral).
- Table 1 뇌 상관(EVC / 상위 / 전체): imagery MindEye1 0.180/0.135/0.155, Brain Diffuser 0.133/0.127/0.141, iCNN 0.113/0.062/0.113, MindEye2 0.069/0.055/0.061. vision MindEye1 0.374/0.253/0.311, iCNN 0.442/0.246/0.338. 저자: 「EVC의 하락이 상위보다 훨씬 가파르다」.
- 첫 imagery run의 특별한 처리 언급은 **없음**(Saha Roy는 첫 run 제외, 벤치마크는 모두 포함 → 전처리 불일치 확정).
- 확인 못 함: Appendix 전부(PDF가 403).
