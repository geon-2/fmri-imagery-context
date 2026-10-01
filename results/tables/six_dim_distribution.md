# 6항목 값 분포와 이미지당 라벨 수 (pass 0, 이미지 11,150장, casefold 기준)

값 하나가 얼마나 많은 이미지에 나오는지(이미지 비율)와, 이미지당 라벨 수가 어떻게 분포하는지를 본다. 진단용이며 판정 기준은 두지 않았다.
코어 1,000장은 프롬프트 v9, 그 밖은 v10이다.

### 코어 (v9) (이미지 1,000장)

| 항목 | 라벨 수 | 고유 값 | 가장 흔한 값의 이미지 비율 |
|---|---|---|---|
| place_type | 1587 | 1394 | 1.2% (bedroom) |
| environment | 2705 | 2669 | 0.3% (soft natural daylight) |
| activity | 1196 | 1193 | 0.2% (tennis player serving during a match) |
| implied_event | 1033 | 1031 | 0.2% (the batter intends to swing if the incoming pitch is hittable) |
| scale | 1034 | 1004 | 0.5% (wide interior room view) |
| notes | 257 | 257 | 0.1% (foreground blur suggests the photo was taken through or near a barrier) |

### 그 밖 (v10) (이미지 10,150장)

| 항목 | 라벨 수 | 고유 값 | 가장 흔한 값의 이미지 비율 |
|---|---|---|---|
| place_type | 20544 | 14300 | 0.9% (bedroom) |
| environment | 34328 | 29314 | 2.7% (outdoors) |
| activity | 20437 | 19923 | 0.1% (architectural sightseeing or documentation) |
| implied_event | 11009 | 10983 | 0.0% (the disc has just been thrown toward the jumping player) |
| scale | 14900 | 13077 | 0.3% (wide interior room view) |
| notes | 2366 | 2331 | 0.1% (visible photographer watermark in the lower corner) |

### 코어 (v9): 이미지당 라벨 수 분포 (이미지 1,000장)

| 항목 | 0개 | 1개 | 2개 | 3개 | 4개 | 5개 이상 | 평균 | 최소 개수 충족 |
|---|---|---|---|---|---|---|---|---|
| place_type | 0.1% | 41.5% | 58.0% | 0.4% | 0.0% | 0.0% | 1.59 | 58.4% |
| environment | 0.0% | 0.2% | 40.8% | 47.9% | 10.5% | 0.6% | 2.71 | 59.0% |
| activity | 0.5% | 79.6% | 19.7% | 0.2% | 0.0% | 0.0% | 1.20 | 19.9% |
| implied_event | 9.2% | 78.3% | 12.5% | 0.0% | 0.0% | 0.0% | 1.03 | - |
| scale | 0.0% | 96.6% | 3.4% | 0.0% | 0.0% | 0.0% | 1.03 | - |
| notes | 74.3% | 25.7% | 0.0% | 0.0% | 0.0% | 0.0% | 0.26 | - |

### 그 밖 (v10): 이미지당 라벨 수 분포 (이미지 10,150장)

| 항목 | 0개 | 1개 | 2개 | 3개 | 4개 | 5개 이상 | 평균 | 최소 개수 충족 |
|---|---|---|---|---|---|---|---|---|
| place_type | 0.0% | 0.0% | 97.6% | 2.4% | 0.0% | 0.0% | 2.02 | 100.0% |
| environment | 0.0% | 0.0% | 0.0% | 61.8% | 38.2% | 0.0% | 3.38 | 100.0% |
| activity | 0.0% | 0.0% | 98.7% | 1.3% | 0.0% | 0.0% | 2.01 | 100.0% |
| implied_event | 7.1% | 77.5% | 15.3% | 0.1% | 0.0% | 0.0% | 1.08 | - |
| scale | 0.0% | 53.2% | 46.8% | 0.0% | 0.0% | 0.0% | 1.47 | - |
| notes | 76.7% | 23.3% | 0.0% | 0.0% | 0.0% | 0.0% | 0.23 | - |
