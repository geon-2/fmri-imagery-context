# 라벨 차원 후보 검토 — 표현 점검 (채택 여부 판단이 아님)

표현 목록과 매칭 방식은 결과를 보기 전에 고정했다(스크립트 상단 주석). 이 표는 개방/폐쇄, 깊이·규모, 이동 가능성 표현이 **이미 기존 차원에 얼마나 나오는지**만 보여 준다.

### A. 파일럿 300장 (adjudicate v3, 5차원, 값별 support 집계) (이미지 300장)

**주 기준: 단어 단위**

| 표현 계열 | 차원 | 일치 이미지 | 비율 | 일치 라벨 수 | 고유 값 | 평균 support | 평균 confidence | 상위 값(이미지 수) |
|---|---|---|---|---|---|---|---|---|
| 개방/폐쇄 | place_type | 16 | 5.3% | 22 | 21 | 0.42 | 0.61 | open water(2), open-plan kitchen and dining area(1), residential interior / open-plan living area(1), open field / countryside(1) |
| 개방/폐쇄 | environment | 109 | 36.3% | 180 | 91 | 0.45 | 0.74 | open(36), open space(23), enclosed(7), open / expansive(6) |
| 개방/폐쇄 | activity | 0 | 0.0% | 0 | 0 | - | - | - |
| 개방/폐쇄 | implied_event | 9 | 3.0% | 16 | 16 | 0.37 | 0.42 | shop closed or off-hours, quiet morning(1), shop may be closed or quiet (off-hours or winter lull)(1), shop not yet open or quiet off-hours(1), about to open the bags and cut(1) |
| 개방/폐쇄 | scale | 48 | 16.0% | 76 | 58 | 0.38 | 0.62 | open landscape(6), open outdoor area(4), open field(3), small enclosed space(3) |
| 깊이/규모 | place_type | 3 | 1.0% | 4 | 4 | 0.42 | 0.64 | bay with distant hills(1), shallow coastal water(1), shallow sea water near beach(1), shallow ocean water(1) |
| 깊이/규모 | environment | 17 | 5.7% | 25 | 12 | 0.47 | 0.75 | shallow depth of field(9), shallow depth of field / blurred background(4), hazy distant hills(2), blurred background, shallow depth of field(2) |
| 깊이/규모 | activity | 1 | 0.3% | 1 | 1 | 1.00 | 0.95 | deep frying(1) |
| 깊이/규모 | implied_event | 0 | 0.0% | 0 | 0 | - | - | - |
| 깊이/규모 | scale | 126 | 42.0% | 207 | 111 | 0.50 | 0.77 | close-up(61), medium close-up(11), close-up / tabletop(5), close-up portrait(4) |
| 이동 가능성 | place_type | 0 | 0.0% | 0 | 0 | - | - | - |
| 이동 가능성 | environment | 0 | 0.0% | 0 | 0 | - | - | - |
| 이동 가능성 | activity | 0 | 0.0% | 0 | 0 | - | - | - |
| 이동 가능성 | implied_event | 0 | 0.0% | 0 | 0 | - | - | - |
| 이동 가능성 | scale | 0 | 0.0% | 0 | 0 | - | - | - |

**민감도: 부분 문자열**

| 표현 계열 | 차원 | 일치 이미지 | 비율 | 일치 라벨 수 | 고유 값 | 평균 support | 평균 confidence | 상위 값(이미지 수) |
|---|---|---|---|---|---|---|---|---|
| 개방/폐쇄 | place_type | 16 | 5.3% | 22 | 21 | 0.42 | 0.61 | open water(2), open-plan kitchen and dining area(1), residential interior / open-plan living area(1), open field / countryside(1) |
| 개방/폐쇄 | environment | 109 | 36.3% | 180 | 91 | 0.45 | 0.74 | open(36), open space(23), enclosed(7), open / expansive(6) |
| 개방/폐쇄 | activity | 0 | 0.0% | 0 | 0 | - | - | - |
| 개방/폐쇄 | implied_event | 16 | 5.3% | 26 | 26 | 0.37 | 0.41 | shop closed or off-hours, quiet morning(1), shop may be closed or quiet (off-hours or winter lull)(1), shop not yet open or quiet off-hours(1), hydrant opened to cool off during a heat wave(1) |
| 개방/폐쇄 | scale | 48 | 16.0% | 76 | 58 | 0.38 | 0.62 | open landscape(6), open outdoor area(4), open field(3), small enclosed space(3) |
| 깊이/규모 | place_type | 3 | 1.0% | 4 | 4 | 0.42 | 0.64 | bay with distant hills(1), shallow coastal water(1), shallow sea water near beach(1), shallow ocean water(1) |
| 깊이/규모 | environment | 17 | 5.7% | 25 | 12 | 0.47 | 0.75 | shallow depth of field(9), shallow depth of field / blurred background(4), hazy distant hills(2), blurred background, shallow depth of field(2) |
| 깊이/규모 | activity | 1 | 0.3% | 1 | 1 | 1.00 | 0.95 | deep frying(1) |
| 깊이/규모 | implied_event | 0 | 0.0% | 0 | 0 | - | - | - |
| 깊이/규모 | scale | 126 | 42.0% | 207 | 111 | 0.50 | 0.77 | close-up(61), medium close-up(11), close-up / tabletop(5), close-up portrait(4) |
| 이동 가능성 | place_type | 0 | 0.0% | 0 | 0 | - | - | - |
| 이동 가능성 | environment | 0 | 0.0% | 0 | 0 | - | - | - |
| 이동 가능성 | activity | 0 | 0.0% | 0 | 0 | - | - | - |
| 이동 가능성 | implied_event | 0 | 0.0% | 0 | 0 | - | - | - |
| 이동 가능성 | scale | 0 | 0.0% | 0 | 0 | - | - | - |

### B. 1만 장 (v9/v10, N=1이라 support 없음, 6차원) (이미지 10,000장)

**주 기준: 단어 단위**

| 표현 계열 | 차원 | 일치 이미지 | 비율 | 일치 라벨 수 | 고유 값 | 평균 support | 평균 confidence | 상위 값(이미지 수) |
|---|---|---|---|---|---|---|---|---|
| 개방/폐쇄 | place_type | 628 | 6.3% | 628 | 428 | - | 0.87 | open sky airspace(19), open sky(15), open grassy field(10), open water recreation area(10) |
| 개방/폐쇄 | environment | 2018 | 20.2% | 2027 | 1925 | - | 0.87 | compact enclosed room(6), wide open sky(5), open grassy field(5), open grassy countryside(5) |
| 개방/폐쇄 | activity | 361 | 3.6% | 370 | 370 | - | 0.87 | storefront standing closed or inactive at the moment(1), man holding pastry up to his open mouth(1), a dog is sitting on a closed toilet while a cat sits on nearby storage(1), a small child is sitting in the grass and holding an open umbrella(1) |
| 개방/폐쇄 | implied_event | 279 | 2.8% | 280 | 280 | - | 0.61 | vehicles passed through while the shutter stayed open(1), the cat appears to have chosen the basin as a cool enclosed resting spot(1), attendees may soon take food from the open boxes(1), the open case suggests the performance is for tips(1) |
| 개방/폐쇄 | scale | 140 | 1.4% | 140 | 136 | - | 0.88 | medium telephoto action view on open water(3), medium telephoto action shot on open water(2), telephoto view of a single aircraft in open sky(2), medium wildlife view in open grass(1) |
| 개방/폐쇄 | notes | 7 | 0.1% | 7 | 7 | - | 0.83 | the exaggerated open-mouth pose appears humorous or staged(1), open jaws and splashing water make the interaction look tense(1), motion blur and open-mouthed expression make the scene feel candid(1), the closed eyes and soft costume give the portrait a staged holiday feel(1) |
| 깊이/규모 | place_type | 208 | 2.1% | 208 | 174 | - | 0.82 | food close-up setting(9), dining table close-up(7), food preparation close-up(5), meal plate close-up(4) |
| 깊이/규모 | environment | 1170 | 11.7% | 1241 | 1089 | - | 0.87 | shallow depth of field(42), very shallow depth of field(16), indoor tabletop close-up(12), indoor close-up food scene(8) |
| 깊이/규모 | activity | 121 | 1.2% | 121 | 121 | - | 0.87 | turning or edging across deep snow(1), carving a snowboard through shallow water and snow(1), dog swimming and splashing across shallow water(1), dog is running through shallow beach water while carrying a toy(1) |
| 깊이/규모 | implied_event | 48 | 0.5% | 48 | 48 | - | 0.62 | the pose is being held for a humorous close-up photograph(1), the decorative frosting was applied before the close-up photograph(1), the group is gathered for close-up animal viewing(1), the meal was plated shortly before the close-up photograph(1) |
| 깊이/규모 | scale | 1490 | 14.9% | 1506 | 1267 | - | 0.92 | tight food close-up(21), close-up tabletop view(16), tight tabletop close-up(14), food close-up(8) |
| 깊이/규모 | notes | 44 | 0.4% | 44 | 44 | - | 0.82 | shallow focus isolates the bottles from the boats behind them(1), tilt-shift or selective blur makes the distant scene look miniature(1), shallow-focus professional portrait photograph(1), background is intentionally blurred by shallow focus(1) |
| 이동 가능성 | place_type | 0 | 0.0% | 0 | 0 | - | - | - |
| 이동 가능성 | environment | 1 | 0.0% | 1 | 1 | - | 0.88 | walkable city-center environment(1) |
| 이동 가능성 | activity | 0 | 0.0% | 0 | 0 | - | - | - |
| 이동 가능성 | implied_event | 0 | 0.0% | 0 | 0 | - | - | - |
| 이동 가능성 | scale | 0 | 0.0% | 0 | 0 | - | - | - |
| 이동 가능성 | notes | 0 | 0.0% | 0 | 0 | - | - | - |

**민감도: 부분 문자열**

| 표현 계열 | 차원 | 일치 이미지 | 비율 | 일치 라벨 수 | 고유 값 | 평균 support | 평균 confidence | 상위 값(이미지 수) |
|---|---|---|---|---|---|---|---|---|
| 개방/폐쇄 | place_type | 634 | 6.3% | 634 | 434 | - | 0.87 | open sky airspace(19), open sky(15), open grassy field(10), open water recreation area(10) |
| 개방/폐쇄 | environment | 2045 | 20.4% | 2058 | 1956 | - | 0.87 | compact enclosed room(6), wide open sky(5), open grassy field(5), open grassy countryside(5) |
| 개방/폐쇄 | activity | 398 | 4.0% | 408 | 408 | - | 0.87 | storefront standing closed or inactive at the moment(1), man holding pastry up to his open mouth(1), a dog is sitting on a closed toilet while a cat sits on nearby storage(1), a small child is sitting in the grass and holding an open umbrella(1) |
| 개방/폐쇄 | implied_event | 410 | 4.1% | 413 | 413 | - | 0.62 | the hydrant has likely been opened or damaged shortly before the photo(1), someone likely left the vehicle stationary while the dog climbed into the elevated opening(1), vehicles passed through while the shutter stayed open(1), the cat appears to have chosen the basin as a cool enclosed resting spot(1) |
| 개방/폐쇄 | scale | 157 | 1.6% | 157 | 153 | - | 0.88 | medium telephoto action view on open water(3), medium telephoto action shot on open water(2), telephoto view of a single aircraft in open sky(2), close street detail focused on the bin opening(1) |
| 개방/폐쇄 | notes | 8 | 0.1% | 8 | 8 | - | 0.81 | the exaggerated open-mouth pose appears humorous or staged(1), open jaws and splashing water make the interaction look tense(1), motion blur and open-mouthed expression make the scene feel candid(1), the closed eyes and soft costume give the portrait a staged holiday feel(1) |
| 깊이/규모 | place_type | 212 | 2.1% | 212 | 178 | - | 0.82 | food close-up setting(9), dining table close-up(7), food preparation close-up(5), meal plate close-up(4) |
| 깊이/규모 | environment | 1175 | 11.8% | 1246 | 1094 | - | 0.87 | shallow depth of field(42), very shallow depth of field(16), indoor tabletop close-up(12), indoor close-up food scene(8) |
| 깊이/규모 | activity | 123 | 1.2% | 123 | 123 | - | 0.87 | turning or edging across deep snow(1), carving a snowboard through shallow water and snow(1), dog swimming and splashing across shallow water(1), dog is running through shallow beach water while carrying a toy(1) |
| 깊이/규모 | implied_event | 71 | 0.7% | 71 | 71 | - | 0.62 | the pose is being held for a humorous close-up photograph(1), the animal has been brought into the shallows for cleaning(1), the decorative frosting was applied before the close-up photograph(1), the boat was left moored and the tide or conditions have exposed it in the shallows(1) |
| 깊이/규모 | scale | 1526 | 15.3% | 1542 | 1301 | - | 0.92 | tight food close-up(21), close-up tabletop view(16), tight tabletop close-up(14), food close-up(8) |
| 깊이/규모 | notes | 45 | 0.5% | 45 | 45 | - | 0.82 | shallow focus isolates the bottles from the boats behind them(1), tilt-shift or selective blur makes the distant scene look miniature(1), shallow-focus professional portrait photograph(1), background is intentionally blurred by shallow focus(1) |
| 이동 가능성 | place_type | 0 | 0.0% | 0 | 0 | - | - | - |
| 이동 가능성 | environment | 1 | 0.0% | 1 | 1 | - | 0.88 | walkable city-center environment(1) |
| 이동 가능성 | activity | 0 | 0.0% | 0 | 0 | - | - | - |
| 이동 가능성 | implied_event | 0 | 0.0% | 0 | 0 | - | - | - |
| 이동 가능성 | scale | 0 | 0.0% | 0 | 0 | - | - | - |
| 이동 가능성 | notes | 0 | 0.0% | 0 | 0 | - | - | - |

### C. 사람 라벨 200장 (영어/한국어 열을 함께 봄, 열 이름이 곧 차원) (이미지 200장)

**주 기준: 단어 단위**

| 표현 계열 | 차원 | 일치 이미지 | 비율 | 일치 라벨 수 | 고유 값 | 평균 support | 평균 confidence | 상위 값(이미지 수) |
|---|---|---|---|---|---|---|---|---|
| 개방/폐쇄 | activity | 5 | 2.5% | 5 | 5 | - | - | open fire hydrant on the right spraying water(1), a cat with mouth open as if yawning or about to meow, haunches on the ground, front legs extended(1), cat sitting somewhat ambiguously on top of a car with eyes closed (dozing)(1), girl-looking child sitting on the grass with an open umbrella smiling(1) |
| 개방/폐쇄 | dropped | 1 | 0.5% | 1 | 1 | - | - | high-ceilinged restroom with sink and toilet visible through open door(1) |
| 개방/폐쇄 | environment | 1 | 0.5% | 1 | 1 | - | - | bright (seems open-air)(1) |
| 개방/폐쇄 | framing | 0 | 0.0% | 0 | 0 | - | - | - |
| 개방/폐쇄 | notes | 0 | 0.0% | 0 | 0 | - | - | - |
| 개방/폐쇄 | place_type | 2 | 1.0% | 3 | 2 | - | - | not an open field but forest-like with sparse trees(2), wide neat open lot(1) |
| 개방/폐쇄 | space_scale | 0 | 0.0% | 0 | 0 | - | - | - |
| 깊이/규모 | activity | 3 | 1.5% | 3 | 3 | - | - | distant vehicle lights visible far away(1), people canoeing standing on surfboards or wading in shallow water(1), 넓은 목장으로 보이는 곳이고 정면에 소 세마리가 있음(1) |
| 깊이/규모 | dropped | 0 | 0.0% | 0 | 0 | - | - | - |
| 깊이/규모 | environment | 0 | 0.0% | 0 | 0 | - | - | - |
| 깊이/규모 | framing | 15 | 7.5% | 22 | 14 | - | - | close-up(3), close-up from directly in front, slightly above(2), closer than a wide shot but wider than a close-up(2), close-up with the sandwich in front(2) |
| 깊이/규모 | notes | 0 | 0.0% | 0 | 0 | - | - | - |
| 깊이/규모 | place_type | 1 | 0.5% | 1 | 1 | - | - | 넓은 마당이 있는 마구간?(1) |
| 깊이/규모 | space_scale | 0 | 0.0% | 0 | 0 | - | - | - |
| 이동 가능성 | activity | 0 | 0.0% | 0 | 0 | - | - | - |
| 이동 가능성 | dropped | 0 | 0.0% | 0 | 0 | - | - | - |
| 이동 가능성 | environment | 0 | 0.0% | 0 | 0 | - | - | - |
| 이동 가능성 | framing | 0 | 0.0% | 0 | 0 | - | - | - |
| 이동 가능성 | notes | 0 | 0.0% | 0 | 0 | - | - | - |
| 이동 가능성 | place_type | 0 | 0.0% | 0 | 0 | - | - | - |
| 이동 가능성 | space_scale | 0 | 0.0% | 0 | 0 | - | - | - |

**민감도: 부분 문자열**

| 표현 계열 | 차원 | 일치 이미지 | 비율 | 일치 라벨 수 | 고유 값 | 평균 support | 평균 confidence | 상위 값(이미지 수) |
|---|---|---|---|---|---|---|---|---|
| 개방/폐쇄 | activity | 7 | 3.5% | 7 | 7 | - | - | open fire hydrant on the right spraying water(1), a cat with mouth open as if yawning or about to meow, haunches on the ground, front legs extended(1), a woman opening a pink parasol/umbrella walking right to left in front of a passing vehicle(1), cat sitting somewhat ambiguously on top of a car with eyes closed (dozing)(1) |
| 개방/폐쇄 | dropped | 1 | 0.5% | 1 | 1 | - | - | high-ceilinged restroom with sink and toilet visible through open door(1) |
| 개방/폐쇄 | environment | 1 | 0.5% | 1 | 1 | - | - | bright (seems open-air)(1) |
| 개방/폐쇄 | framing | 0 | 0.0% | 0 | 0 | - | - | - |
| 개방/폐쇄 | notes | 0 | 0.0% | 0 | 0 | - | - | - |
| 개방/폐쇄 | place_type | 2 | 1.0% | 3 | 2 | - | - | not an open field but forest-like with sparse trees(2), wide neat open lot(1) |
| 개방/폐쇄 | space_scale | 0 | 0.0% | 0 | 0 | - | - | - |
| 깊이/규모 | activity | 3 | 1.5% | 3 | 3 | - | - | distant vehicle lights visible far away(1), people canoeing standing on surfboards or wading in shallow water(1), 넓은 목장으로 보이는 곳이고 정면에 소 세마리가 있음(1) |
| 깊이/규모 | dropped | 0 | 0.0% | 0 | 0 | - | - | - |
| 깊이/규모 | environment | 0 | 0.0% | 0 | 0 | - | - | - |
| 깊이/규모 | framing | 15 | 7.5% | 22 | 14 | - | - | close-up(3), close-up from directly in front, slightly above(2), closer than a wide shot but wider than a close-up(2), close-up with the sandwich in front(2) |
| 깊이/규모 | notes | 0 | 0.0% | 0 | 0 | - | - | - |
| 깊이/규모 | place_type | 1 | 0.5% | 1 | 1 | - | - | 넓은 마당이 있는 마구간?(1) |
| 깊이/규모 | space_scale | 0 | 0.0% | 0 | 0 | - | - | - |
| 이동 가능성 | activity | 0 | 0.0% | 0 | 0 | - | - | - |
| 이동 가능성 | dropped | 0 | 0.0% | 0 | 0 | - | - | - |
| 이동 가능성 | environment | 0 | 0.0% | 0 | 0 | - | - | - |
| 이동 가능성 | framing | 0 | 0.0% | 0 | 0 | - | - | - |
| 이동 가능성 | notes | 0 | 0.0% | 0 | 0 | - | - | - |
| 이동 가능성 | place_type | 0 | 0.0% | 0 | 0 | - | - | - |
| 이동 가능성 | space_scale | 0 | 0.0% | 0 | 0 | - | - | - |
