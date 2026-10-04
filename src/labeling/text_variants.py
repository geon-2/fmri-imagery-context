"""한 이미지의 세 텍스트(원본 캡션, 사물, 맥락 라벨)를 임베딩 입력 단위로 만드는 규칙. 노트북과 시험 코드가 같이 쓴다.

주 방식(단위별 임베딩 후 평균)의 단위:
  원본  캡션 5개 각각          사물  사물 이름 각각          맥락  라벨 구(phrase) 각각
변형 V2(한 번에 임베딩)의 문자열:
  원본  캡션 5개를 공백으로 이음(문장 끝 마침표가 없으면 붙임)
  사물  사물 이름을 알파벳순으로 ', '로 이은 문자열 하나(Doerig 방식, 이전에 만든 것과 같음)
  맥락  라벨을 항목 순서(place_type, environment, activity, implied_event, scale, notes)로 '; '로 이은 문자열 하나
변형 V3(맥락만): 항목 안에서만 평균, 항목끼리는 분리(6블록, 비어 있는 항목은 마스크로 표시)
"""
DIMS = ("place_type", "environment", "activity", "implied_event", "scale", "notes")


def original_units(captions):
    return [c.strip() for c in captions if c and c.strip()]


def original_joined(captions):
    return " ".join(c if c.endswith((".", "!", "?")) else c + "." for c in original_units(captions))


def object_names(object_only):
    return [n for n in (object_only or "").split(", ") if n]


def object_joined(object_only):
    return ", ".join(sorted(object_names(object_only)))


def context_by_dim(labels):
    """labels: {dim: [[value, confidence], ...]} → {dim: [phrase, ...]}"""
    return {d: [v[0].strip() for v in labels.get(d, []) if v and v[0].strip()] for d in DIMS}


def context_units(labels):
    by = context_by_dim(labels)
    return [p for d in DIMS for p in by[d]]


def context_joined(labels):
    units = context_units(labels)
    return "; ".join(units) + ("." if units else "")
