"""
FC 온라인 전력분석실
- 내 분석: 나 vs 같은 경기 상대 평균 비교 → 강점/약점, 질 때 공통점
- 상대 스카우팅: 3초 스카우팅 카드(최근 10경기) + 내 기록과 겹쳐 본 매칭 전략
데이터: 넥슨 Open API (https://openapi.nexon.com)
실행: streamlit run app.py
"""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

import deep
from common import FEATURE_INFO, FEATURES, match_features, side_row
from common import (DEFAULT_COEF, HEADER_TYPE, calibrate, db_connect, detect_flip,
                    latest_model, save_match, xg_values)

BASE = "https://open.api.nexon.com"
QUICK_N = 10  # 스카우팅 카드용 경기 수
st.set_page_config(page_title="FC 전력분석실", page_icon="⚽", layout="wide")


# ================================================================ API
class ApiError(Exception):
    def __init__(self, msg, status=None):
        super().__init__(msg)
        self.status = status


def _get(path, params, key):
    for attempt in range(5):
        r = requests.get(BASE + path, params=params,
                         headers={"x-nxopen-api-key": key}, timeout=15)
        if r.status_code == 429:  # 요청 한도 → 잠깐 쉬고 재시도 (계속이면 하루 한도 소진)
            if attempt >= 1:
                break
            time.sleep(0.7)
            continue
        if r.status_code != 200:
            try:
                msg = r.json().get("error", {}).get("message", r.text)
            except Exception:
                msg = r.text
            raise ApiError(f"{r.status_code}: {msg}", r.status_code)
        return r.json()
    raise ApiError("넥슨 API 요청 한도를 넘었어요.", 429)


@st.cache_data(ttl=86400, show_spinner=False)
def meta(name):
    r = requests.get(f"{BASE}/static/fconline/meta/{name}.json", timeout=30)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=3600, show_spinner=False)
def get_ouid(nick, key):
    return _get("/fconline/v1/id", {"nickname": nick}, key)["ouid"]


@st.cache_data(ttl=600, show_spinner=False)
def get_maxdiv(ouid, key):
    return _get("/fconline/v1/user/maxdivision", {"ouid": ouid}, key)


@st.cache_data(ttl=120, show_spinner=False)
def get_matches(ouid, mtype, limit, key):
    return _get("/fconline/v1/user/match",
                {"ouid": ouid, "matchtype": mtype, "offset": 0, "limit": limit}, key)


@st.cache_resource
def detail_store():
    return {}  # 경기 상세 캐시 (경기 기록은 바뀌지 않으니 계속 재사용)


def fetch_details(ids, key, label):
    """여러 경기 상세를 동시에 불러와서 로딩 시간을 줄임."""
    store = detail_store()
    todo = [m for m in ids if m not in store]
    if todo:  # DB에 저장된 경기는 넥슨 호출 없이 꺼내 씀
        store.update(details_from_db(tuple(todo)))
        todo = [m for m in todo if m not in store]
    if todo and not st.session_state.get("api_exhausted"):
        bar = st.progress(0.0, text=label)

        def work(mid):
            try:
                return mid, _get("/fconline/v1/match-detail", {"matchid": mid}, key)
            except ApiError as e:
                return mid, ("LIMIT" if e.status == 429 else None)

        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = [ex.submit(work, m) for m in todo]
            for i, f in enumerate(as_completed(futs), 1):
                mid, d = f.result()
                if d == "LIMIT":
                    st.session_state.api_exhausted = True
                elif d:
                    store[mid] = d
                bar.progress(i / len(todo), text=f"{label} ({i}/{len(todo)})")
        bar.empty()
    return [store[m] for m in ids if m in store]


# ================================================================ 지표 정의
# (키, 이름, 높을수록 좋은가)
PERF = [
    ("win", "승률(%)", True),
    ("gf", "경기당 득점", True),
    ("ga", "경기당 실점", False),
    ("shots", "슈팅 수", True),
    ("sot_rate", "유효슈팅 비율(%)", True),
    ("conv", "슈팅 대비 득점(%)", True),
    ("shots_against", "허용 슈팅", False),
    ("sot_against", "허용 유효슈팅", False),
    ("pass_rate", "패스 성공률(%)", True),
    ("through_rate", "스루패스 성공률(%)", True),
    ("tackle_rate", "태클 성공률(%)", True),
    ("block_rate", "블록 성공률(%)", True),
    ("rating", "평균 평점", True),
    ("fouls", "파울", False),
]
STYLE = [
    ("poss", "점유율(%)"),
    ("box_share", "박스 안 슈팅 비중(%)"),
    ("head_share", "헤딩 슈팅 비중(%)"),
    ("long_share", "롱패스 비중(%)"),
    ("through_try", "스루패스 시도"),
    ("offside", "오프사이드"),
]
LABEL = {k: l for k, l, _ in PERF} | dict(STYLE)

TIP_SELF_WEAK = {
    "win": "승률이 같은 구간 상대들보다 낮아요. 아래 세부 약점부터 하나씩 잡아 보세요.",
    "gf": "득점이 부족해요. 찬스 창출과 마무리 중 어디서 막히는지 슈팅 지표를 같이 보세요.",
    "ga": "실점이 많아요. 수비 라인 간격, 무리한 태클 타이밍을 점검해 보세요.",
    "shots": "슈팅까지 가는 전개가 적어요. 빌드업이 마무리로 이어지지 않고 있어요.",
    "sot_rate": "슈팅이 골문을 자주 벗어나요. 각도 없는 슈팅, 급한 슈팅을 줄여 보세요.",
    "conv": "찬스 대비 마무리가 약해요. 무리한 중거리보다 박스 안에서 한 번 더 연결해 보세요.",
    "shots_against": "슈팅을 많이 허용해요. 박스 앞 공간 커버가 늦는지 확인해 보세요.",
    "sot_against": "유효슈팅을 많이 허용해요. 슈팅 각도를 좁히는 수비 위치를 신경 써 보세요.",
    "pass_rate": "패스 미스가 잦아요. 압박을 받을 때 짧고 안전한 선택을 늘려 보세요.",
    "through_rate": "스루패스 성공률이 낮아요. 뒷공간이 확실히 열렸을 때만 시도해 보세요.",
    "tackle_rate": "태클 실패가 많아요. 태클보다 컨테인과 위치 선정으로 막는 비중을 늘려 보세요.",
    "block_rate": "블록 성공률이 낮아요. 슈팅 코스 앞에 몸을 두는 타이밍이 늦을 수 있어요.",
    "rating": "선수 평균 평점이 낮아요. 아래 선수별 표에서 평점이 낮은 선수를 확인해 보세요.",
    "fouls": "파울이 많아요. 박스 근처 프리킥과 카드 리스크가 커집니다.",
}
TIP_SCOUT_WEAK = {  # 상대 약점 → 공략 포인트
    "ga": "실점이 많은 상대. 초반부터 적극적으로 두드려",
    "shots_against": "슈팅을 쉽게 내줌. 박스 앞에서 과감하게 슈팅",
    "sot_against": "유효슈팅을 쉽게 내줌. 기회 오면 망설이지 말고 슈팅",
    "pass_rate": "패스 미스가 잦음. 전방 압박이 잘 통함",
    "tackle_rate": "태클 실패가 많음. 드리블 돌파와 개인기가 유효",
    "block_rate": "블록이 약함. 중거리 슈팅도 가치 있음",
    "fouls": "파울이 많음. 박스 근처 드리블로 프리킥 유도",
    "conv": "마무리가 약함. 박스 안 결정적 찬스만 막으면 됨",
    "sot_rate": "슈팅 정확도가 낮음. 먼 거리 슈팅은 내줘도 부담 적음",
    "through_rate": "스루패스가 잘 안 통함. 라인을 조금 올려도 됨",
    "shots": "공격 전개가 둔함. 라인 올려 압박해도 역습 위험 낮음",
    "win": "최근 폼이 안 좋음. 초반 기세로 밀어붙이기 좋음",
    "gf": "득점력이 낮음. 선제골 넣으면 유리하게 흘러감",
}
TIP_SCOUT_STRONG = {  # 상대 강점 → 주의할 점
    "conv": "결정력이 좋음. 박스 안 슈팅 공간을 절대 내주지 마",
    "sot_rate": "슈팅이 정확함. 슈팅 각도를 끝까지 좁혀",
    "gf": "득점력이 높음. 선제 실점만은 피하는 운영",
    "through_rate": "스루패스가 잘 통함. 수비 라인 너무 올리지 마",
    "tackle_rate": "태클이 좋음. 무리한 드리블보다 원투 패스",
    "pass_rate": "빌드업이 안정적. 무리한 압박보다 지역 수비",
    "ga": "수비가 단단함. 급하게 두드리지 말고 기다려",
    "sot_against": "유효슈팅을 잘 안 내줌. 확실한 찬스까지 연결",
    "block_rate": "블록을 잘함. 중거리보다 침투 위주로",
    "shots": "슈팅을 많이 만듦. 박스 앞 수비 숫자 늘리기",
    "win": "최근 폼이 좋음. 초반 실점 조심",
}

# 3) 매칭 전략: (내 강점 키, 상대 약점 키, 전략)
MATCHUP = [
    ("through_rate", "ga", "내 스루패스가 잘 통하고 상대는 실점이 많아. 뒷공간 침투 위주로 공격해"),
    ("through_rate", "shots_against", "내 스루패스 + 상대는 슈팅을 쉽게 내줘. 라인 뒤로 찔러서 슈팅까지 가져가"),
    ("conv", "sot_against", "내 결정력이 좋고 상대는 유효슈팅을 쉽게 내줘. 찬스 오면 바로 마무리해"),
    ("sot_rate", "block_rate", "내 슈팅이 정확하고 상대 블록이 약해. 박스 앞 중거리도 적극적으로"),
    ("tackle_rate", "pass_rate", "내 태클이 좋고 상대는 패스 미스가 잦아. 전방 압박으로 공 뺏어"),
    ("pass_rate", "tackle_rate", "내 패스가 안정적이고 상대 태클이 약해. 짧게 풀다가 드리블로 파고들어"),
    ("sot_against", "conv", "내가 슈팅 각을 잘 막고 상대 결정력이 약해. 수비 안정 후 역습"),
    ("ga", "shots", "내 수비가 단단하고 상대 공격 전개가 둔해. 라인 올려 점유하며 압박"),
    ("shots", "shots_against", "내가 슈팅을 많이 만들고 상대는 많이 허용해. 슈팅 수로 밀어붙여"),
    ("gf", "ga", "내 득점력이 좋고 상대 실점이 많아. 공격적으로 운영해도 돼"),
]
# (상대 강점 키, 내 약점 키, 경고)
RISK = [
    ("conv", "shots_against", "상대 결정력이 좋은데 나는 슈팅 허용이 많아. 박스 앞 공간 차단이 최우선"),
    ("conv", "sot_against", "상대 결정력이 좋은데 나는 유효슈팅 허용이 많아. 슈팅 각 좁히기에 집중"),
    ("through_rate", "ga", "상대 스루패스가 잘 통하고 내 실점이 많은 편. 라인을 한 칸 내려"),
    ("tackle_rate", "pass_rate", "상대 태클이 좋고 내 패스 미스가 잦아. 무리한 전진 패스 대신 안전하게"),
    ("pass_rate", "tackle_rate", "상대 빌드업이 좋고 내 태클 성공률이 낮아. 태클보다 컨테인으로 버텨"),
    ("gf", "ga", "상대 득점력이 높고 내 실점도 많은 편. 선제 실점만은 피해"),
    ("shots", "shots_against", "상대가 슈팅을 많이 만들고 나는 많이 허용해. 박스 앞 수비 숫자 늘려"),
    ("ga", "conv", "상대 수비가 단단하고 내 마무리가 약해. 슈팅 전에 한 번 더 연결해서 확실한 찬스로"),
]

CONTROLLER = {"keyboard": "키보드", "gamepad": "패드", "pad": "패드"}
LOSS_EXCLUDE = {"win", "gf", "ga", "conv", "rating"}  # 지면 당연히 나빠지는 지표 제외


def sdiv(a, b):
    return a / b * 100 if b else np.nan


def goals(side):
    sh = side.get("shoot") or {}
    return sh.get("goalTotalDisplay", sh.get("goalTotal", 0)) or 0


def side_metrics(s, o):
    md = s.get("matchDetail") or {}
    sh = s.get("shoot") or {}
    ps = s.get("pass") or {}
    df = s.get("defence") or {}
    osh = o.get("shoot") or {}
    shots = sh.get("shootTotal", 0) or 0
    ptry = ps.get("passTry", 0) or 0
    return {
        "win": 100.0 if md.get("matchResult") == "승" else 0.0,
        "gf": goals(s),
        "ga": goals(o),
        "shots": shots,
        "sot_rate": sdiv(sh.get("effectiveShootTotal", 0), shots),
        "conv": sdiv(sh.get("goalTotal", 0), shots),
        "shots_against": osh.get("shootTotal", 0),
        "sot_against": osh.get("effectiveShootTotal", 0),
        "pass_rate": sdiv(ps.get("passSuccess", 0), ptry),
        "through_rate": sdiv(ps.get("throughPassSuccess", 0), ps.get("throughPassTry", 0)),
        "tackle_rate": sdiv(df.get("tackleSuccess", 0), df.get("tackleTry", 0)),
        "block_rate": sdiv(df.get("blockSuccess", 0), df.get("blockTry", 0)),
        "rating": md.get("averageRating", np.nan),
        "fouls": md.get("foul", 0),
        "poss": md.get("possession", np.nan),
        "box_share": sdiv(sh.get("shootInPenalty", 0), shots),
        "head_share": sdiv(sh.get("shootHeading", 0), shots),
        "long_share": sdiv(ps.get("longPassTry", 0), ptry),
        "through_try": ps.get("throughPassTry", 0),
        "offside": md.get("OffsideCount", 0),
    }


PLAYER_KEYS = ["goal", "assist", "shoot", "effectiveShoot", "spRating", "passTry",
               "passSuccess", "dribbleTry", "dribbleSuccess", "tackleTry", "tackle"]


# ================================================================ 분석
def compare(me, bm):
    """경기별 (대상 - 그 경기 상대) 차이로 효과크기 계산. 양수 = 좋은 쪽."""
    out = []
    for k, label, higher_better in PERF:
        a, b = me[k].astype(float), bm[k].astype(float)
        diff = (a - b).dropna()
        if len(diff) < 3:
            continue
        sd = diff.std(ddof=1)
        eff = diff.mean() / sd if sd and sd > 0 else 0.0
        if not higher_better:
            eff = -eff
        out.append({"key": k, "지표": label, "대상": a.mean(), "비교 평균": b.mean(),
                    "차이": a.mean() - b.mean(), "효과": eff})
    return pd.DataFrame(out)


def goal_minute(gt):
    period, sec = gt >> 24, (gt & 0xFFFFFF) / 60
    base = {0: 0, 1: 45, 2: 90, 3: 105}.get(period, 120)
    return sec if sec >= base else base + sec


BINS = [-1, 15, 30, 45, 60, 75, 90, 200]
BIN_LABELS = ["0-15분", "16-30분", "31-45분", "46-60분", "61-75분", "76-90분", "90분+"]


def timing(shots):
    mins = [goal_minute(s.get("goalTime", 0)) for s in shots
            if s.get("result") == 3 and (s.get("goalTime", 0) >> 24) < 4]
    cut = pd.cut(pd.Series(mins, dtype=float), BINS, labels=BIN_LABELS)
    return cut.value_counts().reindex(BIN_LABELS, fill_value=0).astype(float)


def style_tags(me, bm):
    tags = []
    poss = me["poss"].mean()
    tags.append("점유형" if poss >= 55 else "역습형" if poss <= 45 else "밸런스형")
    box = me["box_share"].mean()
    if box >= 70:
        tags.append("박스 안 침투 선호")
    elif box <= 50:
        tags.append("중거리 슈팅 많음")
    if me["head_share"].mean() >= 20:
        tags.append("크로스·헤딩 의존")
    if me["long_share"].mean() >= 15:
        tags.append("롱패스 많음")
    if bm["through_try"].mean() and me["through_try"].mean() >= 1.3 * bm["through_try"].mean():
        tags.append("스루패스 적극 활용")
    if me["offside"].mean() >= 2:
        tags.append("뒷공간 침투 성향")
    return tags


def analyze(nick, mtype, n, key, label="경기 기록 불러오는 중"):
    """닉네임 → 분석 결과 dict. 실패하면 에러 문자열."""
    offline = False
    try:
        ouid = get_ouid(nick.strip(), key)
    except ApiError as e:
        ouid = db_find_ouid(nick.strip()) if e.status == 429 else None
        if ouid is None:
            if e.status == 429:
                return (f"오늘 넥슨 API 한도를 다 써서 새로 조회할 수 없고, '{nick}'의 경기도 DB에 없어요. "
                        "한도가 초기화된 뒤 다시 시도하세요.")
            return f"'{nick}' 닉네임을 찾지 못했어요. 철자를 확인하세요. ({e})"
        offline = True
    if not offline:
        try:
            ids = get_matches(ouid, mtype, n, key)
        except ApiError as e:
            if e.status != 429:
                return f"경기 목록을 불러오지 못했어요. API 키와 경기 종류를 확인하세요. ({e})"
            offline = True
    if offline:  # 한도 소진 → DB에 쌓인 경기로 분석
        st.session_state.api_exhausted = True
        ids = db_match_ids(ouid, mtype, n)
        if len(ids) < 3:
            return (f"오늘 넥슨 API 한도를 다 썼고, DB에 '{nick}'의 경기가 {len(ids)}개뿐이라 분석이 어려워요. "
                    "한도가 초기화된 뒤 다시 시도하세요.")
        st.info(f"오늘 넥슨 API 한도를 다 써서, DB에 저장된 '{nick}'의 경기 {len(ids)}개로 분석했어요. "
                "최신 경기가 빠져 있을 수 있고, 예전에 저장된 경기는 선수별 기록이 비어 있을 수 있어요.")
    details = fetch_details(ids, key, label)
    db_save(details, ouid)
    try:
        if offline:
            raise ApiError("offline", 429)
        best = next((d for d in get_maxdiv(ouid, key) if d.get("matchType") == mtype), {})
    except ApiError:
        best = db_user_division(ouid)
    max_div, max_date = best.get("division"), (best.get("achievementDate") or "")[:10]
    opp_ouids = tuple(sorted({x.get("ouid") for d in details for x in d.get("matchInfo") or []
                              if x.get("ouid") != ouid}))
    est_div, n_opp = estimate_tier(opp_ouids)
    # 경기 기록에 경기 당시 등급이 있으면 그게 가장 정확 (최근 경기 기준)
    recent_divs = [x.get("division") for d in sorted(details, key=lambda d: d.get("matchDate", ""), reverse=True)[:10]
                   for x in d.get("matchInfo") or [] if x.get("ouid") == ouid and x.get("division")]
    match_div = pd.Series(recent_divs).mode().iat[0] if recent_divs else None
    recent_max = bool(max_date) and max_date >= (pd.Timestamp.now() - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    if match_div:  # 경기 기록상 등급
        est_div, n_opp = int(match_div), -2
    elif recent_max:  # 최근에 최고 등급을 찍었으면 그게 현재 등급에 가장 가까움
        est_div, n_opp = max_div, -1
    division = est_div or max_div
    details.sort(key=lambda d: d.get("matchDate", ""), reverse=True)

    me_rows, op_rows, me_shots, op_shots, players, ctrls = [], [], [], [], [], []
    for d in details:
        info = d.get("matchInfo") or []
        if len(info) != 2:
            continue
        me = next((x for x in info if x.get("ouid") == ouid), None)
        if me is None:
            continue
        op = info[1] if info[0] is me else info[0]
        md = me.get("matchDetail") or {}
        if md.get("matchEndType", 0) != 0:
            continue  # 몰수승/몰수패 제외
        idx = len(me_rows)
        m = side_metrics(me, op)
        m.update(idx=idx, date=d.get("matchDate", "")[:16].replace("T", " "),
                 opp=op.get("nickname"), score=f"{goals(me)} : {goals(op)}",
                 result=md.get("matchResult"))
        me_rows.append(m)
        op_rows.append(side_metrics(op, me))
        me_shots += [{**s, "_m": idx} for s in me.get("shootDetail") or []]
        op_shots += [{**s, "_m": idx} for s in op.get("shootDetail") or []]
        if md.get("controller"):
            ctrls.append(md["controller"])
        for p in me.get("player") or []:
            stt = p.get("status") or {}
            if p.get("spPosition") == 28 or not stt.get("spRating"):
                continue
            players.append({"spId": p.get("spId"), "pos": p.get("spPosition"), "m": idx,
                            "grade": p.get("spGrade"),
                            **{k: (stt.get(k) or 0) for k in PLAYER_KEYS}})
    if len(me_rows) < 3:
        return f"'{nick}'의 분석 가능한 경기가 3경기 미만이에요. 경기 종류를 바꾸거나 경기 수를 늘려 보세요."

    me_df, bm_df = pd.DataFrame(me_rows), pd.DataFrame(op_rows)
    ctrl = pd.Series(ctrls).mode().iat[0] if ctrls else None
    return {"details": details, "nick": nick, "ouid": ouid, "division": division, "max_div": max_div,
            "max_date": max_date, "est_div": est_div, "n_opp": n_opp, "me": me_df, "bm": bm_df,
            "me_shots": me_shots, "op_shots": op_shots, "players": pd.DataFrame(players),
            "cmp": compare(me_df, bm_df), "tags": style_tags(me_df, bm_df),
            "controller": CONTROLLER.get(str(ctrl).lower(), ctrl) if ctrl else None}


def loss_pattern(r):
    """2) 진 경기 vs 이긴 경기 비교."""
    me = r["me"]
    W, L = me[me.result == "승"], me[me.result == "패"]
    if len(W) < 3 or len(L) < 3:
        return None
    items = []
    keys = [(k, l, hb) for k, l, hb in PERF if k not in LOSS_EXCLUDE] + \
           [(k, l, None) for k, l in STYLE]
    for k, label, hb in keys:
        a, b = W[k].astype(float).dropna(), L[k].astype(float).dropna()
        if len(a) < 3 or len(b) < 3:
            continue
        sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        if not sp:
            continue
        d = (b.mean() - a.mean()) / sp
        if abs(d) >= 0.5:
            items.append({"지표": label, "이긴 경기": a.mean(), "진 경기": b.mean(),
                          "d": d, "hb": hb})
    items.sort(key=lambda x: -abs(x["d"]))

    wi, li = set(W.idx), set(L.idx)
    tw = timing([s for s in r["op_shots"] if s["_m"] in wi]) / len(W)
    tl = timing([s for s in r["op_shots"] if s["_m"] in li]) / len(L)
    gap = tl - tw
    bucket = gap.idxmax() if gap.max() > 0.15 else None
    return {"items": items[:5], "bucket": bucket,
            "tl": tl[bucket] if bucket else 0, "tw": tw[bucket] if bucket else 0,
            "nw": len(W), "nl": len(L)}


def matchup(mine, opp):
    """3) 내 강점 × 상대 약점 → 전략, 상대 강점 × 내 약점 → 경고."""
    me = dict(zip(mine["cmp"].key, mine["cmp"].효과))
    op = dict(zip(opp["cmp"].key, opp["cmp"].효과))
    plans = sorted([(me.get(a, 0) - op.get(b, 0), t) for a, b, t in MATCHUP
                    if me.get(a, 0) >= 0.2 and op.get(b, 0) <= -0.2], reverse=True)
    risks = sorted([(op.get(a, 0) - me.get(b, 0), t) for a, b, t in RISK
                    if op.get(a, 0) >= 0.2 and me.get(b, 0) <= -0.2], reverse=True)
    extra = []
    my_style, op_style = mine["tags"][0], opp["tags"][0]
    if op_style == "역습형" and my_style == "점유형":
        extra.append("상대는 역습형이고 나는 점유형. 점유할 때 뒷공간 커버를 남겨둬")
    elif op_style == "점유형" and my_style == "역습형":
        extra.append("상대는 점유형이고 나는 역습형. 내 스타일대로 받아치기 좋은 매치업")
    return [t for _, t in plans[:2]], [t for _, t in risks[:2]], extra


def fmt(v):
    return "-" if pd.isna(v) else f"{v:.1f}"



# ================================================================ DB (선택: 없으면 앱만 단독 동작)
def _secret(name):
    v = os.environ.get(name, "")
    try:
        v = v or st.secrets.get(name, "")
    except Exception:
        pass
    return (v or "").strip()


@st.cache_resource(show_spinner=False)
def db():
    url = _secret("DATABASE_URL")
    if not url:
        return None
    try:
        c = db_connect(url)
        c.execute("set statement_timeout = '6s'")  # 느린 쿼리가 화면을 붙잡지 않게
        return c
    except Exception:
        return None


def cursor():
    c = db()
    if c is None:
        return None
    try:
        cur = c.cursor()
        cur.execute("select 1")
        cur.fetchone()
        return cur
    except Exception:
        db.clear()
        c = db()
        return c.cursor() if c else None


@st.cache_data(ttl=3600, show_spinner=False)
def load_model():
    cur = cursor()
    if cur is None:
        return DEFAULT_COEF, None, 0
    try:
        return latest_model(cur)
    except Exception:
        return DEFAULT_COEF, None, 0


@st.cache_resource
def saved_ids():
    return set()  # 이미 DB에 저장한 경기 (중복 저장 방지)


def db_save(details, ouid):
    """조회한 경기를 DB에 쌓는 작업을 뒤에서 따로 돌림 → 분석 화면을 기다리게 하지 않음."""
    url = _secret("DATABASE_URL")
    if not url:
        return
    done = saved_ids()
    todo = [d for d in details if d.get("matchId") not in done]
    if not todo:
        return
    _, flip, _ = load_model()
    if flip is None:
        flip = detect_flip([s for d in details for x in d.get("matchInfo") or []
                            for s in x.get("shootDetail") or []])

    def work():
        try:
            conn = db_connect(url)
            cur = conn.cursor()
            for d in todo:
                save_match(cur, d, flip)
                done.add(d.get("matchId"))
            cur.execute("insert into crawl_queue(ouid) values (%s) on conflict do nothing", (ouid,))
            conn.close()
        except Exception:
            pass

    threading.Thread(target=work, daemon=True).start()


@st.cache_data(ttl=1800, show_spinner=False)
def tier_population(mtype, division):
    """같은 등급 유저들의 경기당 xG/허용xG/결정력/막판방어 분포. 30명 미만이면 등급 범위를 넓힘."""
    cur = cursor()
    if cur is None or division is None:
        return None
    try:
        ids = sorted(d["divisionId"] for d in meta("division"))
        names = {d["divisionId"]: d["divisionName"] for d in meta("division")}
        if division not in ids:
            return None
        cur.execute("select max(id) from patches")
        pid = cur.fetchone()[0]
        i = ids.index(division)
        for span in (0, 1, 2):
            near = ids[max(0, i - span): i + span + 1]
            for patch in (pid, None):
                cur.execute("""
                  select avg(xg), avg(xga), avg(gf - xg), avg(ga - xga) from side_summary
                  where match_type = %s and division = any(%s)
                    and (%s::int is null or patch_id = %s)
                  group by ouid having count(*) >= 10""", (mtype, near, patch, patch))
                rows = cur.fetchall()
                if len(rows) >= 30:
                    return {"pop": np.array(rows, dtype=float), "n": len(rows),
                            "divs": [names.get(d, str(d)) for d in near],
                            "patch_only": patch is not None}
    except Exception:
        return None
    return None


@st.cache_data(ttl=600, show_spinner=False)
def estimate_tier(opp_ouids):
    """추정 현재 등급 = 최근 상대들의 등급 중앙값 (매칭은 비슷한 실력끼리 잡히니까).
    최고 등급을 90일 안에 찍은 상대가 5명 이상이면 그 상대들만 씀 (예전 시즌 기록 제외)."""
    cur = cursor()
    if cur is None or not opp_ouids:
        return None, 0
    try:
        cur.execute("""
          with t as (
            select u.ouid,
                   coalesce(c.cur_division, u.max_division) as d,
                   (c.cur_division is not null or u.max_division_date > now() - interval '90 days') as recent
            from users u left join user_cur_division c on c.ouid = u.ouid and c.match_type = 50
            where u.ouid = any(%s))
          select percentile_disc(0.5) within group (order by d) filter (where recent),
                 count(*) filter (where recent),
                 percentile_disc(0.5) within group (order by d), count(*)
          from t where d is not null""", (list(opp_ouids),))
        v_recent, n_recent, v_all, n_all = cur.fetchone()
        if n_recent >= 5:
            return v_recent, n_recent
        return (v_all, n_all) if n_all >= 5 else (None, n_all)
    except Exception:
        return None, 0


@st.cache_data(ttl=1800, show_spinner=False)
def tier_table(mtype):
    cur = cursor()
    if cur is None:
        return None
    try:
        cur.execute("select * from tier_metrics where match_type = %s and users >= 10", (mtype,))
        cols = [c.name for c in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=cols)
        return df if len(df) else None
    except Exception:
        return None



# ---------------------------------------------------------------- DB에서 경기 꺼내 쓰기 (넥슨 호출 절약)
SIDE_MAP = {  # DB 열 → 넥슨 응답 구조
    "matchDetail": {"matchResult": "result", "possession": "possession", "foul": "fouls",
                    "averageRating": "rating", "OffsideCount": "offside", "controller": "controller"},
    "shoot": {"shootTotal": "shots", "effectiveShootTotal": "sot", "goalTotal": "gf",
              "goalTotalDisplay": "gf", "shootInPenalty": "shots_box", "shootHeading": "shots_head"},
    "pass": {"passTry": "pass_try", "passSuccess": "pass_succ", "throughPassTry": "through_try",
             "throughPassSuccess": "through_succ", "longPassTry": "long_try"},
    "defence": {"tackleTry": "tackle_try", "tackleSuccess": "tackle_succ",
                "blockTry": "block_try", "blockSuccess": "block_succ"},
}


@st.cache_data(ttl=600, show_spinner=False)
def details_from_db(match_ids):
    """DB에 저장된 경기를 넥슨 응답과 같은 모양으로 복원 (선수별 기록은 저장 안 해서 비어 있음)."""
    cur = cursor()
    if cur is None or not match_ids:
        return {}
    try:
        cur.execute("select match_id, match_type, match_date from matches where match_id = any(%s)",
                    (list(match_ids),))
        ms = {r[0]: r for r in cur.fetchall()}
        if not ms:
            return {}
        cur.execute("select * from match_sides where match_id = any(%s)", (list(ms),))
        cols = [c.name for c in cur.description]
        sides = [dict(zip(cols, r)) for r in cur.fetchall()]
        cur.execute("select match_id, ouid, x, y, type, result, in_penalty, goal_time, assisted, "
                    "assist_x, assist_y from shots where match_id = any(%s)", (list(ms),))
        shots = {}
        for mid, o, x, y, t, res, ip, gt, a, ax, ay in cur.fetchall():
            shots.setdefault((mid, o), []).append(
                {"x": x, "y": y, "type": t, "result": res, "inPenalty": ip, "goalTime": gt or 0,
                 # 어시스트 좌표 칸이 생긴 뒤 저장된 슈팅만 판단 (그 전 행은 정보 없음)
                 # (0.5, 0.5)는 넥슨의 '어시스트 없음' 기본 좌표 → 잠깐 잘못 저장된 행도 바로잡음
                 "assist": None if ax is None or a is None else bool(a) and not (ax == 0.5 and ay == 0.5),
                 "assistX": ax, "assistY": ay})
        players = {}
        try:
            from common import PLAYER_FIELDS
            cols = ["match_id", "ouid", "sp_id", "sp_position", "sp_grade", *PLAYER_FIELDS.values()]
            cur.execute(f"select {','.join(cols)} from player_stats where match_id = any(%s)", (list(ms),))
            for r in cur.fetchall():
                stt = dict(zip(PLAYER_FIELDS.keys(), r[5:]))
                players.setdefault((r[0], r[1]), []).append(
                    {"spId": r[2], "spPosition": r[3], "spGrade": r[4], "status": stt})
        except Exception:
            players = {}
        out = {}
        for mid, mt, md in ms.values():
            info = []
            for sd in (x for x in sides if x["match_id"] == mid):
                side = {"ouid": sd["ouid"], "nickname": sd["nickname"], "division": sd.get("division"),
                        "player": players.get((mid, sd["ouid"]), []),
                        "shootDetail": shots.get((mid, sd["ouid"]), [])}
                for part, mp in SIDE_MAP.items():
                    side[part] = {k: sd.get(v) for k, v in mp.items()}
                side["matchDetail"]["matchEndType"] = 0
                info.append(side)
            if len(info) == 2:
                out[mid] = {"_src": "db", "matchId": mid, "matchType": mt,
                            "matchDate": md.strftime("%Y-%m-%dT%H:%M:%S") if md else "",
                            "matchInfo": info}
        return out
    except Exception:
        return {}


def db_find_ouid(nick):
    cur = cursor()
    if cur is None:
        return None
    try:
        cur.execute("select ouid from users where nickname = %s limit 1", (nick,))
        row = cur.fetchone()
        return row[0] if row else None
    except Exception:
        return None


def db_match_ids(ouid, mtype, n):
    cur = cursor()
    if cur is None:
        return []
    try:
        cur.execute("select m.match_id from match_sides ms join matches m using (match_id) "
                    "where ms.ouid = %s and m.match_type = %s order by m.match_date desc limit %s",
                    (ouid, mtype, n))
        return [r[0] for r in cur.fetchall()]
    except Exception:
        return []


def db_user_division(ouid):
    cur = cursor()
    if cur is None:
        return {}
    try:
        cur.execute("select max_division, max_division_date from users where ouid = %s", (ouid,))
        row = cur.fetchone()
        if not row or row[0] is None:
            return {}
        return {"division": row[0], "achievementDate": row[1].strftime("%Y-%m-%d") if row[1] else ""}
    except Exception:
        return {}


# ================================================================ 진단 (xG 분해)
COMP4 = [("create", "찬스 창출", 1, "경기당 xG · 좋은 슈팅 기회를 얼마나 만드는지"),
         ("finish", "결정력", 1, "실제 골 − xG · 기대보다 더 넣는지"),
         ("defend", "수비 조직", -1, "경기당 허용 xG · 위험한 찬스를 얼마나 내주는지"),
         ("keeper", "막판 방어", -1, "실제 실점 − 허용 xG · 골키퍼·블록, 그리고 운")]
COMP3 = [("balance", "찬스 싸움", 1, "경기당 (xG − 허용 xG) · 상대보다 좋은 찬스를 많이 만드는지"),
         COMP4[1], COMP4[3]]
POP_COL = {"create": 0, "defend": 1, "finish": 2, "keeper": 3}
TYPE_NAME = {"create": "찬스 부족형", "finish": "마무리 부족형", "defend": "수비 조직 불안형",
             "keeper": "막판 방어 불안형", "balance": "찬스 싸움 열세형"}
CONCL_SELF = {
    "create": "마무리보다 슈팅 기회 자체가 부족한 게 핵심이에요. 박스 안으로 들어가는 횟수를 늘리는 게 우선이에요.",
    "finish": "찬스는 만드는데 기대만큼 못 넣고 있어요. 슈팅 위치와 종류 선택을 점검하세요. 다만 결정력은 운의 영향도 커요.",
    "defend": "상대에게 좋은 찬스를 너무 많이 내주고 있어요. 박스 앞 공간과 수비 라인 간격부터 점검하세요.",
    "keeper": "허용한 찬스에 비해 실점이 많아요. 블록이나 골키퍼 문제일 수도 있지만 운의 영향이 가장 큰 항목이에요.",
    "balance": "상대보다 좋은 찬스를 덜 만들거나 더 많이 내주고 있어요. 공격과 수비 중 어디서 밀리는지는 비교 데이터가 쌓이면 나눠서 볼 수 있어요.",
}
CONCL_SCOUT = {
    "create": "찬스를 잘 못 만드는 상대예요. 라인을 올려 압박해도 위험이 적어요.",
    "finish": "마무리가 약한 상대예요. 슈팅은 좀 내줘도 박스 안 결정적 찬스만 막으세요.",
    "defend": "좋은 찬스를 많이 내주는 상대예요. 박스 안 침투를 적극적으로 노리세요.",
    "keeper": "허용 찬스 대비 실점이 많은 상대예요. 유효슈팅을 최대한 많이 만드세요.",
    "balance": "찬스 싸움에서 자주 밀리는 상대예요. 주도권을 잡고 운영하세요.",
}
SURE = ("확실한 강점", "확실한 약점")


def per_match_xg(r, coef, flip):
    n = len(r["me"])
    xf, xa = np.zeros(n), np.zeros(n)
    for s, p in zip(r["me_shots"], xg_values(r["me_shots"], coef, flip)):
        xf[s["_m"]] += p
    for s, p in zip(r["op_shots"], xg_values(r["op_shots"], coef, flip)):
        xa[s["_m"]] += p
    return xf, xa


def diagnose(r, coef, flip, tier):
    me = r["me"]
    n = len(me)
    xf, xa = per_match_xg(r, coef, flip)
    gf, ga = me.gf.values.astype(float), me.ga.values.astype(float)
    arr = {"create": xf, "finish": gf - xf, "defend": xa, "keeper": ga - xa, "balance": xf - xa}
    if tier:
        pop = tier["pop"]
        comps = COMP4
        bench = {k: pop[:, c].mean() for k, c in POP_COL.items()}
    else:
        comps = COMP3
        bench = {"balance": 0.0, "finish": 0.0, "keeper": 0.0}
    idx = np.random.default_rng(0).integers(0, n, (2000, n))  # 부트스트랩: 경기 재추출
    out = []
    for key, name, sign, desc in comps:
        a, b = arr[key], bench[key]
        boot = sign * (a[idx].mean(1) - b)
        est = sign * (a.mean() - b)
        lo95, hi95 = np.percentile(boot, [2.5, 97.5])
        lo80, hi80 = np.percentile(boot, [10, 90])
        if lo95 > 0:
            verdict = "확실한 강점"
        elif hi95 < 0:
            verdict = "확실한 약점"
        elif lo80 > 0:
            verdict = "강점 가능성"
        elif hi80 < 0:
            verdict = "약점 가능성"
        else:
            verdict = "판단 보류"
        need = None
        se = boot.std()
        if verdict not in SURE and abs(est) > 0.03 and se > 0:
            need = int(min(500, max(5, np.ceil(n * (1.96 * se / abs(est)) ** 2 - n))))
        pct = None
        if tier:
            vals = tier["pop"][:, POP_COL[key]]
            better = (vals > a.mean()).mean() if sign > 0 else (vals < a.mean()).mean()
            pct = max(1, int(round(100 * better)))
        out.append({"key": key, "name": name, "desc": desc, "value": a.mean(), "bench": b,
                    "impact": est, "verdict": verdict, "need": need, "pct": pct})
    return out, xf, xa


def _resid(shots, xs, mask, n):
    m = np.array([mask(s) for s in shots], dtype=bool)
    if not m.any():
        return np.nan
    g = np.array([s.get("result") == 3 for s in shots])[m].sum()
    return (g - xs[m].sum()) / n


def drill(r, key, coef, flip):
    """가장 큰 약점을 세부 지표로 쪼개서 보여줌 (이번 경기 상대들과 비교)."""
    me, bm, n = r["me"], r["bm"], len(r["me"])
    ms, os_ = r["me_shots"], r["op_shots"]
    xm, xo = xg_values(ms, coef, flip), xg_values(os_, coef, flip)
    avg = lambda x: x.mean() if len(x) else np.nan
    rows = []
    if key in ("create", "balance"):
        rows += [("경기당 슈팅", me.shots.mean(), bm.shots.mean()),
                 ("슈팅 1개당 xG (슈팅 질)", avg(xm), avg(xo)),
                 ("박스 안 슈팅 비중(%)", me.box_share.mean(), bm.box_share.mean())]
    if key in ("defend", "balance"):
        rows += [("경기당 허용 슈팅", me.shots_against.mean(), bm.shots_against.mean()),
                 ("허용 슈팅 1개당 xG", avg(xo), avg(xm)),
                 ("허용 유효슈팅", me.sot_against.mean(), bm.sot_against.mean())]
    if key == "finish":
        rows += [("박스 안 골 − xG (경기당)", _resid(ms, xm, lambda s: s.get("inPenalty"), n),
                  _resid(os_, xo, lambda s: s.get("inPenalty"), n)),
                 ("박스 밖 골 − xG (경기당)", _resid(ms, xm, lambda s: not s.get("inPenalty"), n),
                  _resid(os_, xo, lambda s: not s.get("inPenalty"), n)),
                 ("헤딩 골 − xG (경기당)", _resid(ms, xm, lambda s: s.get("type") == HEADER_TYPE, n),
                  _resid(os_, xo, lambda s: s.get("type") == HEADER_TYPE, n)),
                 ("유효슈팅 비율(%)", me.sot_rate.mean(), bm.sot_rate.mean())]
    if key == "keeper":
        conc = lambda g, s: g.sum() / s.sum() * 100 if s.sum() else np.nan
        rows += [("허용 유효슈팅 대비 실점(%)", conc(me.ga, me.sot_against), conc(bm.ga, bm.sot_against)),
                 ("블록 성공률(%)", me.block_rate.mean(), bm.block_rate.mean())]
    return pd.DataFrame(rows, columns=["세부 지표", "대상", "이번 경기 상대들"]).round(2)


TIER_METRICS = [  # (키, 이름, 높을수록 좋은가)
    ("xg", "찬스 창출 (경기당 xG)", True), ("xga", "허용 찬스 (경기당 허용 xG)", False),
    ("gf", "경기당 득점", True), ("ga", "경기당 실점", False), ("win_rate", "승률(%)", True),
    ("shots", "경기당 슈팅", True), ("sot_rate", "유효슈팅 비율(%)", True),
    ("box_share", "박스 안 슈팅 비중(%)", None), ("pass_rate", "패스 성공률(%)", True),
    ("through_rate", "스루패스 성공률(%)", True), ("tackle_rate", "태클 성공률(%)", True),
    ("block_rate", "블록 성공률(%)", True), ("possession", "점유율(%)", None),
]


def div_names():
    try:
        return {d["divisionId"]: d["divisionName"] for d in meta("division")}
    except Exception:
        return {}


def render_tier(r):
    st.subheader("등급별 비교")
    names = div_names()
    est, mx = r.get("est_div"), r.get("max_div")
    line = []
    if est and r.get("n_opp") == -2:
        line.append(f"현재 등급 **{names.get(est, est)}** (최근 경기 기록 기준)")
    elif est and r.get("n_opp") == -1:
        line.append(f"현재 등급 **{names.get(est, est)}** (최근 90일 안에 달성한 최고 등급)")
    elif est:
        line.append(f"추정 현재 등급 **{names.get(est, est)}** (최근 상대 {r['n_opp']}명의 등급 기준)")
    if mx:
        line.append(f"역대 최고 등급 **{names.get(mx, mx)}**" + (f" ({r['max_date']} 달성)" if r.get("max_date") else ""))
    if line:
        st.markdown(" · ".join(line))
    tt = tier_table(st.session_state.mtype)
    if tt is None:
        st.caption("등급별 평균을 만들 데이터가 아직 부족해요. 수집기가 등급마다 유저 10명 이상을 모으면 나타나요.")
        return
    tt = tt.sort_values("division")  # 등급 번호가 작을수록 높은 등급
    coef, flip, n_model = load_model()
    shots_all = r["me_shots"] + r["op_shots"]
    if flip is None:
        flip = detect_flip(shots_all)
    if not n_model:
        coef = calibrate(DEFAULT_COEF, shots_all, flip)
    xf, xa = per_match_xg(r, coef, flip)
    me = r["me"]
    mine = {"xg": xf.mean(), "xga": xa.mean(), "gf": me.gf.mean(), "ga": me.ga.mean(),
            "win_rate": me.win.mean(), "shots": me.shots.mean(), "sot_rate": me.sot_rate.mean(),
            "box_share": me.box_share.mean(), "pass_rate": me.pass_rate.mean(),
            "through_rate": me.through_rate.mean(), "tackle_rate": me.tackle_rate.mean(),
            "block_rate": me.block_rate.mean(), "possession": me.poss.mean()}
    my_row = tt[tt.division == r.get("division")]
    rows = []
    for k, label, hb in TIER_METRICS:
        v = mine.get(k)
        col = tt[k].astype(float)
        if pd.isna(v) or col.isna().all():
            continue
        nm = lambda idx: names.get(tt.loc[idx, "division"], tt.loc[idx, "division"])
        hi, lo = col.idxmax(), col.idxmin()
        if v > col.max():    # 모든 등급 평균보다 높음
            lv = f"{nm(hi)} 이상" if hb is True else f"{nm(hi)} 미만" if hb is False else "모든 등급보다 높음"
        elif v < col.min():  # 모든 등급 평균보다 낮음
            lv = f"{nm(lo)} 미만" if hb is True else f"{nm(lo)} 이상" if hb is False else "모든 등급보다 낮음"
        else:
            lv = nm((col - v).abs().idxmin())  # 평균이 가장 비슷한 등급
        rows.append({"지표": label, "나": round(v, 2),
                     "내 등급 평균": round(float(my_row[k].iloc[0]), 2) if len(my_row) else None,
                     "이 수치에 해당하는 등급": lv if hb is not None else f"{lv} (스타일 지표, 좋고 나쁨 없음)"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("'이 수치에 해당하는 등급'은 그 지표만 놓고 봤을 때 평균이 가장 비슷한 등급이에요. "
               "예를 들어 찬스 창출이 두 등급 위 수준이면 공격 전개는 이미 상위권이라는 뜻이에요.")
    with st.expander("등급별 평균 지표 전체 보기"):
        show = tt.copy()
        show.insert(0, "등급", show.division.map(lambda d: names.get(d, d)))
        show = show.drop(columns=["match_type", "division"]).rename(columns={
            "users": "유저 수", "sides": "경기 수", **{k: l for k, l, _ in TIER_METRICS}})
        st.dataframe(show.round(2), hide_index=True, width="stretch")


DIAG_CSS = """
<style>
.dg-card{border:2px solid #3b7dd8;border-radius:14px;padding:18px 22px;margin:6px 0 12px}
.dg-type{font-size:1.8rem;font-weight:800;margin-bottom:10px}
.dg-row{font-size:1.1rem;line-height:1.5;margin:4px 0}
.dg-k{display:inline-block;min-width:5.5em;font-weight:700;opacity:.8}
.dg-concl{font-size:1.1rem;margin-top:12px;padding-top:10px;border-top:1px solid rgba(128,128,128,.3)}
</style>
"""


def render_diagnosis(r, mode):
    coef, flip, n_model = load_model()
    shots_all = r["me_shots"] + r["op_shots"]
    if flip is None:
        flip = detect_flip(shots_all)
    if not n_model:
        coef = calibrate(DEFAULT_COEF, shots_all, flip)
    tier = tier_population(st.session_state.mtype, r.get("division"))
    res, xf, xa = diagnose(r, coef, flip, tier)

    weak = [x for x in res if x["verdict"] in ("확실한 약점", "약점 가능성")]
    strong = [x for x in res if x["verdict"] in ("확실한 강점", "강점 가능성")]
    hold = [x for x in res if x["verdict"] == "판단 보류"]
    primary = min(weak, key=lambda x: x["impact"]) if weak else None
    title = TYPE_NAME[primary["key"]] if primary else "뚜렷한 약점 없음"

    def tag(x):
        t = x["name"] + (" (확실)" if x["verdict"] in SURE else " (가능성)")
        return t + (f" · 상위 {x['pct']}%" if x["pct"] else "")

    rows = [f'<div class="dg-row"><span class="dg-k">특화</span>'
            f'{", ".join(tag(x) for x in sorted(strong, key=lambda x: -x["impact"])) or "없음"}</div>',
            f'<div class="dg-row"><span class="dg-k">못하는 것</span>'
            f'{", ".join(tag(x) for x in sorted(weak, key=lambda x: x["impact"])) or "없음"}</div>']
    if hold:
        rows.append('<div class="dg-row"><span class="dg-k">판단 보류</span>' + ", ".join(
            x["name"] + (f" ({x['need']}경기 더 필요)" if x["need"] else " (기준과 차이 없음)")
            for x in hold) + "</div>")
    concl = ""
    if primary:
        concl = (CONCL_SELF if mode == "self" else CONCL_SCOUT)[primary["key"]]
    elif hold and not strong:
        concl = "아직은 기준과 뚜렷하게 다른 부분이 없어요. 경기 수를 늘리면 판단이 선명해져요."
    head = "진단" if mode == "self" else "상대 유형"
    st.markdown(DIAG_CSS + f'<div class="dg-card"><div class="dg-row" style="opacity:.7">{head}</div>'
                f'<div class="dg-type">{title}</div>' + "".join(rows)
                + (f'<div class="dg-concl">{concl}</div>' if concl else "") + "</div>",
                unsafe_allow_html=True)

    parts = " / ".join(f"{x['name']} {x['impact']:+.2f}" for x in res)
    st.caption(f"기준 대비 경기당 득실 영향(골): {parts}  (+ 는 유리, − 는 불리)")

    tbl = pd.DataFrame([{"항목": x["name"], "설명": x["desc"], "대상": round(x["value"], 2),
                         "기준": round(x["bench"], 2), "영향(골/경기)": round(x["impact"], 2),
                         "판정": x["verdict"],
                         **({"등급 내 위치": f"상위 {x['pct']}%"} if tier else {}),
                         "확정까지": f"{x['need']}경기 더" if x["need"] else "-"} for x in res])
    st.dataframe(tbl, hide_index=True, width="stretch")

    if tier:
        st.caption(f"비교 기준: {', '.join(tier['divs'])} 유저 {tier['n']}명 "
                   f"({'현재 패치' if tier['patch_only'] else '최근 28일 전체'}, 유저당 10경기 이상)")
    else:
        st.caption("같은 등급 유저 데이터가 아직 부족해서, 찬스 창출과 수비 조직을 합친 '찬스 싸움'으로 봤어요. "
                   "데이터베이스에 유저가 쌓이면 둘을 나눠서 등급 내 위치까지 보여줘요.")
    st.caption(f"xG 모델: {'수집한 슈팅 ' + format(n_model, ',') + '개로 학습' if n_model else '기본 모델을 이번 경기 슈팅에 맞춰 보정한 임시 모델'}. "
               "판정은 경기를 2000번 재추출한 부트스트랩 결과예요 (확실 95%, 가능성 80%).")

    if primary:
        st.markdown(f"**{primary['name']} 세부 분석**")
        st.dataframe(drill(r, primary["key"], coef, flip), hide_index=True, width="stretch")
    with st.expander("xG가 뭐야?"):
        st.write("xG(기대 득점)는 슈팅 위치, 골대까지 거리와 각도, 헤딩 여부로 '이 슈팅이 골이 될 확률'을 계산한 값이에요. "
                 "예를 들어 xG 0.3짜리 슈팅은 10번 중 3번쯤 들어가는 기회예요. "
                 "경기당 xG는 찬스를 얼마나 만들었는지, '실제 골 − xG'는 기대보다 더 넣었는지를 보여줘요.")

# ================================================================ 시각화
CARD_CSS = """
<style>
.sc-card{border:2px solid #1f8a5b;border-radius:14px;padding:18px 22px;margin:8px 0 16px}
.sc-head{font-size:1.05rem;opacity:.75;margin-bottom:6px}
.sc-style{font-size:1.9rem;font-weight:800;margin-bottom:12px;line-height:1.25}
.sc-row{font-size:1.35rem;line-height:1.45;margin:8px 0}
.sc-tag{display:inline-block;min-width:3.2em;font-weight:800}
.sc-go{color:#1f8a5b}.sc-no{color:#d0463b}.sc-plan{color:#3b7dd8}
.sc-foot{font-size:.95rem;opacity:.7;margin-top:10px}
</style>
"""


def scout_card(opp, mine=None):
    cmp_df = opp["cmp"]
    weak = cmp_df[cmp_df.효과 <= -0.3].sort_values("효과")
    strong = cmp_df[cmp_df.효과 >= 0.3].sort_values("효과", ascending=False)
    attack = next((TIP_SCOUT_WEAK[k] for k in weak.key if k in TIP_SCOUT_WEAK),
                  "뚜렷한 약점 없음. 내 스타일대로 운영")
    caution = next((TIP_SCOUT_STRONG[k] for k in strong.key if k in TIP_SCOUT_STRONG),
                   "뚜렷한 강점 없음")
    me = opp["me"]
    w, dr = (me.result == "승").sum(), (me.result == "무").sum()
    l = len(me) - w - dr
    head = f"{opp['nick']} · 최근 {len(me)}경기 {w}승 {dr}무 {l}패"
    if opp["controller"]:
        head += f" · {opp['controller']}"

    rows = [f'<div class="sc-row"><span class="sc-tag sc-go">공략</span>{attack}</div>',
            f'<div class="sc-row"><span class="sc-tag sc-no">주의</span>{caution}</div>']
    if mine:
        plans, risks, extra = matchup(mine, opp)
        for t in plans[:1] + extra[:1]:
            rows.append(f'<div class="sc-row"><span class="sc-tag sc-plan">전략</span>{t}</div>')
        for t in risks[:1]:
            rows.append(f'<div class="sc-row"><span class="sc-tag sc-no">경고</span>{t}</div>')

    foot = ""
    t_conc = timing(opp["op_shots"])
    if t_conc.max() > 0:
        foot = f'<div class="sc-foot">상대가 가장 많이 실점하는 구간: {t_conc.idxmax()}</div>'
    st.markdown(CARD_CSS + f'<div class="sc-card"><div class="sc-head">{head}</div>'
                f'<div class="sc-style">{" · ".join(opp["tags"][:3])}</div>'
                + "".join(rows) + foot + "</div>", unsafe_allow_html=True)


def radar(cmp_df, name):
    cats = cmp_df["지표"].tolist()
    vals = (50 + 25 * cmp_df["효과"]).clip(0, 100).tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[50] * (len(cats) + 1), theta=cats + cats[:1],
                                  name="비교 평균", line=dict(dash="dot", color="#8a8f98")))
    fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=cats + cats[:1], name=name,
                                  fill="toself", line=dict(color="#1f8a5b")))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], showticklabels=False)),
                      height=420, margin=dict(l=40, r=40, t=30, b=30),
                      legend=dict(orientation="h", y=-0.1))
    return fig


def shot_map(shots, title):
    color = {1: "#3b7dd8", 2: "#b0b4ba", 3: "#1f8a5b"}
    name = {1: "유효슈팅", 2: "빗나감", 3: "골"}
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0.5, y0=0, x1=1, y1=1, line=dict(color="#6b7280"))
    fig.add_shape(type="rect", x0=0.843, y0=0.204, x1=1, y1=0.796, line=dict(color="#6b7280"))
    fig.add_shape(type="rect", x0=0.948, y0=0.368, x1=1, y1=0.632, line=dict(color="#6b7280"))
    for res in (2, 1, 3):
        pts = [s for s in shots if s.get("result") == res]
        if pts:
            fig.add_trace(go.Scatter(x=[p.get("x") for p in pts], y=[p.get("y") for p in pts],
                                     mode="markers", name=name[res],
                                     marker=dict(color=color[res], size=9 if res == 3 else 7,
                                                 opacity=0.85)))
    fig.update_layout(title=title, height=380, xaxis=dict(range=[0.48, 1.02], visible=False),
                      yaxis=dict(range=[-0.02, 1.02], visible=False, scaleanchor="x"),
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
    return fig


def player_table(pdf):
    if pdf.empty:
        return pdf
    names = {x["id"]: x["name"] for x in meta("spid")}
    pos = {x["spposition"]: x["desc"] for x in meta("spposition")}
    g = pdf.groupby("spId")
    t = g.agg(경기=("spRating", "size"), 평점=("spRating", "mean"), 골=("goal", "sum"),
              도움=("assist", "sum"), 슈팅=("shoot", "sum"), 유효슈팅=("effectiveShoot", "sum"),
              패스시도=("passTry", "sum"), 패스성공=("passSuccess", "sum"),
              드리블시도=("dribbleTry", "sum"), 드리블성공=("dribbleSuccess", "sum"),
              태클시도=("tackleTry", "sum"), 태클성공=("tackle", "sum"))
    t.insert(0, "포지션", g["pos"].agg(lambda s: pos.get(int(s.mode().iat[0]), "?")))
    t["패스성공률"] = t.패스성공 / t.패스시도.replace(0, np.nan) * 100
    t["드리블성공률"] = t.드리블성공 / t.드리블시도.replace(0, np.nan) * 100
    t["태클성공률"] = t.태클성공 / t.태클시도.replace(0, np.nan) * 100
    t.index = [names.get(i, str(i)) for i in t.index]
    cols = ["포지션", "경기", "평점", "골", "도움", "슈팅", "유효슈팅",
            "패스성공률", "드리블성공률", "태클성공률"]
    return t[cols].sort_values("평점", ascending=False).round(2)


def render_loss(r, mode):
    st.subheader("질 때 공통점" if mode == "self" else "상대가 질 때 공통점")
    lp = loss_pattern(r)
    if lp is None:
        st.write("이긴 경기와 진 경기가 각각 3경기 이상 있어야 비교할 수 있어요. 경기 수를 늘려 보세요.")
        return
    st.caption(f"이긴 {lp['nw']}경기와 진 {lp['nl']}경기를 비교했어요. "
               "지면 당연히 나빠지는 득점·실점·승률은 빼고 봤어요.")
    if not lp["items"] and not lp["bucket"]:
        st.write("이긴 경기와 진 경기 사이에 뚜렷한 차이가 없어요. 패배가 특정 패턴보다는 경기마다 다른 이유로 나오고 있어요.")
    for it in lp["items"]:
        arrow = "높아요" if it["진 경기"] > it["이긴 경기"] else "낮아요"
        st.markdown(f"- 진 경기에서 **{it['지표']}**이(가) {arrow}: "
                    f"이긴 경기 {fmt(it['이긴 경기'])} → 진 경기 {fmt(it['진 경기'])}")
    if lp["bucket"]:
        st.markdown(f"- 진 경기에서는 **{lp['bucket']}** 실점이 특히 많아요: "
                    f"경기당 {lp['tw']:.2f}골 → {lp['tl']:.2f}골")


def render_full(r, mode):
    me, bm, cmp_df = r["me"], r["bm"], r["cmp"]
    nick = r["nick"]
    names = div_names()
    best_txt = names.get(r.get("est_div") or r.get("max_div"), "-")
    best_label = ("현재 등급" if r.get("n_opp") in (-1, -2) else "추정 현재 등급") if r.get("est_div") else "역대 최고 등급"
    w, dr = (me.result == "승").sum(), (me.result == "무").sum()
    l = len(me) - w - dr
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("분석 경기", f"{len(me)}경기")
    c2.metric("전적", f"{w}승 {dr}무 {l}패")
    c3.metric("평균 스코어", f"{me.gf.mean():.1f} : {me.ga.mean():.1f}")
    c4.metric(best_label, best_txt)

    render_diagnosis(r, mode)
    render_tier(r)
    st.subheader("세부 지표 (이번 경기 상대 대비)")
    strong = cmp_df[cmp_df.효과 >= 0.3].sort_values("효과", ascending=False)
    weak = cmp_df[cmp_df.효과 <= -0.3].sort_values("효과")
    left, right = st.columns([1.1, 1])
    with left:
        st.plotly_chart(radar(cmp_df, nick), width="stretch")
    with right:
        st.subheader("플레이 스타일")
        st.write(" / ".join(r["tags"]) + (f" / {r['controller']}" if r["controller"] else ""))
        st.subheader("강점" if mode == "self" else "주의할 점 (상대 강점)")
        if strong.empty:
            st.write("비교 평균보다 뚜렷하게 앞서는 지표는 없어요.")
        for _, x in strong.iterrows():
            line = f"**{x.지표}** {fmt(x.대상)} (평균 {fmt(x['비교 평균'])})"
            if mode == "scout" and x.key in TIP_SCOUT_STRONG:
                line += f" — {TIP_SCOUT_STRONG[x.key]}"
            st.markdown("- " + line)
        st.subheader("약점" if mode == "self" else "공략 포인트 (상대 약점)")
        if weak.empty:
            st.write("비교 평균보다 뚜렷하게 뒤처지는 지표는 없어요.")
        tips = TIP_SELF_WEAK if mode == "self" else TIP_SCOUT_WEAK
        for _, x in weak.iterrows():
            line = f"**{x.지표}** {fmt(x.대상)} (평균 {fmt(x['비교 평균'])})"
            if x.key in tips:
                line += f" — {tips[x.key]}"
            st.markdown("- " + line)

    render_loss(r, mode)

    st.subheader("시간대별 득점·실점 (경기당)")
    tdf = pd.DataFrame({"득점": timing(r["me_shots"]), "실점": timing(r["op_shots"])}) / len(me)
    st.bar_chart(tdf, color=["#1f8a5b", "#d0463b"])

    s1, s2 = st.columns(2)
    s1.plotly_chart(shot_map(r["me_shots"], "슈팅 위치"), width="stretch")
    s2.plotly_chart(shot_map(r["op_shots"], "허용한 슈팅 위치"), width="stretch")

    with st.expander("전체 지표 비교표"):
        show = cmp_df[["지표", "대상", "비교 평균", "차이", "효과"]].copy()
        show.columns = ["지표", nick, "비교 평균", "차이", "효과크기 (+ 좋음)"]
        st.dataframe(show.round(2), hide_index=True, width="stretch")
        sty = pd.DataFrame({"지표": [LABEL[k] for k, _ in STYLE],
                            nick: [me[k].mean() for k, _ in STYLE],
                            "비교 평균": [bm[k].mean() for k, _ in STYLE]}).round(1)
        st.dataframe(sty, hide_index=True, width="stretch")
    with st.expander("선수별 기록"):
        if st.toggle("불러오기 (처음 한 번은 선수 이름 데이터를 받느라 몇 초 걸려요)",
                     key=f"pt_{mode}_{r['nick']}"):
            st.dataframe(player_table(r["players"]), width="stretch")
    with st.expander("최근 경기"):
        st.dataframe(me[["date", "opp", "score", "result"]]
                     .rename(columns={"date": "일시", "opp": "상대", "score": "스코어",
                                      "result": "결과"}),
                     hide_index=True, width="stretch")



# ================================================================ 심층 분석 화면
@st.cache_data(ttl=3600, show_spinner=False)
def pop_flow():
    """DB에 쌓인 전체 공식경기로 계산한 경기 흐름 평균."""
    cur = cursor()
    if cur is None:
        return None
    try:
        cur.execute("select s.match_id, s.ouid, s.goal_time from shots s join matches m using (match_id) "
                    "where s.result = 3 and m.match_type = %s", (st.session_state.mtype,))
        goals = pd.DataFrame(cur.fetchall(), columns=["match_id", "ouid", "goal_time"])
        cur.execute("select ms.match_id, ms.ouid, ms.result from match_sides ms join matches m using (match_id) "
                    "where m.match_type = %s", (st.session_state.mtype,))
        sides = pd.DataFrame(cur.fetchall(), columns=["match_id", "ouid", "result"])
        if len(sides) < 400:
            return None
        res = deep.population_flows(goals, sides)
        res["_matches"] = len(sides) // 2
        return res
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def pop_route_share(flip):
    """DB 전체 슈팅의 루트별 비중 (어시스트 위치가 저장된 슈팅만)."""
    cur = cursor()
    if cur is None:
        return None
    try:
        cur.execute("select x, y, in_penalty, assisted, assist_x, assist_y from shots "
                    "where assist_x is not null and assisted is not null")
        rows = cur.fetchall()
        if len(rows) < 2000:
            return None
        routes = pd.Series([deep.classify_route({"x": x, "y": y, "inPenalty": ip,
                                                 "assist": bool(a) and not (ax == 0.5 and ay == 0.5),
                                                 "assistX": ax, "assistY": ay}, flip)
                            for x, y, ip, a, ax, ay in rows])
        return routes.value_counts(normalize=True) * 100
    except Exception:
        return None


def render_deep(r, mode="self"):
    me, n = r["me"], len(r["me"])
    coef, flip, n_model = load_model()
    shots_all = r["me_shots"] + r["op_shots"]
    if flip is None:
        flip = detect_flip(shots_all)
    if not n_model:
        coef = calibrate(DEFAULT_COEF, shots_all, flip)
    who = "" if mode == "self" else f"{r['nick']}의 "

    # ---- 1. 루트
    st.subheader(f"① {who}공격·실점 루트")
    st.caption("슈팅 위치와 어시스트 위치로 찬스가 어떤 길로 만들어졌는지 분류했어요. "
               "측면 크로스 / 컷백(골라인 근처에서 뒤로 내준 패스) / 침투 패스 / 박스 근처 연계 / "
               "개인 돌파·세컨볼(어시스트 없음) / 중거리(박스 밖)")
    att = deep.route_table(r["me_shots"], coef, flip, n)
    cov = getattr(deep.route_table, "coverage", 1.0)
    dfd = deep.route_table(r["op_shots"], coef, flip, n)
    pop = pop_route_share(flip)
    if cov < 0.95:
        st.info(f"슈팅 중 {cov * 100:.0f}%만 어시스트 정보가 있어서 그 슈팅들로만 루트를 분류했어요. "
                "어시스트 정보 저장을 시작하기 전에 모인 경기가 섞여 있어서 그래요. 새 경기가 쌓일수록 정확해져요.")
    if att.empty:
        st.write("어시스트 정보가 있는 슈팅이 아직 적어서 루트를 분류할 수 없어요. "
                 "한도가 남아 있을 때 분석하면 넥슨에서 새로 받아온 경기로 분류해요.")
    for t in deep.route_insights(att, dfd, pop):
        st.markdown("- " + t)
    if len(att) and len(dfd):
        chart = pd.DataFrame({"공격 xG (경기당)": att["경기당 xG"].values,
                              "허용 xG (경기당)": dfd["경기당 xG"].values}, index=att["루트"].values)
        st.bar_chart(chart, color=["#1f8a5b", "#d0463b"], horizontal=True, stack=False)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**공격 루트**")
            st.dataframe(att.round(2), hide_index=True, width="stretch")
        with c2:
            st.markdown("**실점 루트**")
            show = dfd.copy()
            if pop is not None:
                show.insert(2, "전체 평균 비중(%)", pop.reindex(show.index).fillna(0).values)
            st.dataframe(show.round(2), hide_index=True, width="stretch")
    if pop is None:
        st.caption("어시스트 위치가 저장된 슈팅이 2,000개 이상 쌓이면 실점 루트에 '전체 평균 비중'도 같이 보여줘요.")

    # ---- 2. 흐름
    st.subheader(f"② {who}경기 흐름")
    mine = deep.flow_summary(deep.flows_from_details(r["details"], r["ouid"]))
    popf = pop_flow()
    if not mine:
        st.write("흐름을 계산할 경기가 없어요.")
    else:
        rows, flags = [], []
        cnt = mine["_n"]
        need = {"선제골 넣었을 때 승률(%)": cnt["first"], "먼저 실점했을 때 승점 획득률(%)": cnt["behind"],
                "앞서다가 못 이긴 비율(%)": cnt["led"]}
        for k, better in deep.FLOW_BETTER.items():
            v = mine.get(k)
            b = popf.get(k) if popf else np.nan
            base_n = need.get(k, cnt["all"])
            judge = "-"
            if popf and not pd.isna(v) and not pd.isna(b) and base_n >= 8:
                gap = v - b
                big = abs(gap) >= (10 if "%" in k else 0.15)
                if big:
                    good = (gap > 0) == better
                    judge = "강점" if good else "약점"
                    if not good:
                        flags.append((abs(gap), k))
            rows.append({"지표": k, "대상": None if pd.isna(v) else round(v, 1),
                         "전체 평균": None if pd.isna(b) else round(b, 1),
                         "해당 경기 수": base_n, "판정": judge if base_n >= 8 else "경기 부족"})
        for _, k in sorted(flags, reverse=True)[:2]:
            st.markdown(f"- **{k}** 약점: {deep.FLOW_TIP[k]}" if mode == "self"
                        else f"- **{k}** 약점이 있는 상대예요.")
        if popf is None:
            st.caption("DB에 공식경기가 200경기 이상 쌓이면 전체 평균과 비교해서 강점·약점을 판정해요.")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if popf:
            st.caption(f"전체 평균: DB에 쌓인 공식경기 {popf['_matches']:,}경기 기준. "
                       "차이가 10%p(경기당 수치는 0.15골) 이상이고 해당 경기가 8경기 이상일 때만 판정해요.")

    # ---- 3. 조합 패턴
    st.subheader(f"③ {who}승패를 가르는 조합 패턴")
    rules, n_all = deep.find_rules(me)
    if n_all < 30:
        st.write(f"이 분석은 30경기 이상 필요해요 (지금 {n_all}경기). 사이드바에서 분석 경기 수를 늘려 주세요.")
    elif not rules:
        st.write("승률을 뚜렷하게 바꾸는 조건 조합을 찾지 못했어요. 경기 수를 늘리면 나타날 수 있어요.")
    else:
        lose = [x for x in rules if x["z"] < 0]
        win = [x for x in rules if x["z"] > 0]
        if lose:
            st.markdown("**이럴 때 진다**")
            for x in lose:
                st.markdown(f"- {x['조건']} → 승률 **{x['승률']:.0f}%** ({x['경기 수']}경기, 평소 {x['전체 승률']:.0f}%)")
        if win:
            st.markdown("**이럴 때 이긴다**")
            for x in win:
                st.markdown(f"- {x['조건']} → 승률 **{x['승률']:.0f}%** ({x['경기 수']}경기, 평소 {x['전체 승률']:.0f}%)")
        st.caption("여러 조건을 한꺼번에 시험해서 찾은 '후보 패턴'이에요. 우연히 맞아떨어진 것도 섞일 수 있으니, "
                   "경기 수를 늘려도 계속 나오는 패턴을 믿으세요.")



# ================================================================ 선수·팀 화면 (랭커 비교 / 팀 변경 비교)
@st.cache_data(ttl=43200, show_spinner=False)
def ranker_stats(mtype, players, key):
    """TOP 1만 랭커들이 같은 선수(카드+포지션)를 썼을 때의 평균 기록 (넥슨 ranker-stats).
    반환: (기록 dict, 한도 초과 여부, 진단 정보)"""
    out, limited, diag = {}, False, {"errors": [], "sample": None, "requested": len(players)}
    for i in range(0, len(players), 10):
        chunk = [{"id": int(sp), "po": int(po)} for sp, po in players[i:i + 10]]
        try:
            res = _get("/fconline/v1/ranker-stats",
                       {"matchtype": mtype, "players": json.dumps(chunk, separators=(",", ":"))}, key)
        except ApiError as e:
            if e.status == 429:
                limited = True
                break
            diag["errors"].append(str(e)[:200])
            continue
        if diag["sample"] is None:
            diag["sample"] = (res[:1] if isinstance(res, list) else res)
        for row in (res if isinstance(res, list) else []):
            sp = row.get("spId", row.get("spid"))
            po = row.get("spPosition", row.get("po", row.get("spposition")))
            if sp is None or po is None:
                continue
            out[(int(sp), int(po))] = deep.ranker_row(row.get("status"))
    return out, limited, diag


def sp_names():
    try:
        return {x["id"]: x["name"] for x in meta("spid")}
    except Exception:
        return {}


def pos_names():
    try:
        return {x["spposition"]: x["desc"] for x in meta("spposition")}
    except Exception:
        return {}


def render_players_team(r, key):
    names, pnames = sp_names(), pos_names()
    pl = r["players"]
    label = lambda sp, pos=None: f"{names.get(int(sp), str(sp))}" + (f" ({pnames.get(int(pos), pos)})" if pos is not None else "")
    st.caption(f"{r['nick']} · 최근 {len(r['me'])}경기 기준")
    if pl is None or pl.empty:
        st.warning("선수 기록이 있는 경기가 없어요. 넥슨 API 한도가 초기화된 뒤 다시 분석하면 선수 기록이 같이 불러와져요.")
        return

    # ---- 랭커 비교
    st.subheader("① 랭커 비교: 같은 선수를 랭커보다 잘 쓰고 있을까?")
    mine = deep.player_per_game(pl, min_games=5)
    if mine.empty:
        st.write("5경기 이상 뛴 선수가 없어요. 분석 경기 수를 늘려 보세요.")
    else:
        keys = tuple((int(sp), int(row.pos)) for sp, row in mine.iterrows())
        rk, limited, rdiag = ranker_stats(st.session_state.mtype, keys, key)
        if limited:
            st.info("오늘 넥슨 API 한도를 다 써서 랭커 기록을 일부만 불러왔어요. 한도가 초기화되면 전부 보여줘요.")
        cmp_df = deep.ranker_compare(mine, rk)
        if rdiag["errors"] and not rk:
            st.warning("넥슨 랭커 기록을 불러오지 못했어요: " + rdiag["errors"][0])
        if cmp_df.empty:
            st.write("랭커 기록이 있는 선수가 없어요. 랭커들이 잘 쓰지 않는 카드이거나, 아직 랭커 기록을 못 불러온 거예요.")
        else:
            bad = cmp_df[cmp_df.verdict == "못 살리는 중"]
            good = cmp_df[cmp_df.verdict == "랭커보다 잘 씀"]
            if len(bad):
                st.markdown("**🔻 랭커만큼 못 살리고 있는 선수**")
                for x in bad.itertuples():
                    st.markdown(f"- **{label(x.spId, x.pos)}** · 랭커 대비 {x.index:.0f}% · {x.detail}")
            if len(good):
                st.markdown("**🔺 랭커보다 잘 쓰고 있는 선수**")
                for x in good.sort_values("index", ascending=False).itertuples():
                    st.markdown(f"- **{label(x.spId, x.pos)}** · 랭커 대비 {x.index:.0f}% · {x.detail}")
            if not len(bad) and not len(good):
                st.markdown("- 모든 선수가 랭커와 비슷한 수준으로 쓰이고 있어요.")
            tbl = cmp_df.assign(선수=[label(a, b) for a, b in zip(cmp_df.spId, cmp_df.pos)])
            st.dataframe(tbl[["선수", "games", "index", "verdict", "detail"]].rename(columns={
                "games": "내 경기 수", "index": "랭커 대비(%)", "verdict": "판정", "detail": "세부 (나 vs 랭커)"}).round(0),
                hide_index=True, width="stretch")
        st.caption("랭커 = 공식경기 TOP 10,000이 같은 카드를 같은 포지션에 썼을 때의 평균 (넥슨 제공). "
                   "공격수는 골·슈팅, 미드필더는 패스·도움·드리블, 수비수는 태클·패스 위주로 점수를 매겨요. "
                   "100%면 랭커와 같은 수준이에요. 내가 5경기 이상 쓴 선수만 비교해요.")

    # ---- 팀 변경 비교
    st.subheader("② 팀 변경 비교: 팀을 바꾼 게 효과가 있었을까?")
    det = deep.detect_change(pl, r["me"])
    order = r["me"].sort_values("date").idx.tolist()
    order = [m for m in order if m in set(pl.m)]
    if len(order) < 16:
        st.write(f"선수 기록이 있는 경기가 16경기 이상 필요해요 (지금 {len(order)}경기). 분석 경기 수를 늘려 보세요.")
        return
    dates = r["me"].set_index("idx").date
    if det:
        k_auto, changed, _ = det
        st.markdown(f"주전이 **{changed}명** 바뀐 시점을 찾았어요: **{dates[order[k_auto]][:10]}** 경기부터 새 팀으로 봤어요.")
    else:
        k_auto = len(order) // 2
        st.markdown("주전이 3명 이상 바뀐 시점을 찾지 못했어요. 아래에서 나눌 시점을 직접 고를 수 있어요.")
    opts = list(range(8, len(order) - 8 + 1))
    k = st.select_slider("새 팀 시작 경기 (직접 조정 가능)", options=opts,
                         value=min(max(k_auto, opts[0]), opts[-1]),
                         format_func=lambda i: dates[order[i]][:16], key=f"split_{r['nick']}")
    before, after = order[:k], order[k:]

    coef, flip, n_model = load_model()
    shots_all = r["me_shots"] + r["op_shots"]
    if flip is None:
        flip = detect_flip(shots_all)
    if not n_model:
        coef = calibrate(DEFAULT_COEF, shots_all, flip)
    xf, xa = per_match_xg(r, coef, flip)
    me = r["me"].set_index("idx")
    series = {"승률(%)": (me.win, True), "경기당 득점": (me.gf, True), "경기당 실점": (me.ga, False),
              "찬스 창출 (xG)": (pd.Series(xf), True), "허용 찬스 (xG)": (pd.Series(xa), False),
              "점유율(%)": (me.poss, None), "슈팅 수": (me.shots, True), "패스 성공률(%)": (me.pass_rate, True)}
    rows = []
    for name, (ser, hb) in series.items():
        a, b = ser.reindex(before).values, ser.reindex(after).values
        est, verdict = deep.segment_diff(a, b, True if hb is None else hb)
        rows.append({"지표": name, f"이전 팀 ({len(before)}경기)": np.nanmean(a), f"새 팀 ({len(after)}경기)": np.nanmean(b),
                     "변화": est, "판정": verdict if hb is not None else "스타일 지표"})
    st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, width="stretch")

    pairs, out_p, in_p = deep.swap_pairs(pl, before, after)
    if pairs:
        st.markdown("**바뀐 선수끼리 비교** (빠진 선수는 이전 팀, 들어온 선수는 새 팀 기록)")
        pb = deep.player_per_game(pl[pl.m.isin(before)], min_games=3)
        pa = deep.player_per_game(pl[pl.m.isin(after)], min_games=3)
        prow = []
        for o, i in pairs:
            if o in pb.index and i in pa.index:
                ro, ri = pb.loc[o], pa.loc[i]
                prow.append({"포지션": pnames.get(int(ri.pos), ri.pos), "빠진 선수": names.get(int(o), o),
                             "들어온 선수": names.get(int(i), i),
                             "평점": f"{ro.rating:.2f} → {ri.rating:.2f}",
                             "경기당 골": f"{ro.goal:.2f} → {ri.goal:.2f}",
                             "경기당 도움": f"{ro.assist:.2f} → {ri.assist:.2f}",
                             "패스 성공률": f"{ro.pass_rate:.0f} → {ri.pass_rate:.0f}" if not pd.isna(ro.pass_rate) and not pd.isna(ri.pass_rate) else "-",
                             "경기당 태클": f"{ro.tackle:.2f} → {ri.tackle:.2f}"})
        if prow:
            st.dataframe(pd.DataFrame(prow), hide_index=True, width="stretch")
    st.caption("두 구간을 경기 단위로 재추출해서 비교했어요 (확실 95%, 가능성 80%). "
               "같은 시기에 패치나 등급 변화가 있었다면 그 효과도 섞여 있을 수 있어요.")


# ================================================================ 복합 분석 (계층 모형 + 머신러닝 기여도)
@st.cache_data(ttl=1800, show_spinner=False)
def eb_population(mtype, division):
    """같은 등급 유저들의 경기당 평균·분산·경기 수 (실력 보정용). 30명이 안 되면 등급 범위를 넓힘."""
    cur = cursor()
    if cur is None or division is None:
        return None
    try:
        ids = sorted(d["divisionId"] for d in meta("division"))
        names = {d["divisionId"]: d["divisionName"] for d in meta("division")}
        if division not in ids:
            return None
        i = ids.index(division)
        for span in (0, 1, 2):
            near = ids[max(0, i - span): i + span + 1]
            cur.execute("""
              select count(*), avg(xg), var_samp(xg), avg(xga), var_samp(xga),
                     avg(gf - xg), var_samp(gf - xg), avg(ga - xga), var_samp(ga - xga)
              from side_summary where match_type = %s and division = any(%s)
              group by ouid having count(*) >= 5""", (mtype, near))
            rows = np.array(cur.fetchall(), dtype=float)
            if len(rows) >= 30:
                return {"rows": rows, "divs": [names.get(d, str(d)) for d in near]}
    except Exception:
        return None
    return None


# (키, 이름, 높을수록 좋으면 1, eb_population 결과에서 평균이 있는 열 번호)
EB_COMPS = [("create", "찬스 창출 (경기당 xG)", 1, 1), ("finish", "결정력 (골 − xG)", 1, 5),
            ("defend", "수비 조직 (경기당 허용 xG)", -1, 3), ("keeper", "막판 방어 (실점 − 허용 xG)", -1, 7)]


def eb_estimate(x, pop, col):
    """경험적 베이즈: 경기 수가 적을수록 등급 평균 쪽으로 당겨서 '진짜 실력'을 추정."""
    n_i, m_i, v_i = pop[:, 0], pop[:, col], np.nan_to_num(pop[:, col + 1], nan=0.0)
    sigma2 = float(np.sum((n_i - 1) * v_i) / max(np.sum(n_i - 1), 1))  # 경기마다 흔들리는 정도
    mu = float(np.mean(m_i))
    tau2 = max(float(np.var(m_i, ddof=1) - np.mean(sigma2 / n_i)), 0.02 * float(np.var(m_i, ddof=1)) + 1e-6)
    n = len(x)
    prec = n / sigma2 + 1 / tau2
    post = (n / sigma2 * float(np.mean(x)) + mu / tau2) / prec
    return {"obs": float(np.mean(x)), "post": post, "sd": prec ** -0.5, "mu": mu,
            "weight": (n / sigma2) / prec, "n": n}


@st.cache_resource(show_spinner=False)
def load_ml(_version):
    cur = cursor()
    if cur is None:
        return None
    try:
        import lightgbm as lgb
        cur.execute("select id, metrics, baseline, model from ml_models where id = %s", (_version,))
        row = cur.fetchone()
        if not row:
            return None
        return {"id": row[0], "metrics": row[1], "baseline": row[2],
                "booster": lgb.Booster(model_str=row[3])}
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def ml_status():
    cur = cursor()
    if cur is None:
        return None, 0
    try:
        cur.execute("select max(id) from ml_models")
        mid = cur.fetchone()[0]
        cur.execute("select count(*) from matches where match_type = 50")
        return mid, cur.fetchone()[0]
    except Exception:
        return None, 0


@st.cache_data(ttl=3600, show_spinner=False)
def user_divisions(ouids):
    cur = cursor()
    if cur is None or not ouids:
        return {}
    try:
        cur.execute("select ouid, max_division from users where ouid = any(%s)", (list(ouids),))
        return dict(cur.fetchall())
    except Exception:
        return {}


def user_feature_frame(r):
    """분석 대상의 경기마다 학습 때와 같은 방식으로 특징을 만듦."""
    opp_ids = tuple(sorted({x.get("ouid") for d in r["details"] for x in d.get("matchInfo") or []
                            if x.get("ouid") != r["ouid"]}))
    odiv = user_divisions(opp_ids)
    rows, meta_rows = [], []
    for d in r["details"]:
        info = d.get("matchInfo") or []
        me = next((x for x in info if x.get("ouid") == r["ouid"]), None)
        if me is None or len(info) != 2 or (me.get("matchDetail") or {}).get("matchEndType", 0) != 0:
            continue
        op = info[1] if info[0] is me else info[0]
        a, b = side_row(me, op), side_row(op, me)
        rows.append(match_features(a, b, r.get("division"), odiv.get(op.get("ouid"))))
        meta_rows.append({"date": d.get("matchDate", "")[:10], "opp": op.get("nickname"),
                          "score": f"{a['gf']}:{a['ga']}", "result": a["result"]})
    return pd.DataFrame(rows, columns=FEATURES).astype(float), pd.DataFrame(meta_rows)


def render_complex(r):
    st.caption(f"{r['nick']} · 최근 {len(r['me'])}경기 기준")
    coef, flip, n_model = load_model()
    shots_all = r["me_shots"] + r["op_shots"]
    if flip is None:
        flip = detect_flip(shots_all)
    if not n_model:
        coef = calibrate(DEFAULT_COEF, shots_all, flip)
    xf, xa = per_match_xg(r, coef, flip)
    gf, ga = r["me"].gf.values.astype(float), r["me"].ga.values.astype(float)
    comp = {"create": xf, "finish": gf - xf, "defend": xa, "keeper": ga - xa}

    # ---- 1. 계층 모형
    st.subheader("① 보정된 실력 (계층 모형)")
    pop = eb_population(st.session_state.mtype, r.get("division"))
    if pop is None:
        st.write("같은 등급 유저가 30명 이상(유저당 5경기 이상) 모여야 보정할 수 있어요. 수집이 진행되면 자동으로 나타나요.")
    else:
        rows, notes = [], []
        for key, name, sign, col in EB_COMPS:
            e = eb_estimate(comp[key], pop["rows"], col)
            lo, hi = e["post"] - 1.96 * e["sd"], e["post"] + 1.96 * e["sd"]
            gap = sign * (e["post"] - e["mu"])
            if (sign > 0 and lo > e["mu"]) or (sign < 0 and hi < e["mu"]):
                v = "확실한 강점"
            elif (sign > 0 and hi < e["mu"]) or (sign < 0 and lo > e["mu"]):
                v = "확실한 약점"
            else:
                v = "강점 쪽" if gap > 0.05 else "약점 쪽" if gap < -0.05 else "등급 평균 수준"
            rows.append({"항목": name, "관측값": e["obs"], "보정된 실력": e["post"],
                         "95% 범위": f"{lo:.2f} ~ {hi:.2f}", "등급 평균": e["mu"],
                         "내 기록 반영 비율": f"{e['weight'] * 100:.0f}%", "판정": v})
            if key == "finish" and e["weight"] < 0.5:
                notes.append("결정력은 운의 영향이 커서 내 기록보다 등급 평균 쪽으로 많이 보정됐어요.")
        st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, width="stretch")
        st.caption(f"비교 기준: {', '.join(pop['divs'])} 유저 {len(pop['rows'])}명. "
                   "경기 수가 적거나 경기마다 들쭉날쭉한 항목일수록 등급 평균 쪽으로 당겨서, 운에 덜 흔들리는 실력을 추정해요. "
                   + " ".join(notes))

    # ---- 2. 머신러닝 기여도
    st.subheader("② 승률 요인 분해 (머신러닝)")
    mid, n_matches = ml_status()
    ml = load_ml(mid) if mid else None
    if ml is None:
        st.write(f"복합분석 모델은 공식경기 5,000경기가 모이면 매일 새벽 자동으로 학습돼요 "
                 f"(지금 {n_matches:,}경기, {min(100, n_matches / 50):.0f}%).")
        st.progress(min(1.0, n_matches / 5000))
        return
    X, info = user_feature_frame(r)
    if len(X) < 10:
        st.write("요인 분해는 10경기 이상 필요해요.")
        return
    bl = ml["baseline"]
    base = bl.get(str(r.get("division")), bl["all"])
    contrib = ml["booster"].predict(X, pred_contrib=True)[:, :-1]
    user_c = pd.Series(contrib.mean(0), index=FEATURES)
    diff = user_c - pd.Series(base["contrib"])
    p0 = base["win"]
    pp = diff * p0 * (1 - p0) * 100  # 로그 오즈 차이 → 승률 %p (근사)
    cause = pp.drop(["my_div", "opp_div"])
    grp = cause.groupby(lambda k: FEATURE_INFO[k][1]).sum().sort_values()
    m = ml["metrics"]
    total = cause.sum()
    st.markdown(f"같은 등급 평균과 비교했을 때, 과정 지표로 설명되는 승률 차이는 **{total:+.1f}%p**예요.")
    st.bar_chart(grp.rename("승률 영향(%p)"), horizontal=True, color="#3b7dd8")
    neg, pos = cause.sort_values().head(3), cause.sort_values(ascending=False).head(3)
    st.markdown("**승률을 깎는 요인**")
    for k, v in neg.items():
        if v < -0.3:
            st.markdown(f"- {FEATURE_INFO[k][0]}: **{v:+.1f}%p** (내 평균 {X[k].mean():.2f})")
    st.markdown("**승률을 올리는 요인**")
    for k, v in pos.items():
        if v > 0.3:
            st.markdown(f"- {FEATURE_INFO[k][0]}: **{v:+.1f}%p** (내 평균 {X[k].mean():.2f})")
    ctx = pp[["my_div", "opp_div"]].sum()
    if abs(ctx) >= 1:
        st.caption(f"참고: 상대 등급 같은 상황 요인이 승률에 {ctx:+.1f}%p 영향을 줬어요 (실력과 무관한 부분).")

    lost = info.index[info.result == "패"][:8]
    if len(lost):
        st.markdown("**최근 진 경기의 주요 원인**")
        rows = []
        for i in lost:
            c = pd.Series(contrib[i], index=FEATURES).drop(["my_div", "opp_div"]) - \
                pd.Series(base["contrib"]).drop(["my_div", "opp_div"])
            worst = c.sort_values().head(2)
            rows.append({"날짜": info.date[i], "상대": info.opp[i], "스코어": info.score[i],
                         "주요 원인": " / ".join(FEATURE_INFO[k][0] for k in worst.index)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"모델 정확도: 최근 {m['n_valid']:,}경기로 검증한 결과 AUC {m['auc']:.2f}, "
               f"승패 적중률 {m['accuracy'] * 100:.0f}% (찍기 기준 {m['base_accuracy'] * 100:.0f}%). "
               "AUC는 0.5면 찍는 수준, 0.7 이상이면 쓸 만한 수준이에요. "
               "요인은 상관관계라서 '이걸 고치면 반드시 이긴다'는 뜻은 아니에요.")

# ================================================================ 화면
st.title("FC 전력분석실")

with st.sidebar:
    server_key = _secret("NEXON_API_KEY")
    if server_key:  # 운영자 키가 있으면 방문자는 아무것도 입력 안 해도 됨 (키는 화면에 절대 노출 안 함)
        with st.expander("내 API 키로 쓰기 (선택)"):
            own_key = st.text_input("넥슨 Open API 키", type="password", key="own_key",
                                    help="많이 쓰는 경우에만 넣으세요. 비워두면 기본 키를 써요.")
        api_key = own_key or server_key
    else:
        api_key = st.text_input("넥슨 Open API 키", type="password",
                                help="openapi.nexon.com 에서 FC온라인 애플리케이션을 등록하면 발급돼요.")
    try:
        mtypes = {m["desc"]: m["matchtype"] for m in meta("matchtype")}
    except Exception:
        mtypes = {"공식경기": 50, "감독모드": 52, "친선경기": 40}
    names = list(mtypes)
    mname = st.selectbox("경기 종류", names,
                         index=names.index("공식경기") if "공식경기" in names else 0)
    n_games = st.slider("상세 분석 경기 수", 10, 100, 30, step=10)

if not api_key:
    st.info("왼쪽 사이드바에 넥슨 Open API 키를 입력하세요.")
    st.stop()
st.session_state.api_key = api_key
st.session_state.mtype = mtypes[mname]
MT = mtypes[mname]

def run_with_status(nick, n, title):
    """단계별 진행 상황을 보여주면서 분석 + 무거운 계산을 미리 끝냄."""
    with st.status(f"{title}: 1/3 경기 기록 불러오는 중", expanded=True) as stt:
        st.caption(f"처음 조회하는 경기는 {n}경기 기준 20초~1분 걸려요. 한 번 불러온 경기는 다음부터 바로 떠요.")
        r = analyze(nick, MT, n, api_key)
        if isinstance(r, str):
            stt.update(label=f"{title}: 실패", state="error", expanded=False)
            return r
        stt.update(label=f"{title}: 2/3 등급 비교 데이터 확인 중")
        load_model()
        tier_population(MT, r.get("division"))
        stt.update(label=f"{title}: 3/3 화면 그리는 중")
        stt.update(label=f"{title}: 완료 ({len(r['me'])}경기)", state="complete", expanded=False)
    return r


tab_scout, tab_me, tab_deep, tab_pt, tab_cx = st.tabs(["상대 스카우팅", "내 분석", "심층 분석", "선수·팀", "복합 분석"])

with tab_scout:
    c1, c2 = st.columns([2, 1])
    onick = c1.text_input("상대 닉네임", key="op_nick",
                          placeholder="로딩 화면에 뜬 상대 닉네임")
    my_nick = c2.text_input("내 닉네임 (전략 추천용)", key="my_nick_scout",
                            value=st.session_state.get("saved_me", ""))
    if st.button("스카우팅", key="op_btn", type="primary", width="stretch") and onick:
        if my_nick:
            st.session_state.saved_me = my_nick
        opp = analyze(onick, MT, QUICK_N, api_key, "상대 최근 경기 불러오는 중")
        mine = None
        if not isinstance(opp, str) and my_nick:
            mine = analyze(my_nick, MT, 20, api_key, "내 최근 경기 불러오는 중")
        st.session_state.scout = (onick, opp, mine)
        st.session_state.pop("scout_full", None)

    if "scout" in st.session_state:  # 버튼 다시 안 눌러도 카드 유지
        onick_s, opp, mine = st.session_state.scout
        if isinstance(opp, str):
            st.error(opp)
        else:
            if isinstance(mine, str):
                st.warning(mine + " 전략 추천 없이 보여줄게요.")
                mine = None
            scout_card(opp, mine)
            if mine:
                plans, risks, extra = matchup(mine, opp)
                more = plans[1:] + extra[1:] + risks[1:]
                if more:
                    st.markdown("**추가 전략 메모**")
                    for t in more:
                        st.markdown(f"- {t}")
            if st.button(f"{onick_s} 상세 분석 보기 (최근 {n_games}경기)", key="op_full"):
                st.session_state.scout_full = run_with_status(onick_s, n_games, "상세 분석")
            full = st.session_state.get("scout_full")
            if full is not None:
                if isinstance(full, str):
                    st.error(full)
                else:
                    render_full(full, "scout")

with tab_me:
    nick = st.text_input("내 닉네임", key="me_nick",
                         value=st.session_state.get("saved_me", ""))
    if st.button("분석하기", key="me_btn", type="primary") and nick:
        st.session_state.saved_me = nick
        st.session_state.me_result = run_with_status(nick, n_games, "내 분석")
    res = st.session_state.get("me_result")  # 다른 버튼을 눌러도 결과 유지
    if res is not None:
        if isinstance(res, str):
            st.error(res)
        else:
            render_full(res, "self")

with tab_deep:
    st.caption("공격·실점 루트, 경기 흐름, 승패를 가르는 조합 패턴을 분석해요. 조합 패턴은 경기가 많을수록 정확해서 50경기 이상을 추천해요.")
    dn = st.text_input("닉네임", key="deep_nick", value=st.session_state.get("saved_me", ""))
    dgames = st.select_slider("분석 경기 수", [30, 50, 80, 100], value=50, key="deep_n")
    st.caption("처음 조회하는 경기는 넥슨 하루 호출 한도(1,000건)를 써요. 100경기면 약 100건이에요.")
    if st.button("심층 분석", key="deep_btn", type="primary") and dn:
        st.session_state.deep_result = run_with_status(dn, dgames, "심층 분석")
    dres = st.session_state.get("deep_result")
    if dres is not None:
        if isinstance(dres, str):
            st.error(dres)
        else:
            render_deep(dres, "other" if st.session_state.get("saved_me") and dres["nick"] != st.session_state.get("saved_me") else "self")

with tab_pt:
    st.caption("내 선수를 랭커들과 비교하고, 팀을 바꾼 전후를 비교해요. 팀 변경 비교는 경기가 많을수록 정확해서 80경기 이상을 추천해요.")
    pn = st.text_input("닉네임", key="pt_nick", value=st.session_state.get("saved_me", ""))
    pgames = st.select_slider("분석 경기 수", [30, 50, 80, 100], value=80, key="pt_n")
    if st.button("선수·팀 분석", key="pt_btn", type="primary") and pn:
        st.session_state.pt_result = run_with_status(pn, pgames, "선수·팀 분석")
    pres = st.session_state.get("pt_result")
    if pres is not None:
        if isinstance(pres, str):
            st.error(pres)
        else:
            render_players_team(pres, api_key)

with tab_cx:
    st.caption("운에 덜 흔들리는 보정된 실력과, 여러 지표를 함께 본 승률 요인을 보여줘요.")
    cn = st.text_input("닉네임", key="cx_nick", value=st.session_state.get("saved_me", ""))
    cgames = st.select_slider("분석 경기 수", [30, 50, 80, 100], value=50, key="cx_n")
    if st.button("복합 분석", key="cx_btn", type="primary") and cn:
        st.session_state.cx_result = run_with_status(cn, cgames, "복합 분석")
    cres = st.session_state.get("cx_result")
    if cres is not None:
        if isinstance(cres, str):
            st.error(cres)
        else:
            render_complex(cres)

st.divider()
st.caption("Data based on NEXON Open API")
