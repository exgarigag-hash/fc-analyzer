"""심층 분석: 공격·실점 루트 / 경기 흐름 / 승패를 가르는 조합 패턴.
Streamlit에 의존하지 않는 순수 계산 함수만 모아둠 (테스트하기 쉽게)."""
import math
from itertools import combinations

import numpy as np
import pandas as pd

from common import goal_minute, has_assist, xg_values

# ================================================================ 1. 공격·실점 루트
ROUTES = {
    "cross": "측면 크로스", "cutback": "컷백", "through": "침투 패스",
    "combo": "박스 근처 연계", "solo": "개인 돌파·세컨볼", "long": "중거리",
}


def _xy(x, y, flip):
    x, y = float(x or 0), float(y or 0)
    return (1 - x if flip else x), y


def classify_route(s, flip=False):
    """슈팅 1개를 어떤 경로로 만든 찬스인지 분류 (슈팅 위치 + 어시스트 위치 기준)."""
    x, y = _xy(s.get("x"), s.get("y"), flip)
    inbox = bool(s.get("inPenalty")) or (x >= 0.843 and 0.204 <= y <= 0.796)
    if not inbox:
        return "long"
    assisted = has_assist(s)
    if assisted is None:                     # 어시스트 정보가 없는 슈팅 → 분류 불가
        return "unknown"
    if not assisted or s.get("assistX") is None:
        return "solo"
    ax, ay = _xy(s.get("assistX"), s.get("assistY"), flip)
    if ax >= 0.88 and ax > x + 0.01:          # 골라인 근처에서 뒤로 내준 패스
        return "cutback"
    if (ay < 0.2 or ay > 0.8) and ax >= 0.6:  # 측면에서 올린 공
        return "cross"
    if ax < 0.72:                             # 비교적 먼 곳에서 찔러준 패스
        return "through"
    return "combo"


def route_table(shots, coef, flip, n_matches):
    """루트별 슈팅 비중, 경기당 슈팅·골·xG, 슈팅당 xG."""
    if not shots:
        return pd.DataFrame()
    xs = xg_values(shots, coef, flip)
    df = pd.DataFrame({"route": [classify_route(s, flip) for s in shots], "xg": xs,
                       "goal": [s.get("result") == 3 for s in shots]})
    known = df.route != "unknown"
    route_table.coverage = float(known.mean()) if len(df) else 0.0  # 분류 가능한 슈팅 비율
    if known.sum() < 20:
        return pd.DataFrame()
    df = df[known]
    n_matches = n_matches * route_table.coverage  # 분류 가능한 슈팅이 나온 경기 수로 환산
    g = df.groupby("route").agg(shots=("xg", "size"), xg=("xg", "sum"), goals=("goal", "sum"))
    g = g.reindex(list(ROUTES), fill_value=0)
    out = pd.DataFrame({
        "루트": [ROUTES[k] for k in g.index],
        "슈팅 비중(%)": g.shots / max(len(df), 1) * 100,
        "경기당 슈팅": g.shots / n_matches,
        "경기당 xG": g.xg / n_matches,
        "경기당 골": g.goals / n_matches,
        "슈팅당 xG": g.xg / g.shots.replace(0, np.nan),
        "골 − xG (경기당)": (g.goals - g.xg) / n_matches,
    }, index=g.index)
    return out


def route_insights(att, dfd, pop_share=None):
    """루트 표에서 눈에 띄는 점을 문장으로."""
    tips = []
    if len(att):
        main = att["경기당 xG"].idxmax()
        tips.append(f"공격 찬스(xG)를 가장 많이 만드는 루트: **{ROUTES[main]}**")
        q = att[att["경기당 슈팅"] >= 0.5]["슈팅당 xG"]
        if len(q) >= 2:
            best = q.idxmax()
            if best != main:
                tips.append(f"슈팅 1개의 질이 가장 좋은 루트: **{ROUTES[best]}** (슈팅당 xG {q[best]:.2f}). 이 루트를 더 늘려볼 만해요.")
        miss = att["골 − xG (경기당)"].idxmin()
        if att.loc[miss, "골 − xG (경기당)"] < -0.15:
            tips.append(f"**{ROUTES[miss]}** 찬스에서 기대보다 덜 넣고 있어요 (경기당 {att.loc[miss, '골 − xG (경기당)']:+.2f}골).")
    if len(dfd):
        worst = dfd["경기당 xG"].idxmax()
        tips.append(f"가장 위험한 찬스를 내주는 루트: **{ROUTES[worst]}** (경기당 허용 xG {dfd.loc[worst, '경기당 xG']:.2f})")
        if pop_share is not None:
            gap = (dfd["슈팅 비중(%)"] - pop_share.reindex(dfd.index).fillna(0))
            k = gap.idxmax()
            if gap[k] >= 8:
                tips.append(f"**{ROUTES[k]}** 허용 비중이 전체 평균보다 {gap[k]:.0f}%p 높아요. 이 루트 수비가 특히 약해요.")
    return tips


# ================================================================ 2. 경기 흐름
def side_flow(my_goals, opp_goals, result):
    """한 경기의 흐름 지표. my_goals/opp_goals = 득점 시각(분) 목록."""
    ev = sorted([(m, 1) for m in my_goals] + [(m, -1) for m in opp_goals])
    lead = led = trailed = 0
    for _, d in ev:
        lead += d
        led |= lead > 0
        trailed |= lead < 0
    quick = sum(1 for m in my_goals if any(0 < o - m <= 5 for o in opp_goals))
    return {
        "first": (ev[0][1] == 1) if ev else None,
        "win": result == "승", "point": result in ("승", "무"),
        "led": bool(led), "trailed": bool(trailed),
        "late_for": sum(1 for m in my_goals if m > 75),
        "late_against": sum(1 for m in opp_goals if m > 75),
        "quick_concede": quick, "goals": len(my_goals),
    }


def flow_summary(flows):
    f = pd.DataFrame(flows)
    if f.empty:
        return {}
    scored = f[f["first"].notna()]
    first = scored[scored["first"] == True]
    behind = scored[scored["first"] == False]
    led = f[f.led]
    return {
        "선제골 비율(%)": (scored["first"] == True).mean() * 100 if len(scored) else np.nan,
        "선제골 넣었을 때 승률(%)": first.win.mean() * 100 if len(first) else np.nan,
        "먼저 실점했을 때 승점 획득률(%)": behind.point.mean() * 100 if len(behind) else np.nan,
        "앞서다가 못 이긴 비율(%)": (~led.win).mean() * 100 if len(led) else np.nan,
        "76분 이후 득점 (경기당)": f.late_for.mean(),
        "76분 이후 실점 (경기당)": f.late_against.mean(),
        "득점 후 5분 내 실점 비율(%)": f.quick_concede.sum() / f.goals.sum() * 100 if f.goals.sum() else np.nan,
        "_n": {"first": len(first), "behind": len(behind), "led": len(led), "all": len(f)},
    }


FLOW_BETTER = {  # 높을수록 좋은가
    "선제골 비율(%)": True, "선제골 넣었을 때 승률(%)": True,
    "먼저 실점했을 때 승점 획득률(%)": True, "앞서다가 못 이긴 비율(%)": False,
    "76분 이후 득점 (경기당)": True, "76분 이후 실점 (경기당)": False,
    "득점 후 5분 내 실점 비율(%)": False,
}
FLOW_TIP = {
    "선제골 비율(%)": "경기 초반 공격 전개가 느려요. 초반 15분에 더 적극적으로 나가볼 만해요.",
    "선제골 넣었을 때 승률(%)": "앞서 나간 뒤 경기를 마무리하는 힘이 약해요. 리드 시 운영(점유 유지, 라인 조절)을 점검하세요.",
    "먼저 실점했을 때 승점 획득률(%)": "먼저 실점하면 쉽게 무너져요. 실점 직후 무리하게 라인을 올리는지 확인해 보세요.",
    "앞서다가 못 이긴 비율(%)": "리드를 지키지 못하는 경기가 많아요. 앞서 있을 때 후반 수비 집중력이 떨어지는지 보세요.",
    "76분 이후 득점 (경기당)": "후반 막판 결정력이 떨어져요. 막판 교체나 공격 전환 전략을 고민해 보세요.",
    "76분 이후 실점 (경기당)": "후반 막판 실점이 많아요. 체력 관리(교체 타이밍)와 막판 수비 라인을 점검하세요.",
    "득점 후 5분 내 실점 비율(%)": "골 넣은 직후 바로 실점하는 경우가 많아요. 득점 후 킥오프 직후 수비 집중력을 챙기세요.",
}


def flows_from_details(details, ouid):
    out = []
    for d in details:
        info = d.get("matchInfo") or []
        me = next((x for x in info if x.get("ouid") == ouid), None)
        if me is None or len(info) != 2:
            continue
        op = info[1] if info[0] is me else info[0]
        gm = lambda side: [goal_minute(s.get("goalTime", 0)) for s in side.get("shootDetail") or []
                           if s.get("result") == 3 and (s.get("goalTime", 0) >> 24) < 4]
        out.append(side_flow(gm(me), gm(op), (me.get("matchDetail") or {}).get("matchResult")))
    return out


def population_flows(goals_df, sides_df):
    """DB 데이터로 전체 유저의 흐름 지표. goals_df: match_id,ouid,goal_time / sides_df: match_id,ouid,result"""
    goals_df = goals_df.copy()
    goals_df["minute"] = [goal_minute(int(g)) for g in goals_df.goal_time.fillna(0)]
    by = {k: v.minute.tolist() for k, v in goals_df.groupby(["match_id", "ouid"])}
    other = sides_df.groupby("match_id").ouid.apply(list).to_dict()
    flows = []
    for r in sides_df.itertuples():
        opp = [o for o in other.get(r.match_id, []) if o != r.ouid]
        if not opp:
            continue
        flows.append(side_flow(by.get((r.match_id, r.ouid), []),
                               by.get((r.match_id, opp[0]), []), r.result))
    return flow_summary(flows)


# ================================================================ 3. 조합 패턴 (승패를 가르는 조건)
RULE_FEATURES = {
    "poss": "점유율", "shots": "슈팅 수", "box_share": "박스 안 슈팅 비중",
    "head_share": "헤딩 슈팅 비중", "pass_rate": "패스 성공률", "long_share": "롱패스 비중",
    "through_try": "스루패스 시도", "tackle_rate": "태클 성공률", "fouls": "파울",
    "offside": "오프사이드", "shots_against": "허용 슈팅",
}


def _fmt_cond(feat, op, thr):
    unit = "%" if feat in ("poss", "box_share", "head_share", "pass_rate", "long_share", "tackle_rate") else ""
    return f"{RULE_FEATURES[feat]} {thr:.0f}{unit} {'이상' if op == '>=' else '이하'}"


def find_rules(me, min_n=8, z_min=2.0):
    """경기들을 조건으로 나눠서 승률이 크게 달라지는 조합을 찾음 (1개 조건 + 2개 조건 AND)."""
    y = (me["win"] > 0).astype(int).values
    n_all = len(y)
    if n_all < 30:
        return [], n_all
    p = y.mean()
    if p in (0, 1):
        return [], n_all
    se1 = math.sqrt(p * (1 - p))

    conds = []
    for f in RULE_FEATURES:
        if f not in me:
            continue
        v = me[f].astype(float)
        if v.isna().mean() > 0.3 or v.nunique() < 4:
            continue
        for q in (0.3, 0.5, 0.7):
            thr = float(v.quantile(q))
            for op in (">=", "<="):
                mask = ((v >= thr) if op == ">=" else (v <= thr)).fillna(False).values
                conds.append(((f, op, thr), mask))

    def score(mask):
        n = mask.sum()
        if n < min_n or n_all - n < 5:
            return None
        ps = y[mask].mean()
        return ps, n, (ps - p) / (se1 / math.sqrt(n))

    singles = []
    for c, m in conds:
        s = score(m)
        if s:
            singles.append((abs(s[2]), [c], m, s))
    singles.sort(key=lambda t: -t[0])
    rules = [t for t in singles if t[0] >= z_min]

    top = singles[:14]
    for (z1, c1, m1, _), (z2, c2, m2, _) in combinations(top, 2):
        if c1[0][0] == c2[0][0]:
            continue
        m = m1 & m2
        s = score(m)
        if s and abs(s[2]) >= z_min and abs(s[2]) >= max(z1, z2) + 0.3:
            rules.append((abs(s[2]), c1 + c2, m, s))

    rules.sort(key=lambda t: -t[0])
    picked, used = [], []
    for z, cs, m, (ps, n, zz) in rules:
        if any((m & u).sum() / max(min(m.sum(), u.sum()), 1) > 0.7 for u in used):
            continue  # 거의 같은 경기들을 가리키는 규칙은 하나만
        picked.append({"조건": " + ".join(_fmt_cond(*c) for c in cs), "승률": ps * 100,
                       "경기 수": int(n), "전체 승률": p * 100, "z": zz})
        used.append(m)
        if len(picked) >= 5:
            break
    return picked, n_all


# ================================================================ 4. 랭커 비교
def pos_group(pos):
    pos = int(pos)
    if pos == 0:
        return "GK"
    if 1 <= pos <= 8:
        return "DF"
    if 9 <= pos <= 19:
        return "MF"
    return "FW"


def player_per_game(players, min_games=5):
    """선수(카드+포지션)별 경기당 기록. players: analyze가 만든 선수 행 DataFrame."""
    if players is None or players.empty:
        return pd.DataFrame()
    df = players.copy()
    df["pos_main"] = df.groupby("spId")["pos"].transform(lambda s: s.mode().iat[0])
    g = df.groupby("spId")
    out = pd.DataFrame({
        "pos": g["pos_main"].first(), "games": g.size(),
        "goal": g["goal"].mean(), "assist": g["assist"].mean(), "shoot": g["shoot"].mean(),
        "sot_rate": g["effectiveShoot"].sum() / g["shoot"].sum().replace(0, np.nan) * 100,
        "pass_rate": g["passSuccess"].sum() / g["passTry"].sum().replace(0, np.nan) * 100,
        "dribble_rate": g["dribbleSuccess"].sum() / g["dribbleTry"].sum().replace(0, np.nan) * 100,
        "tackle": g["tackle"].mean(), "rating": g["spRating"].mean(),
    })
    return out[out.games >= min_games]


# 포지션 그룹별로 볼 지표 (키, 이름, 가중치)
GROUP_KEYS = {
    "FW": [("goal", "경기당 골", 2), ("shoot", "경기당 슈팅", 1), ("sot_rate", "유효슈팅 비율(%)", 1),
           ("assist", "경기당 도움", 1)],
    "MF": [("pass_rate", "패스 성공률(%)", 2), ("assist", "경기당 도움", 1),
           ("dribble_rate", "드리블 성공률(%)", 1), ("goal", "경기당 골", 1)],
    "DF": [("tackle", "경기당 태클", 2), ("pass_rate", "패스 성공률(%)", 1)],
    "GK": [("pass_rate", "패스 성공률(%)", 1)],
}


def ranker_row(status):
    """랭커 기록(넥슨 ranker-stats의 status) → 같은 단위로 변환."""
    st = status or {}
    f = lambda k: float(st.get(k) or 0)
    return {
        "goal": f("goal"), "assist": f("assist"), "shoot": f("shoot"),
        "sot_rate": f("effectiveShoot") / f("shoot") * 100 if f("shoot") else np.nan,
        "pass_rate": f("passSuccess") / f("passTry") * 100 if f("passTry") else np.nan,
        "dribble_rate": f("dribbleSuccess") / f("dribbleTry") * 100 if f("dribbleTry") else np.nan,
        "tackle": f("tackle"), "matches": f("matchCount"),
    }


def ranker_compare(mine, rankers):
    """mine: player_per_game 결과, rankers: {(spId, pos): ranker_row}. 선수별 판정."""
    rows = []
    for sp, r in mine.iterrows():
        rk = rankers.get((int(sp), int(r.pos)))
        if not rk:
            continue
        grp = pos_group(r.pos)
        score, wsum, detail = 0.0, 0, []
        for k, label, w in GROUP_KEYS[grp]:
            a, b = r[k], rk[k]
            if pd.isna(a) or pd.isna(b) or b == 0:
                continue
            ratio = a / b
            score += w * np.clip(ratio, 0, 2)
            wsum += w
            detail.append(f"{label} {a:.2f} vs {b:.2f}" if "(%)" not in label else f"{label} {a:.0f} vs {b:.0f}")
        if not wsum:
            continue
        idx = score / wsum  # 1.0 = 랭커와 같은 수준
        verdict = "랭커보다 잘 씀" if idx >= 1.1 else "랭커 수준" if idx >= 0.85 else "못 살리는 중"
        rows.append({"spId": int(sp), "pos": int(r.pos), "group": grp, "games": int(r.games),
                     "index": idx * 100, "verdict": verdict, "detail": " · ".join(detail)})
    return pd.DataFrame(rows).sort_values("index") if rows else pd.DataFrame()


# ================================================================ 5. 팀 변경 비교
def core_squad(players, ms, share=0.5):
    """해당 경기들에서 절반 이상 선발로 나온 선수 = 주전."""
    sub = players[players.m.isin(ms)]
    cnt = sub.groupby("spId").m.nunique()
    return set(cnt[cnt >= share * len(ms)].index)


def detect_change(players, me, min_seg=8):
    """선발 주전이 가장 많이 바뀐 시점을 찾음. 반환: (나누는 위치 k, 바뀐 인원, 시간순 경기 idx 목록)."""
    if players is None or players.empty:
        return None
    order = me.sort_values("date").idx.tolist()  # 오래된 경기 → 최근 경기
    order = [m for m in order if m in set(players.m)]
    if len(order) < 2 * min_seg:
        return None
    xi = players.groupby("m").spId.apply(set).to_dict()  # 경기별 선발 명단
    jac = lambda a, b: len(a & b) / len(a | b) if a | b else 0
    best = None
    for k in range(min_seg, len(order) - min_seg + 1):
        a, b = core_squad(players, order[:k]), core_squad(players, order[k:])
        changed = len(a - b)
        # 각 경기 명단이 자기 구간 주전과 얼마나 더 닮았는지 (나누는 시점이 정확할수록 높음)
        purity = np.mean([jac(xi[m], a) - jac(xi[m], b) for m in order[:k]] +
                         [jac(xi[m], b) - jac(xi[m], a) for m in order[k:]])
        if best is None or (changed, purity) > (best[1], best[2]):
            best = (k, changed, purity)
    if best is None or best[1] < 3:
        return None
    return best[0], best[1], order


def swap_pairs(players, before, after):
    """빠진 주전과 들어온 주전을 포지션으로 짝지음."""
    a, b = core_squad(players, before), core_squad(players, after)
    out_p, in_p = a - b, b - a
    pos = players.groupby("spId").pos.agg(lambda s: int(s.mode().iat[0]))
    pairs, used = [], set()
    for o in sorted(out_p, key=lambda x: pos.get(x, 99)):
        cands = [i for i in in_p if i not in used]
        if not cands:
            break
        exact = [i for i in cands if pos.get(i) == pos.get(o)]
        same_grp = [i for i in cands if pos_group(pos.get(i, 25)) == pos_group(pos.get(o, 25))]
        pick = (exact or same_grp or [None])[0]
        if pick is not None:
            used.add(pick)
            pairs.append((o, pick))
    return pairs, out_p, in_p


def segment_diff(arr_before, arr_after, higher_better=True, n_boot=2000, seed=0):
    """두 구간 평균 차이(이후 − 이전)의 부트스트랩 판정."""
    a, b = np.asarray(arr_before, float), np.asarray(arr_after, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan, "판단 보류"
    rng = np.random.default_rng(seed)
    d = rng.choice(b, (n_boot, len(b))).mean(1) - rng.choice(a, (n_boot, len(a))).mean(1)
    if not higher_better:
        d = -d
    lo95, hi95 = np.percentile(d, [2.5, 97.5])
    lo80, hi80 = np.percentile(d, [10, 90])
    est = b.mean() - a.mean()
    if lo95 > 0:
        v = "확실히 좋아짐"
    elif hi95 < 0:
        v = "확실히 나빠짐"
    elif lo80 > 0:
        v = "좋아졌을 가능성"
    elif hi80 < 0:
        v = "나빠졌을 가능성"
    else:
        v = "차이 없음"
    return est, v
