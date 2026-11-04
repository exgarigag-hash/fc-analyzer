"""순수 계산 함수 테스트 (DB·넥슨 API 없이 실행). 실행: python -m pytest -q"""
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import deep  # noqa: E402
from common import (DEFAULT_COEF, calibrate, detect_flip, fit_logit, goal_minute,  # noqa: E402
                    has_assist, match_features, shot_features, valid_match, xg_values)


# ---------------------------------------------------------------- common
def test_goal_minute_periods():
    assert goal_minute(30 * 60) == 30                     # 전반 30분
    assert goal_minute((1 << 24) | (10 * 60)) == 55       # 후반 경과 10분 → 55분
    assert goal_minute((1 << 24) | (60 * 60)) == 60       # 후반인데 누적 시간으로 들어온 경우
    assert goal_minute((2 << 24) | (5 * 60)) == 95        # 연장 전반


def test_shot_features_and_flip():
    d, a, h = shot_features({"x": 1.0, "y": 0.5})
    assert d == 0 and h == 0
    d1, a1, _ = shot_features({"x": 0.9, "y": 0.5})
    d2, a2, _ = shot_features({"x": 0.1, "y": 0.5}, flip=True)
    assert math.isclose(d1, d2) and math.isclose(a1, a2)
    near, far = shot_features({"x": 0.95, "y": 0.5}), shot_features({"x": 0.7, "y": 0.5})
    assert near[0] < far[0] and near[1] > far[1]          # 가까울수록 각도가 큼


def test_detect_flip():
    shots = [{"x": 0.1, "result": 3}] * 6
    assert detect_flip(shots) is True
    assert detect_flip([{"x": 0.9, "result": 3}] * 6) is False
    assert detect_flip([]) is False


def test_xg_values_range_and_order():
    xs = xg_values([{"x": 0.97, "y": 0.5}, {"x": 0.7, "y": 0.5}], DEFAULT_COEF)
    assert ((xs > 0) & (xs < 1)).all() and xs[0] > xs[1]
    assert len(xg_values([], DEFAULT_COEF)) == 0


def test_calibrate_matches_total_goals():
    rng = np.random.default_rng(0)
    shots = [{"x": float(x), "y": 0.5, "result": 3 if i % 5 == 0 else 1}
             for i, x in enumerate(rng.uniform(0.7, 0.99, 200))]
    c = calibrate(DEFAULT_COEF, shots)
    assert abs(xg_values(shots, c).sum() - 40) < 0.5


def test_has_assist():
    assert has_assist({"assist": True}) is True
    assert has_assist({"assist": False}) is False
    assert has_assist({"assistSpId": -1}) is False
    assert has_assist({"assistSpId": 123}) is True
    assert has_assist({}) is None


def test_fit_logit_recovers_sign():
    rng = np.random.default_rng(1)
    x = rng.normal(size=3000)
    y = (rng.uniform(size=3000) < 1 / (1 + np.exp(-(0.5 + 2 * x)))).astype(float)
    w = fit_logit(np.column_stack([np.ones_like(x), x]), y, l2=0.0)
    assert abs(w[0] - 0.5) < 0.2 and abs(w[1] - 2) < 0.3


def test_valid_match_excludes_forfeit():
    ok = {"matchInfo": [{"matchDetail": {"matchEndType": 0}}] * 2}
    bad = {"matchInfo": [{"matchDetail": {"matchEndType": 1}}, {"matchDetail": {}}]}
    assert valid_match(ok) and not valid_match(bad) and not valid_match({"matchInfo": []})


def test_match_features_ratios():
    me = {"shots": 10, "sot": 5, "shots_box": 8, "pass_try": 0, "possession": 55}
    f = match_features(me, {"shots": 4, "sot": 1})
    assert f["sot_rate"] == 0.5 and f["box_share"] == 0.8
    assert np.isnan(f["pass_rate"])                       # 0으로 나누면 NaN
    assert f["opp_sot_rate"] == 0.25 and np.isnan(f["my_div"])


# ---------------------------------------------------------------- deep
def test_classify_route():
    assert deep.classify_route({"x": 0.6, "y": 0.5}) == "long"
    assert deep.classify_route({"x": 0.9, "y": 0.5, "inPenalty": True}) == "unknown"
    assert deep.classify_route({"x": 0.9, "y": 0.5, "inPenalty": True, "assist": False}) == "solo"
    base = {"x": 0.9, "y": 0.5, "inPenalty": True, "assist": True}
    assert deep.classify_route({**base, "assistX": 0.95, "assistY": 0.3}) == "cutback"
    assert deep.classify_route({**base, "assistX": 0.8, "assistY": 0.1}) == "cross"
    assert deep.classify_route({**base, "assistX": 0.6, "assistY": 0.5}) == "through"
    assert deep.classify_route({**base, "assistX": 0.8, "assistY": 0.5}) == "combo"


def test_route_table_returns_coverage():
    shots = [{"x": 0.9, "y": 0.5, "inPenalty": True, "assist": False, "result": 1}] * 30 + \
            [{"x": 0.9, "y": 0.5, "inPenalty": True, "result": 1}] * 10
    t, cov = deep.route_table(shots, DEFAULT_COEF, False, 10)
    assert math.isclose(cov, 0.75)
    assert math.isclose(t.loc["solo", "슈팅 비중(%)"], 100)
    assert deep.route_table([], DEFAULT_COEF, False, 10)[0].empty


def test_side_flow():
    f = deep.side_flow([10, 80], [12], "승")
    assert f["first"] is True and f["led"] and not f["trailed"]
    assert f["late_for"] == 1 and f["quick_concede"] == 1
    assert deep.side_flow([], [], "무")["first"] is None


def test_flows_from_details_skips_forfeit():
    def side(ouid, end=0):
        return {"ouid": ouid, "shootDetail": [], "matchDetail": {"matchResult": "승", "matchEndType": end}}
    good = {"matchInfo": [side("a"), side("b")]}
    forfeit = {"matchInfo": [side("a", 1), side("b")]}
    assert len(deep.flows_from_details([good, forfeit], "a")) == 1


def test_find_rules_detects_strong_pattern():
    rng = np.random.default_rng(0)
    n = 80
    poss = rng.uniform(30, 70, n)
    win = np.where(poss >= 50, rng.uniform(size=n) < 0.85, rng.uniform(size=n) < 0.15) * 100.0
    me = pd.DataFrame({"win": win, "poss": poss, "shots": rng.integers(5, 15, n).astype(float)})
    rules, n_all = deep.find_rules(me)
    assert n_all == n and rules and "점유율" in rules[0]["조건"]
    assert deep.find_rules(me.head(10)) == ([], 10)


def test_segment_diff():
    assert deep.segment_diff([1] * 10, [5] * 10)[1] == "확실히 좋아짐"
    assert deep.segment_diff([5] * 10, [1] * 10)[1] == "확실히 나빠짐"
    assert deep.segment_diff([5] * 10, [1] * 10, higher_better=False)[1] == "확실히 좋아짐"
    assert deep.segment_diff([1, 2], [3, 4])[1] == "판단 보류"


def test_pos_group():
    assert [deep.pos_group(p) for p in (0, 5, 14, 25)] == ["GK", "DF", "MF", "FW"]


def test_offsides_reads_real_field_name():
    from common import offsides, side_row
    assert offsides({"offsideCount": 3}) == 3
    assert offsides({"OffsideCount": 2}) == 2   # 예전 이름도 계속 읽음
    assert offsides({}) == 0
    assert side_row({"matchDetail": {"offsideCount": 4}}, {})["offside"] == 4
