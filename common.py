"""앱과 수집기가 같이 쓰는 코드: 넥슨 API, 경기 파싱, xG 모델, DB 저장."""
import math
import time

import numpy as np
import requests

BASE = "https://open.api.nexon.com"


# ================================================================ 넥슨 API
class ApiError(Exception):
    def __init__(self, msg, status=None):
        super().__init__(msg)
        self.status = status


class BudgetExceeded(Exception):
    pass


class Nexon:
    """호출 간격과 호출 예산(수집기용)을 지키는 API 클라이언트."""

    def __init__(self, key, interval=0.0, max_calls=None):
        self.key = (key or "").strip()  # 복사할 때 딸려 온 줄바꿈·공백 제거
        self.interval, self.max_calls = interval, max_calls
        self.calls, self._last = 0, 0.0

    def get(self, path, params):
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise BudgetExceeded()
        for attempt in range(5):
            wait = self.interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            self.calls += 1
            r = requests.get(BASE + path, params=params,
                             headers={"x-nxopen-api-key": self.key}, timeout=15)
            if r.status_code == 429:
                time.sleep(1.0 * (attempt + 1))
                continue
            if r.status_code != 200:
                try:
                    msg = r.json().get("error", {}).get("message", r.text)
                except Exception:
                    msg = r.text
                raise ApiError(f"{r.status_code}: {msg}", r.status_code)
            return r.json()
        raise ApiError("요청 한도 초과", 429)

    def ouid(self, nick):
        return self.get("/fconline/v1/id", {"nickname": nick})["ouid"]

    def basic(self, ouid):
        return self.get("/fconline/v1/user/basic", {"ouid": ouid})

    def maxdivision(self, ouid):
        return self.get("/fconline/v1/user/maxdivision", {"ouid": ouid})

    def matches(self, ouid, mtype, limit=20, offset=0):
        return self.get("/fconline/v1/user/match",
                        {"ouid": ouid, "matchtype": mtype, "offset": offset, "limit": limit})

    _recent_params = None  # 넥슨이 받아주는 파라미터 조합 (처음 성공한 걸 기억)

    def recent_matches(self, mtype, limit=100, offset=0):
        """매치 종류별 최근 경기 목록 (특정 유저 없이 전체에서 가져옴).
        넥슨 문서와 실제 허용값이 다를 수 있어서 몇 가지 조합을 차례로 시도."""
        tries = [self._recent_params] if self._recent_params else [
            {"orderby": "desc", "limit": limit}, {"limit": limit},
            {"orderby": "desc", "limit": 20}, {"limit": 20}]
        last = None
        for extra in tries:
            try:
                ids = self.get("/fconline/v1/match", {"matchtype": mtype, "offset": offset, **extra})
                self._recent_params = extra
                return ids
            except ApiError as e:
                last = e
                if e.status != 400:
                    raise
        raise last

    def detail(self, mid):
        return self.get("/fconline/v1/match-detail", {"matchid": mid})


# ================================================================ 경기 파싱
def goals(side):
    sh = side.get("shoot") or {}
    return sh.get("goalTotalDisplay", sh.get("goalTotal", 0)) or 0


def side_row(s, o):
    md, sh = s.get("matchDetail") or {}, s.get("shoot") or {}
    ps, df = s.get("pass") or {}, s.get("defence") or {}
    return {
        "ouid": s.get("ouid"), "nickname": s.get("nickname"),
        "result": md.get("matchResult"), "gf": goals(s), "ga": goals(o),
        "shots": sh.get("shootTotal", 0), "sot": sh.get("effectiveShootTotal", 0),
        "shots_box": sh.get("shootInPenalty", 0), "shots_head": sh.get("shootHeading", 0),
        "pass_try": ps.get("passTry", 0), "pass_succ": ps.get("passSuccess", 0),
        "through_try": ps.get("throughPassTry", 0), "through_succ": ps.get("throughPassSuccess", 0),
        "long_try": ps.get("longPassTry", 0),
        "tackle_try": df.get("tackleTry", 0), "tackle_succ": df.get("tackleSuccess", 0),
        "block_try": df.get("blockTry", 0), "block_succ": df.get("blockSuccess", 0),
        "possession": md.get("possession"), "fouls": md.get("foul", 0),
        "offside": md.get("OffsideCount", 0), "rating": md.get("averageRating"),
        "controller": md.get("controller"),
    }


def valid_match(d):
    info = d.get("matchInfo") or []
    return len(info) == 2 and all((x.get("matchDetail") or {}).get("matchEndType", 0) == 0
                                  for x in info)


# ================================================================ xG (기대 득점)
PITCH_L, PITCH_W, GOAL_W = 105.0, 68.0, 7.32
HEADER_TYPE = 3  # shootDetail.type 중 헤딩 코드 (넥슨 문서 기준, 다르면 여기만 수정)
DEFAULT_COEF = {"b0": -0.5, "dist": -0.11, "angle": 1.3, "header": -0.8}


def shot_features(s, flip=False):
    """슈팅 좌표 → (골대까지 거리 m, 골대가 보이는 각도 rad, 헤딩 여부)."""
    x, y = float(s.get("x") or 0), float(s.get("y") or 0)
    if flip:
        x = 1 - x
    dx, dy = (1 - x) * PITCH_L, (y - 0.5) * PITCH_W
    dist = math.hypot(dx, dy)
    ang = math.atan2(GOAL_W * dx, dx * dx + dy * dy - (GOAL_W / 2) ** 2)
    if ang < 0:
        ang += math.pi
    return dist, ang, 1 if s.get("type") == HEADER_TYPE else 0


def detect_flip(shots):
    """좌표 방향 자동 판별: 골이 x=0 쪽에 몰려 있으면 좌표를 뒤집어야 함."""
    g = [float(s.get("x") or 0) for s in shots if s.get("result") == 3]
    xs = g if len(g) >= 5 else [float(s.get("x") or 0) for s in shots]
    return bool(xs) and float(np.mean(xs)) < 0.5


def xg_values(shots, coef, flip=False):
    if not shots:
        return np.zeros(0)
    f = np.array([shot_features(s, flip) for s in shots])
    z = coef["b0"] + coef["dist"] * f[:, 0] + coef["angle"] * f[:, 1] + coef["header"] * f[:, 2]
    return 1 / (1 + np.exp(-z))


def calibrate(coef, shots, flip=False):
    """기울기는 두고 절편만 조정해서 '총 xG = 총 골'이 되게 맞춤 (데이터가 적을 때용)."""
    goals_n = sum(1 for s in shots if s.get("result") == 3)
    if len(shots) < 50 or goals_n == 0:
        return dict(coef)
    c = dict(coef)
    for _ in range(30):
        p = xg_values(shots, c, flip)
        f, fp = p.sum() - goals_n, (p * (1 - p)).sum()
        if fp <= 0 or abs(f) < 1e-3:
            break
        c["b0"] -= f / fp
    c["b0"] = float(np.clip(c["b0"], coef["b0"] - 2, coef["b0"] + 2))
    return c


def fit_logit(X, y, l2=1.0, iters=30):
    """로지스틱 회귀 (뉴턴법). X에 절편 열 포함."""
    w = np.zeros(X.shape[1])
    reg = np.eye(X.shape[1]) * l2
    reg[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + reg @ w
        H = X.T @ (X * (p * (1 - p))[:, None]) + reg
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return w


# ================================================================ DB (Supabase Postgres)
def db_connect(url):
    import psycopg
    return psycopg.connect(url.strip().strip("\"'"), autocommit=True, prepare_threshold=None, connect_timeout=8)


SIDE_COLS = ["ouid", "nickname", "result", "gf", "ga", "shots", "sot", "shots_box",
             "shots_head", "pass_try", "pass_succ", "through_try", "through_succ", "long_try",
             "tackle_try", "tackle_succ", "block_try", "block_succ", "possession", "fouls",
             "offside", "rating", "controller"]


def latest_model(cur):
    cur.execute("select coef, flip, n_shots from xg_models order by id desc limit 1")
    row = cur.fetchone()
    return (row[0], row[1], row[2]) if row else (DEFAULT_COEF, None, 0)


def save_match(cur, d, flip):
    """경기 1개 저장. 새로 저장했으면 True. 양쪽 유저 ouid는 users에 등록."""
    if not valid_match(d):
        return False
    info = d["matchInfo"]
    cur.execute("insert into matches(match_id, match_type, match_date) values (%s,%s,%s) "
                "on conflict do nothing", (d["matchId"], d.get("matchType"), d.get("matchDate")))
    if cur.rowcount == 0:
        return False
    rows = [side_row(info[0], info[1]), side_row(info[1], info[0])]
    cur.executemany(
        f"insert into match_sides(match_id,{','.join(SIDE_COLS)}) values "
        f"(%s,{','.join(['%s'] * len(SIDE_COLS))}) on conflict do nothing",
        [(d["matchId"], *[r[c] for c in SIDE_COLS]) for r in rows])
    shot_rows = []
    for side in info:
        for s in side.get("shootDetail") or []:
            dist, ang, head = shot_features(s, flip)
            shot_rows.append((d["matchId"], side.get("ouid"), s.get("x"), s.get("y"),
                              s.get("type"), s.get("result"), bool(s.get("inPenalty")),
                              s.get("goalTime"), dist, ang, head))
    if shot_rows:
        cur.executemany("insert into shots(match_id,ouid,x,y,type,result,in_penalty,goal_time,"
                        "dist,angle,header) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", shot_rows)
    cur.executemany("insert into users(ouid, nickname) values (%s,%s) "
                    "on conflict (ouid) do update set nickname = excluded.nickname",
                    [(r["ouid"], r["nickname"]) for r in rows])
    return True
