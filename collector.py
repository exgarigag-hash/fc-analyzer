"""
자동 수집기 (GitHub Actions가 3시간마다 실행)
1) 시드 닉네임 등록  2) 오래된 유저 정보 갱신(30일 규정)  3) 눈덩이 방식 경기 수집
4) xG 모델 재학습    5) 28일 지난 데이터는 익명 집계로 남기고 삭제
환경변수: NEXON_API_KEY, DATABASE_URL, SEED_NICKNAMES(쉼표 구분), MAX_CALLS, CALL_INTERVAL, MATCH_TYPE
"""
import json
import os

import numpy as np

from common import (ApiError, BudgetExceeded, Nexon, db_connect, detect_flip, fit_logit,
                    latest_model, save_match)

MTYPE = int(os.environ.get("MATCH_TYPE", 50))
api = Nexon(os.environ["NEXON_API_KEY"],
            interval=float(os.environ.get("CALL_INTERVAL", 0.3)),
            max_calls=int(os.environ.get("MAX_CALLS", 900)))
conn = db_connect(os.environ["DATABASE_URL"])
cur = conn.cursor()
QUEUE_CAP = 20000


def log(*a):
    print(*a, flush=True)


def register_seeds():
    for nick in filter(None, (n.strip() for n in os.environ.get("SEED_NICKNAMES", "").split(","))):
        try:
            ouid = api.ouid(nick)
        except ApiError as e:
            log("시드 닉네임 실패:", nick, e)
            continue
        cur.execute("insert into users(ouid, nickname) values (%s,%s) on conflict do nothing", (ouid, nick))
        cur.execute("insert into crawl_queue(ouid) values (%s) on conflict do nothing", (ouid,))


def refresh_users(limit=200):
    """새 유저는 등급 조회, 25일 지난 유저는 닉네임·등급 갱신."""
    cur.execute("select ouid from users where refreshed_at is null "
                "or refreshed_at < now() - interval '25 days' "
                "order by refreshed_at nulls first limit %s", (limit,))
    n = 0
    for (ouid,) in cur.fetchall():
        try:
            nick = api.basic(ouid).get("nickname")
            divs = api.maxdivision(ouid)
        except ApiError as e:
            if e.status in (400, 404):  # 없어진 계정 → 삭제
                cur.execute("delete from users where ouid=%s", (ouid,))
                cur.execute("delete from crawl_queue where ouid=%s", (ouid,))
            continue
        div = next((d.get("division") for d in divs if d.get("matchType") == MTYPE), None)
        cur.execute("update users set nickname=%s, max_division=%s, refreshed_at=now() where ouid=%s",
                    (nick, div, ouid))
        n += 1
    log("유저 갱신:", n)


def crawl(buffer):
    cur.execute("select ouid from crawl_queue where done_at is null "
                "or done_at < now() - interval '3 days' order by done_at nulls first limit 200")
    for (ouid,) in cur.fetchall():
        ids = api.matches(ouid, MTYPE, limit=20)
        if ids:
            cur.execute("select match_id from matches where match_id = any(%s)", (ids,))
            have = {r[0] for r in cur.fetchall()}
            for mid in ids:
                if mid not in have and mid not in buffer:
                    try:
                        buffer[mid] = api.detail(mid)
                    except ApiError:
                        pass
        cur.execute("update crawl_queue set done_at=now() where ouid=%s", (ouid,))


def save(buffer):
    coef, flip, n_shots = latest_model(cur)
    if flip is None:
        shots = [s for d in buffer.values() for side in d.get("matchInfo") or []
                 for s in side.get("shootDetail") or []]
        flip = detect_flip(shots)
        cur.execute("update xg_models set flip=%s where id=(select max(id) from xg_models)", (flip,))
        log("좌표 방향 판별: flip =", flip)
    new, ouids = 0, set()
    for d in buffer.values():
        if save_match(cur, d, flip):
            new += 1
            ouids.update(x.get("ouid") for x in d["matchInfo"])
    cur.execute("select count(*) from crawl_queue")
    if cur.fetchone()[0] < QUEUE_CAP:
        cur.executemany("insert into crawl_queue(ouid) values (%s) on conflict do nothing",
                        [(o,) for o in ouids if o])
    log("새 경기 저장:", new)


def retrain():
    cur.execute("select created_at > now() - interval '20 hours', n_shots, flip "
                "from xg_models order by id desc limit 1")
    recent, n_prev, flip = cur.fetchone()
    if recent and n_prev:  # 기본 모델이면 바로 학습
        return
    cur.execute("select dist, angle, header, (result = 3)::int from shots where dist is not null")
    rows = np.array(cur.fetchall(), dtype=float)
    if len(rows) < 2000:
        log("xG 재학습 보류: 슈팅", len(rows), "개 (2000개 필요)")
        return
    X = np.column_stack([np.ones(len(rows)), rows[:, :3]])
    w = fit_logit(X, rows[:, 3])
    coef = {"b0": w[0], "dist": w[1], "angle": w[2], "header": w[3]}
    cur.execute("select max(id) from patches")
    cur.execute("insert into xg_models(patch_id, n_shots, coef, flip) values (%s,%s,%s,%s)",
                (cur.fetchone()[0], len(rows), json.dumps({k: float(v) for k, v in coef.items()}), flip))
    log("xG 재학습 완료:", len(rows), "슈팅", {k: round(float(v), 3) for k, v in coef.items()})


def rollup_and_purge():
    cur.execute("""
      insert into patch_tier_agg(patch_id, match_type, division, day, n_sides, gf, ga, xg, xga)
      select coalesce(patch_id, 0), match_type, coalesce(division, 0),
             (match_date at time zone 'Asia/Seoul')::date,
             count(*), sum(gf), sum(ga), sum(xg), sum(xga)
      from side_summary where collected_at < now() - interval '28 days'
      group by 1, 2, 3, 4
      on conflict (patch_id, match_type, division, day) do update set
        n_sides = patch_tier_agg.n_sides + excluded.n_sides,
        gf = patch_tier_agg.gf + excluded.gf, ga = patch_tier_agg.ga + excluded.ga,
        xg = patch_tier_agg.xg + excluded.xg, xga = patch_tier_agg.xga + excluded.xga""")
    cur.execute("delete from matches where collected_at < now() - interval '28 days'")
    log("오래된 경기 삭제:", cur.rowcount)
    cur.execute("delete from users where coalesce(refreshed_at, created_at) < now() - interval '28 days'")
    log("갱신 안 된 유저 삭제:", cur.rowcount)


buffer = {}
try:
    register_seeds()
    refresh_users(limit=max(20, api.max_calls // 4))
    crawl(buffer)
except BudgetExceeded:
    log("호출 예산 소진:", api.calls)
finally:
    save(buffer)
    retrain()
    rollup_and_purge()
    log("총 API 호출:", api.calls)
