"""
넥슨 API 응답 확인용 스크립트 (GitHub Actions의 inspect 워크플로에서 수동 실행)
- 경기 상세 몇 개를 받아서 실제로 어떤 필드가 오는지 정리
- 슈팅 종류(type) 코드별 골 확률·위치 → 어떤 번호가 헤딩·페널티·프리킥인지 추정
- 시즌 메타데이터, 거래 기록 API가 되는지 확인
유저 식별 정보(ouid, 닉네임)는 출력하지 않음 (로그가 남기 때문)
환경변수: NEXON_API_KEY (필수), DATABASE_URL 또는 SEED_NICKNAMES (유저 찾기용),
         SAMPLE_MATCHES (기본 10), MATCH_TYPE (기본 50)
"""
import json
import os
from collections import Counter, defaultdict

import requests

from common import BASE, ApiError, Nexon

MASK_KEYS = {"ouid", "nickname"}


def mask(obj):
    """ouid·닉네임 값을 가림 (구조는 그대로)."""
    if isinstance(obj, dict):
        return {k: ("***" if k in MASK_KEYS else mask(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [mask(v) for v in obj]
    return obj


def field_summary(objs):
    """여러 dict에 나온 키를 모아서 {키: (나온 횟수, 타입들, 예시 값)}."""
    out = {}
    for o in objs:
        for k, v in (o or {}).items():
            n, types, ex = out.get(k, (0, set(), None))
            types.add(type(v).__name__)
            if ex is None and not isinstance(v, (dict, list)):
                ex = v
            out[k] = (n + 1, types, ex)
    return out


def sections(details):
    """경기 상세들을 부분별(matchInfo, matchDetail, shoot ...)로 나눠 모음."""
    sec = defaultdict(list)
    for d in details:
        sec["(경기)"].append({k: v for k, v in d.items() if k != "matchInfo"})
        for side in d.get("matchInfo") or []:
            sec["matchInfo"].append({k: v for k, v in side.items()
                                     if not isinstance(v, (dict, list))})
            for part in ("matchDetail", "shoot", "pass", "defence"):
                sec[part].append(side.get(part) or {})
            sec["shootDetail[]"] += side.get("shootDetail") or []
            for p in side.get("player") or []:
                sec["player[]"].append({k: v for k, v in p.items() if k != "status"})
                sec["player[].status"].append(p.get("status") or {})
    return sec


def shot_type_table(shots):
    """슈팅 종류 코드별 개수, 골 확률, 평균 위치, 박스 안 비율."""
    g = defaultdict(list)
    for s in shots:
        g[s.get("type")].append(s)
    rows = []
    for t, ss in sorted(g.items(), key=lambda kv: (kv[0] is None, kv[0])):
        n = len(ss)
        xs = [float(s.get("x") or 0) for s in ss]
        ys = [float(s.get("y") or 0) for s in ss]
        rows.append({"type": t, "n": n,
                     "goal%": round(100 * sum(s.get("result") == 3 for s in ss) / n, 1),
                     "avg_x": round(sum(xs) / n, 3), "avg_y": round(sum(ys) / n, 3),
                     "box%": round(100 * sum(bool(s.get("inPenalty")) for s in ss) / n, 1)})
    return rows


def find_ouids(api, mtype, want=3):
    url = os.environ.get("DATABASE_URL")
    if url:
        try:
            from common import db_connect
            with db_connect(url) as c:
                cur = c.cursor()
                cur.execute("select ouid from users order by last_match_at desc nulls last limit %s", (want,))
                rows = [r[0] for r in cur.fetchall()]
                if rows:
                    return rows
        except Exception as e:
            print("DB에서 유저를 못 찾음:", type(e).__name__)
    out = []
    for nick in filter(None, (n.strip() for n in os.environ.get("SEED_NICKNAMES", "").split(","))):
        try:
            out.append(api.ouid(nick))
        except ApiError:
            pass
        if len(out) >= want:
            break
    return out


def md_table(rows):
    if not rows:
        return "(없음)\n"
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def main():
    mtype = int(os.environ.get("MATCH_TYPE", 50))
    n_sample = int(os.environ.get("SAMPLE_MATCHES") or 10)
    api = Nexon(os.environ["NEXON_API_KEY"], interval=0.3, max_calls=60)
    report = ["# 넥슨 API 응답 확인\n"]

    ouids = find_ouids(api, mtype)
    if not ouids:
        raise SystemExit("유저를 찾지 못했어요. DATABASE_URL이나 SEED_NICKNAMES를 확인하세요.")

    ids = []
    for o in ouids:
        try:
            ids += [m for m in api.matches(o, mtype, limit=n_sample) if m not in ids]
        except ApiError as e:
            print("경기 목록 실패:", e.status)
        if len(ids) >= n_sample:
            break
    details = []
    for mid in ids[:n_sample]:
        try:
            details.append(api.detail(mid))
        except ApiError as e:
            print("경기 상세 실패:", e.status)
    report.append(f"경기 상세 {len(details)}개 확인 (경기 종류 {mtype})\n")

    # 1) 필드 목록
    report.append("## 1. 실제로 오는 필드\n")
    for name, objs in sections(details).items():
        rows = [{"필드": k, "나온 수": n, "타입": "/".join(sorted(t)),
                 "예시": "***" if k in MASK_KEYS else json.dumps(ex, ensure_ascii=False)[:40]}
                for k, (n, t, ex) in sorted(field_summary(objs).items())]
        report.append(f"### {name} ({len(objs)}개)\n" + md_table(rows))

    # 2) 슈팅 종류 코드
    shots = [s for d in details for side in d.get("matchInfo") or [] for s in side.get("shootDetail") or []]
    report.append("## 2. 슈팅 종류(type) 코드별 통계\n")
    report.append("골 확률이 아주 높고 위치가 한 점(x≈0.89, y≈0.5 또는 뒤집힌 좌표)에 몰린 번호가 페널티킥, "
                  "박스 밖에서 골 확률이 낮은 번호가 프리킥일 가능성이 커요. 지금 코드는 3번을 헤딩으로 가정해요.\n\n")
    report.append(md_table(shot_type_table(shots)))
    pos = Counter(p.get("spPosition") for d in details for side in d.get("matchInfo") or []
                  for p in side.get("player") or [])
    report.append("\n선수 포지션 번호 분포: " + ", ".join(f"{k}:{v}" for k, v in sorted(pos.items(), key=lambda kv: (kv[0] is None, kv[0]))) + "\n")
    report.append("경기 당시 등급(division) 값이 있는 비율: "
                  f"{sum(1 for d in details for s in d.get('matchInfo') or [] if s.get('division'))}"
                  f" / {sum(len(d.get('matchInfo') or []) for d in details)}\n")

    # 2-1) DB에 쌓인 슈팅 전체로 같은 표 (넥슨 호출 없음, 표본이 훨씬 큼)
    url = os.environ.get("DATABASE_URL")
    if url:
        try:
            from common import db_connect
            with db_connect(url) as c:
                cur = c.cursor()
                cur.execute("""
                  select type, count(*), round(avg((result = 3)::int) * 100, 1), round(avg(x)::numeric, 3),
                         round(avg(y)::numeric, 3), round(avg(in_penalty::int) * 100, 1),
                         round(stddev(x)::numeric, 3), round(stddev(y)::numeric, 3)
                  from shots group by type order by type""")
                rows = [dict(zip(["type", "n", "goal%", "avg_x", "avg_y", "box%", "sd_x", "sd_y"], r))
                        for r in cur.fetchall()]
            report.append("\n### DB에 쌓인 슈팅 전체 기준\n"
                          "위치가 한 점에 몰리면(sd_x, sd_y가 아주 작으면) 페널티킥이에요.\n\n" + md_table(rows))
        except Exception as e:
            report.append(f"\nDB 슈팅 통계 실패: {type(e).__name__}\n")

    # 3) 시즌 메타데이터
    report.append("\n## 3. 시즌 메타데이터 (seasonid)\n")
    try:
        r = requests.get(f"{BASE}/static/fconline/meta/seasonid.json", timeout=30)
        r.raise_for_status()
        seasons = r.json()
        sids = {s.get("seasonId") for s in seasons}
        sp_ids = [p.get("spId") for d in details for side in d.get("matchInfo") or []
                  for p in side.get("player") or [] if p.get("spId")]
        hit = sum(1 for sp in sp_ids if int(sp) // 1000000 in sids)
        report.append(f"시즌 {len(seasons)}개. 예시: {json.dumps(seasons[:3], ensure_ascii=False)[:300]}\n\n"
                      f"선수 카드 ID 앞 3자리가 시즌 ID와 맞는 비율: {hit} / {len(sp_ids)}\n")
    except Exception as e:
        report.append(f"불러오기 실패: {type(e).__name__} {str(e)[:150]}\n")

    # 4) 거래 기록 API
    report.append("\n## 4. 거래 기록 API (/user/trade)\n")
    for tt in ("buy", "sell"):
        try:
            res = api.get("/fconline/v1/user/trade",
                          {"ouid": ouids[0], "tradetype": tt, "offset": 0, "limit": 5})
            keys = sorted({k for row in (res if isinstance(res, list) else []) for k in row})
            report.append(f"- {tt}: {len(res) if isinstance(res, list) else '목록 아님'}건, 필드 {keys}\n")
        except ApiError as e:
            report.append(f"- {tt}: 실패 ({e.status}) {str(e)[:150]}\n")

    report.append(f"\n총 API 호출: {api.calls}\n")
    text = "".join(report)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(text)
    if details:  # 원본 1개 (식별 정보 가림) → Actions 아티팩트로 내려받기
        with open("sample_match.json", "w", encoding="utf-8") as f:
            json.dump(mask(details[0]), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
