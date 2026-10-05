"""
복합분석 모델 학습 (GitHub Actions가 매일 새벽 실행)
- 경기마다 과정 지표(슈팅 전개, 빌드업, 수비 등) + 상황(내 등급, 상대 등급)으로 승리 확률을 예측
- 최근 20% 경기로 정확도를 검증하고, 등급별 평균 기여도(기준선)와 함께 DB에 저장
환경변수: DATABASE_URL, MIN_MATCHES(기본 5000), MATCH_TYPE(기본 50)
"""
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import FEATURES, db_connect, match_features

MIN_MATCHES = int(os.environ.get("MIN_MATCHES") or 5000)
MTYPE = int(os.environ.get("MATCH_TYPE", 50))
conn = db_connect(os.environ["DATABASE_URL"])
cur = conn.cursor()


def log(*a):
    print(*a, flush=True)


cur.execute("select count(*) from matches where match_type = %s", (MTYPE,))
n_matches = cur.fetchone()[0]
log("공식경기 수:", n_matches, "/ 학습 기준:", MIN_MATCHES)
if n_matches < MIN_MATCHES:
    log("데이터가 기준보다 적어서 학습을 건너뛰어요.")
    raise SystemExit(0)

cur.execute("""
  select ms.*, m.match_date, ss.division
  from match_sides ms
  join matches m using (match_id)
  left join side_summary ss on ss.match_id = ms.match_id and ss.ouid = ms.ouid
  where m.match_type = %s""", (MTYPE,))
cols = [c.name for c in cur.description]
df = pd.DataFrame(cur.fetchall(), columns=cols)
log("경기 측면(sides) 수:", len(df))

rows = []
for mid, g in df.groupby("match_id"):
    if len(g) != 2:
        continue
    a, b = g.iloc[0].to_dict(), g.iloc[1].to_dict()
    for me, op in ((a, b), (b, a)):
        f = match_features(me, op, me.get("division"), op.get("division"))
        f.update(_y=int(me.get("result") == "승"), _date=me.get("match_date"),
                 _div=me.get("division"))
        rows.append(f)
data = pd.DataFrame(rows).sort_values("_date").reset_index(drop=True)
X, y = data[FEATURES].astype(float), data["_y"].values

# 시간순으로 최근 20%는 검증용
cut = int(len(data) * 0.8)
params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 50,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
          "verbose": -1, "seed": 0}
dtr, dva = lgb.Dataset(X.iloc[:cut], y[:cut]), lgb.Dataset(X.iloc[cut:], y[cut:])
m = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[dva],
              callbacks=[lgb.early_stopping(50, verbose=False)])
best = max(m.best_iteration, 50)
p = m.predict(X.iloc[cut:], num_iteration=best)


def auc(y_true, score):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y_true == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


yv = y[cut:]
metrics = {
    "auc": auc(yv, p),
    "logloss": float(-np.mean(yv * np.log(np.clip(p, 1e-6, 1)) + (1 - yv) * np.log(np.clip(1 - p, 1e-6, 1)))),
    "accuracy": float(np.mean((p >= 0.5) == yv)),
    "base_accuracy": float(max(yv.mean(), 1 - yv.mean())),
    "n_train": int(cut), "n_valid": int(len(yv)), "best_iter": int(best), "n_matches": int(n_matches),
}
log("검증 결과:", {k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()})

# 전체 데이터로 다시 학습 → 기여도 기준선 계산
final = lgb.train(params, lgb.Dataset(X, y), num_boost_round=best)
contrib = final.predict(X, pred_contrib=True)[:, :-1]
cdf = pd.DataFrame(contrib, columns=FEATURES)
cdf["_div"] = data["_div"].values
baseline = {"all": {"contrib": cdf[FEATURES].mean().to_dict(), "win": float(y.mean()), "n": int(len(y))}}
for div, g in cdf.groupby("_div"):
    if len(g) >= 200:
        baseline[str(int(div))] = {"contrib": g[FEATURES].mean().to_dict(),
                                   "win": float(data.loc[g.index, "_y"].mean()), "n": int(len(g))}
imp = dict(zip(FEATURES, final.feature_importance("gain").tolist()))
metrics["importance"] = imp

cur.execute("insert into ml_models(n_sides, metrics, baseline, model) values (%s,%s,%s,%s)",
            (len(data), json.dumps(metrics), json.dumps(baseline), final.model_to_string()))
cur.execute("delete from ml_models where id not in (select id from ml_models order by id desc limit 3)")
log("모델 저장 완료. 등급별 기준선:", [k for k in baseline])
