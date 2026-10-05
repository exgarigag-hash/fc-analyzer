-- FC 전력분석실 데이터베이스 (Supabase > SQL Editor에 통째로 붙여넣고 Run)

-- 패치 달력: 공지 보고 한 줄씩 추가 (Table Editor에서 직접 입력 가능)
create table if not exists patches (
  id serial primary key,
  start_date date not null unique,
  name text not null,
  note text
);
insert into patches(start_date, name, note)
values ('2026-01-01', '기준 구간', '첫 패치를 입력하기 전까지 쓰는 기본 구간')
on conflict do nothing;

-- 유저 (넥슨 약관: 수집 데이터는 30일 이내 갱신 → refreshed_at으로 관리)
create table if not exists users (
  ouid text primary key,
  nickname text,
  max_division int,
  created_at timestamptz default now(),
  refreshed_at timestamptz
);

create table if not exists matches (
  match_id text primary key,
  match_type int,
  match_date timestamptz,
  collected_at timestamptz default now()
);

create table if not exists match_sides (
  match_id text references matches on delete cascade,
  ouid text,
  nickname text, result text, gf int, ga int, shots int, sot int,
  shots_box int, shots_head int, pass_try int, pass_succ int,
  through_try int, through_succ int, long_try int,
  tackle_try int, tackle_succ int, block_try int, block_succ int,
  possession real, fouls int, offside int, rating real, controller text,
  primary key (match_id, ouid)
);
create index if not exists match_sides_ouid on match_sides(ouid);

create table if not exists shots (
  id bigserial primary key,
  match_id text references matches on delete cascade,
  ouid text,
  x real, y real, type int, result int, in_penalty boolean, goal_time bigint,
  dist real, angle real, header int
);
create index if not exists shots_match on shots(match_id);
-- 어시스트 위치 (공격·실점 루트 분석용)
alter table shots add column if not exists assisted boolean;
alter table shots add column if not exists assist_x real;
alter table shots add column if not exists assist_y real;

-- 선수별 기록 (선수 기여 분석, 랭커 비교, 팀 변경 비교용)
create table if not exists player_stats (
  match_id text references matches on delete cascade,
  ouid text,
  sp_id bigint, sp_position int, sp_grade int,
  shoot int, effective_shoot int, goal int, assist int,
  pass_try int, pass_success int, dribble_try int, dribble_success int, dribble real,
  ball_possesion_try int, ball_possesion_success int, aerial_try int, aerial_success int,
  tackle_try int, tackle int, block_try int, block int, intercept int, defending int,
  yellow_cards int, red_cards int, sp_rating real,
  primary key (match_id, ouid, sp_id)
);
create index if not exists player_stats_sp on player_stats(sp_id);
create index if not exists player_stats_ouid on player_stats(ouid);

create table if not exists crawl_queue (
  ouid text primary key,
  added_at timestamptz default now(),
  done_at timestamptz
);

-- xG 모델 기록 (수집기가 데이터로 재학습해서 추가)
create table if not exists xg_models (
  id serial primary key,
  created_at timestamptz default now(),
  patch_id int,
  n_shots int,
  coef jsonb not null,
  flip boolean
);
insert into xg_models(n_shots, coef, flip)
select 0, '{"b0": -0.5, "dist": -0.11, "angle": 1.3, "header": -0.8}', null
where not exists (select 1 from xg_models);

-- 30일이 지나 지우기 전에 남겨두는 익명 집계 (패치별 추세 분석용, 유저 식별 정보 없음)
create table if not exists patch_tier_agg (
  patch_id int, match_type int, division int, day date,
  n_sides int, gf int, ga int, xg real, xga real,
  primary key (patch_id, match_type, division, day)
);

-- 등급 기록: 경기 종류별 역대 최고 등급과 달성일 (갱신할 때마다 쌓임)
alter table users add column if not exists max_division_date timestamptz;
create table if not exists user_divisions (
  ouid text references users on delete cascade,
  match_type int,
  division int,
  achieved_at timestamptz,
  seen_at timestamptz default now(),
  primary key (ouid, match_type, division)
);

-- ---------------------------------------------------------------- 뷰
create or replace view match_patch as
select m.*,
  (select p.id from patches p
    where p.start_date <= (m.match_date at time zone 'Asia/Seoul')::date
    order by p.start_date desc limit 1) as patch_id
from matches m;

create or replace view shot_xg as
select s.match_id, s.ouid, s.in_penalty, s.result,
  1 / (1 + exp(-((c.coef->>'b0')::float8 + (c.coef->>'dist')::float8 * s.dist
               + (c.coef->>'angle')::float8 * s.angle + (c.coef->>'header')::float8 * s.header))) as xg
from shots s
cross join (select coef from xg_models order by id desc limit 1) c;

-- 추정 현재 등급: 매칭은 비슷한 실력끼리 잡히므로, 최근 상대들의 등급 중앙값으로 추정.
-- 최고 등급을 90일 안에 찍은 상대(= 지금 실력과 가까운 상대)가 5명 이상이면 그 상대들만 사용.
create or replace view user_est_division as
with opp as (
  select ms.ouid, m.match_type, u2.max_division as d,
         coalesce(u2.max_division_date > now() - interval '90 days', false) as recent
  from match_sides ms
  join matches m using (match_id)
  join match_sides o on o.match_id = ms.match_id and o.ouid <> ms.ouid
  join users u2 on u2.ouid = o.ouid
  where u2.max_division is not null
)
select ouid, match_type,
  coalesce(
    case when count(*) filter (where recent) >= 5
         then percentile_disc(0.5) within group (order by d) filter (where recent) end,
    percentile_disc(0.5) within group (order by d)) as est_division,
  count(*) as n_opp
from opp
group by ouid, match_type;

-- division = 비교에 쓰는 등급
--   1) 본인이 최고 등급을 90일 안에 찍었으면 그 등급 (지금 실력과 거의 같음)
--   2) 아니면 상대 5명 이상 기준 추정 현재 등급
--   3) 둘 다 없으면 역대 최고 등급
create or replace view side_summary as
select ms.match_id, ms.ouid, mp.match_type, mp.patch_id, mp.match_date, mp.collected_at,
  case when u.max_division_date > now() - interval '90 days' then u.max_division
       when e.n_opp >= 5 then e.est_division
       else u.max_division end as division,
  ms.gf, ms.ga,
  coalesce((select sum(x.xg) from shot_xg x where x.match_id = ms.match_id and x.ouid = ms.ouid), 0) as xg,
  coalesce((select sum(x.xg) from shot_xg x where x.match_id = ms.match_id and x.ouid <> ms.ouid), 0) as xga,
  u.max_division, e.est_division,
  ms.shots, ms.sot, ms.shots_box, ms.pass_try, ms.pass_succ, ms.through_try, ms.through_succ,
  ms.tackle_try, ms.tackle_succ, ms.block_try, ms.block_succ, ms.possession, ms.result
from match_sides ms
join match_patch mp using (match_id)
left join users u on u.ouid = ms.ouid
left join user_est_division e on e.ouid = ms.ouid and e.match_type = mp.match_type;

-- 등급별 평균 지표표
create or replace view tier_metrics as
select match_type, division,
  count(distinct ouid) as users, count(*) as sides,
  avg(xg) as xg, avg(xga) as xga, avg(gf) as gf, avg(ga) as ga,
  avg((result = '승')::int) * 100 as win_rate,
  avg(shots) as shots,
  sum(sot)::float8 / nullif(sum(shots), 0) * 100 as sot_rate,
  sum(shots_box)::float8 / nullif(sum(shots), 0) * 100 as box_share,
  sum(pass_succ)::float8 / nullif(sum(pass_try), 0) * 100 as pass_rate,
  sum(through_succ)::float8 / nullif(sum(through_try), 0) * 100 as through_rate,
  sum(tackle_succ)::float8 / nullif(sum(tackle_try), 0) * 100 as tackle_rate,
  sum(block_succ)::float8 / nullif(sum(block_try), 0) * 100 as block_rate,
  avg(possession) as possession
from side_summary
where division is not null
group by match_type, division;

-- ---------------------------------------------------------------- 보안 (Supabase RLS)
-- 앱과 수집기는 DATABASE_URL로 직접 접속해서 아래 설정의 영향을 받지 않아요.
-- 대신 Supabase의 공개 API 키로는 아무도 표를 읽거나 쓰지 못하게 막아요.
alter table patches enable row level security;
alter table users enable row level security;
alter table matches enable row level security;
alter table match_sides enable row level security;
alter table shots enable row level security;
alter table crawl_queue enable row level security;
alter table xg_models enable row level security;
alter table patch_tier_agg enable row level security;
alter table user_divisions enable row level security;
revoke all on match_patch, shot_xg, user_est_division, side_summary, tier_metrics from anon, authenticated;
