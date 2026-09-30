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

create or replace view side_summary as
select ms.match_id, ms.ouid, mp.match_type, mp.patch_id, mp.match_date, mp.collected_at,
  u.max_division as division, ms.gf, ms.ga,
  coalesce((select sum(x.xg) from shot_xg x where x.match_id = ms.match_id and x.ouid = ms.ouid), 0) as xg,
  coalesce((select sum(x.xg) from shot_xg x where x.match_id = ms.match_id and x.ouid <> ms.ouid), 0) as xga
from match_sides ms
join match_patch mp using (match_id)
left join users u on u.ouid = ms.ouid;
