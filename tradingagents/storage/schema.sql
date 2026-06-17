-- ════════════════════════════════════════════════════════════════════════
-- BIST TradingAgents — Supabase (Postgres) şeması
-- ────────────────────────────────────────────────────────────────────────
-- Supabase SQL Editor'de (Dashboard → SQL Editor → New query) BİR KEZ
-- çalıştırın. Idempotent'tir: tekrar çalıştırmak güvenlidir.
--
-- Tablolar:
--   holdings            → portföy (elindeki hisseler: adet, alış fiyatı, tarih)
--   analysis_snapshots  → saatlik analiz geçmişi (güven katmanı + karşılaştırma)
--   ai_runs             → (opsiyonel) YZ analiz çıktıları
--   watchlist           → (opsiyonel) portföyde olmayan ama taranan hisseler
-- ════════════════════════════════════════════════════════════════════════

-- ── Portföy ──────────────────────────────────────────────────────────────
create table if not exists holdings (
    id          bigint generated always as identity primary key,
    ticker      text        not null,
    quantity    numeric     not null check (quantity > 0),
    buy_price   numeric     not null check (buy_price >= 0),
    buy_date    date        not null default current_date,
    note        text        not null default '',
    created_at  timestamptz not null default now()
);
create index if not exists holdings_ticker_idx on holdings (ticker);

-- ── Saatlik analiz geçmişi ───────────────────────────────────────────────
-- scope:  'portfolio' | 'bist30' | 'watchlist'
-- source: 'cron' | 'manual' | 'on_add'
-- status: 'AL' | 'SAT' | 'NÖTR'           (Dip-Al stratejisi güncel durumu)
-- decision: 'GÜÇLÜ AL' | 'AL' | 'TUT' | 'SAT' | 'KAÇIN'  (birleşik karar)
-- agreement: çoklu-yöntem mutabakatı özeti (güven göstergesi)
create table if not exists analysis_snapshots (
    id             bigint generated always as identity primary key,
    ts             timestamptz not null default now(),
    ticker         text        not null,
    scope          text        not null default 'portfolio',
    source         text        not null default 'cron',
    status         text,
    decision       text,
    combined_score numeric,
    tech_score     numeric,
    ratio_score    numeric,
    ratio_verdict  text,
    confidence     text,
    agreement      text,
    close          numeric,
    smi            numeric,
    signals        jsonb       not null default '{}'::jsonb,
    health         jsonb       not null default '{}'::jsonb
);
create index if not exists snap_ticker_ts_idx on analysis_snapshots (ticker, ts desc);
create index if not exists snap_scope_ts_idx  on analysis_snapshots (scope, ts desc);

-- Her (ticker, scope) için en güncel snapshot — UI hızlı okuma için.
create or replace view latest_snapshots as
select distinct on (ticker, scope) *
from analysis_snapshots
order by ticker, scope, ts desc;

-- ── (Opsiyonel) YZ analiz çıktıları ──────────────────────────────────────
create table if not exists ai_runs (
    id         bigint generated always as identity primary key,
    ts         timestamptz not null default now(),
    ticker     text        not null,
    rating     text,
    language   text,
    report_md  text,
    meta       jsonb       not null default '{}'::jsonb
);
create index if not exists ai_runs_ticker_ts_idx on ai_runs (ticker, ts desc);

-- ── (Opsiyonel) İzleme listesi ───────────────────────────────────────────
create table if not exists watchlist (
    id         bigint generated always as identity primary key,
    ticker     text        not null unique,
    note       text        not null default '',
    created_at timestamptz not null default now()
);

-- ════════════════════════════════════════════════════════════════════════
-- Güvenlik (RLS) — tek kullanıcılı kurulum
-- ────────────────────────────────────────────────────────────────────────
-- RLS açıyoruz ve HERKESE AÇIK politika EKLEMİYORUZ. Uygulama (Streamlit) ve
-- zamanlayıcı (GitHub Actions) SUPABASE_SERVICE_KEY ile bağlanır; service_role
-- anahtarı RLS'i bypass eder ve yalnızca sunucu tarafı secret'larda durur,
-- tarayıcıya sızmaz. Böylece anon anahtarıyla (varsa) dışarıdan veri okunamaz.
--
-- İleride Supabase Auth ile çok kullanıcıya geçersen, buraya user_id bazlı
-- politikalar eklersin.
alter table holdings           enable row level security;
alter table analysis_snapshots enable row level security;
alter table ai_runs            enable row level security;
alter table watchlist          enable row level security;

-- ════════════════════════════════════════════════════════════════════════
-- Saklama (retention) — Supabase free tier 500 MB; saatlik satırlar birikir.
-- ────────────────────────────────────────────────────────────────────────
-- 90 günden eski saatlik snapshot'ları günde 1 kayda seyreltir (her gün/
-- ticker/scope için yalnız en güncelini tutar). pg_cron varsa zamanlanabilir;
-- yoksa GitHub Actions retention adımı bu fonksiyonu RPC ile çağırır.
create or replace function prune_old_snapshots(retain_days int default 90)
returns integer
language plpgsql
as $$
declare
    deleted integer;
begin
    with ranked as (
        select id,
               row_number() over (
                   partition by ticker, scope, date_trunc('day', ts)
                   order by ts desc
               ) as rn
        from analysis_snapshots
        where ts < now() - make_interval(days => retain_days)
    )
    delete from analysis_snapshots a
    using ranked r
    where a.id = r.id and r.rn > 1;
    get diagnostics deleted = row_count;
    return deleted;
end;
$$;
