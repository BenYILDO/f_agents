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

-- ── Gecelik model önbelleği (Faz H) ──────────────────────────────────────
-- Ağır hesaplar (kalibre yukarı-olasılığı, DSR) gecelik bir GitHub Actions
-- işiyle burada üretilir; saatlik iş bunu okuyup güven skoruna katar → saatlik
-- hafif kalır. Her ticker için tek satır (upsert).
create table if not exists model_cache (
    ticker       text        primary key,
    p_up         numeric,
    brier        numeric,
    auc          numeric,
    dsr          numeric,
    horizon      int,
    n_samples    int,
    updated_at   timestamptz not null default now()
);

-- F0.2 — zengin model kanıtı (sigmoid kalibrasyon + BSS + kalite kapısı).
-- Mevcut kurulumlarda tabloya ek kolonları idempotent olarak ekler; gecelik iş
-- bunları yazar, snapshot _meta'sına "model neden kullanıldı/kullanılmadı" izi taşınır.
alter table model_cache add column if not exists brier_raw          numeric;
alter table model_cache add column if not exists brier_calibrated   numeric;
alter table model_cache add column if not exists brier_skill_score  numeric;
alter table model_cache add column if not exists recent_skill       numeric;
alter table model_cache add column if not exists n_calibration      int;
alter table model_cache add column if not exists n_test             int;
alter table model_cache add column if not exists quality_passed     boolean;
alter table model_cache add column if not exists rejection_reasons  jsonb;
alter table model_cache add column if not exists model_version      text;
alter table model_cache add column if not exists trained_until      text;

-- ── Havuz modeli deposu (S2) ─────────────────────────────────────────────
-- Gecelik iş, tüm evreni tek panelde eğiten pooled modeli burada saklar:
-- artefakt (pickle+zlib+base64) + kalite karnesi. Her eğitim YENİ satırdır
-- (tarihçe birikir); tüketiciler en son quality_passed satırı yükler ve gün
-- içinde yalnız tahmin yapar — model her gün baştan eğitilmez.
create table if not exists pooled_models (
    id                 bigint generated always as identity primary key,
    model_version      text        not null,
    trained_at         timestamptz not null default now(),
    trained_until      text,
    horizon            int,
    threshold          numeric,
    universe_size      int,
    universe           jsonb       not null default '[]'::jsonb,
    n_samples          int,
    n_test             int,
    auc                numeric,
    brier_raw          numeric,
    brier_calibrated   numeric,
    brier_skill_score  numeric,
    quality_passed     boolean     not null default false,
    rejection_reasons  jsonb       not null default '[]'::jsonb,
    feature_importance jsonb       not null default '{}'::jsonb,
    artifact           text,
    meta               jsonb       not null default '{}'::jsonb
);
create index if not exists pooled_models_version_ts_idx
    on pooled_models (model_version, trained_at desc);

-- Challenger tahminleri: pooled modelin ticker başına güncel kazanma olasılığı,
-- mevcut per-ticker şampiyonun YANINA yazılır (davranışı değiştirmez; karne
-- biriktirir — champion/challenger disiplini, plan §S2/S4).
alter table model_cache add column if not exists p_up_pooled     numeric;
alter table model_cache add column if not exists pooled_version  text;
alter table model_cache add column if not exists pooled_quality  boolean;

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
alter table model_cache        enable row level security;
alter table pooled_models      enable row level security;

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
