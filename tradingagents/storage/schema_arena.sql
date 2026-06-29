-- ════════════════════════════════════════════════════════════════════════
-- Paper Trading Arena — MVP şeması (plan §Revize Supabase veri modeli)
-- ────────────────────────────────────────────────────────────────────────
-- Bu şema arena KALICILIĞI içindir. Edge kapısı/replay (UI) Supabase GEREKTİRMEZ
-- — yereldir. Bu tablolar canlı sezon, emir/fill/ledger/pozisyon/equity zincirini
-- ve observer tahminlerini saklamak içindir. Edge kapısı geçildikten SONRA bağlanır.
--
-- Tasarım ilkeleri (plan §Tasarım kuralları):
--   - paper_fills gerçekleşmiş işlemlerin DEĞİŞMEZ kaynağıdır.
--   - Para alanları numeric (Python tarafı Decimal); adet integer.
--   - rules_snapshot/strategy_version değişmez; kural değişince yeni sürüm/sezon.
--   - Benzersiz kısıtlar idempotency sağlar: aynı cron tekrar koşunca yeni emir
--     üretmez, mevcut run_id/emri bulur.
--   - Fill + nakit + pozisyon mümkünse tek RPC transaction'ı içinde atomik.
-- ════════════════════════════════════════════════════════════════════════

create table if not exists arena_seasons (
    id               uuid primary key default gen_random_uuid(),
    season_id        text unique not null,            -- ör. '2026-H2-S1'
    name             text not null,
    status           text not null default 'ACTIVE',  -- ACTIVE | CLOSED
    starting_at      timestamptz not null default now(),
    ending_at        timestamptz,
    base_currency    text not null default 'TRY',
    initial_capital  numeric not null default 100000,
    universe_version text,
    execution_config jsonb not null,                  -- komisyon/slippage/buffer (dondurulmuş)
    benchmark_config jsonb,
    created_at       timestamptz not null default now()
);

create table if not exists paper_accounts (
    id               uuid primary key default gen_random_uuid(),
    season_id        text not null references arena_seasons(season_id) on delete cascade,
    name             text not null,
    profile_code     text not null,                   -- conservative|balanced|aggressive|trend|ml_observer
    status           text not null default 'ACTIVE',  -- ACTIVE | OBSERVER
    initial_cash     numeric not null,
    strategy_version text not null,
    rules_snapshot   jsonb not null,                  -- DEĞİŞMEZ profil kuralları
    created_at       timestamptz not null default now(),
    unique (season_id, profile_code)
);

create table if not exists paper_runs (
    id            uuid primary key default gen_random_uuid(),
    season_id     text not null references arena_seasons(season_id) on delete cascade,
    run_type      text not null,                      -- decision | fill | equity
    signal_asof   timestamptz,
    scheduled_session date,
    status        text not null default 'PENDING',    -- PENDING | DONE | ERROR
    started_at    timestamptz,
    completed_at  timestamptz,
    error_summary text,
    -- Idempotency: aynı seans için aynı tip run iki kez üretilmez
    unique (season_id, run_type, scheduled_session)
);

create table if not exists paper_orders (
    id                uuid primary key default gen_random_uuid(),
    run_id            uuid not null references paper_runs(id) on delete cascade,
    account_id        uuid not null references paper_accounts(id) on delete cascade,
    ticker            text not null,
    asset_type        text not null default 'stock',
    side              text not null,                  -- BUY | SELL
    quantity          integer not null,
    order_type        text not null default 'MOO',    -- market-on-open proxy
    status            text not null default 'PENDING', -- PENDING | FILLED | REJECTED
    signal_asof       timestamptz,
    scheduled_session date,
    reason_snapshot   jsonb,                          -- kararın gerekçesi (sürüm dahil)
    strategy_version  text,
    created_at        timestamptz not null default now(),
    -- Aynı sinyal/seans/yön için tek emir (cron tekrarı yeni emir açmaz)
    unique (account_id, ticker, side, scheduled_session, strategy_version)
);

create table if not exists paper_fills (
    id              uuid primary key default gen_random_uuid(),
    order_id        uuid not null references paper_orders(id) on delete cascade,
    account_id      uuid not null references paper_accounts(id) on delete cascade,
    filled_at       timestamptz not null default now(),
    quantity        integer not null,
    reference_price numeric not null,                 -- open proxy
    fill_price      numeric not null,                 -- ref ± slippage
    commission      numeric not null default 0,
    slippage        numeric not null default 0,
    created_at      timestamptz not null default now(),
    unique (order_id)                                 -- bir emir bir kez dolar
);

create table if not exists paper_cash_ledger (
    id            uuid primary key default gen_random_uuid(),
    account_id    uuid not null references paper_accounts(id) on delete cascade,
    fill_id       uuid references paper_fills(id) on delete set null,
    ts            timestamptz not null default now(),
    event_type    text not null,                      -- BUY | SELL | COMMISSION | DIVIDEND | INIT
    amount        numeric not null,                   -- işaretli (giriş +, çıkış -)
    balance_after numeric not null,
    metadata      jsonb
);

create table if not exists paper_positions (
    account_id    uuid not null references paper_accounts(id) on delete cascade,
    ticker        text not null,
    asset_type    text not null default 'stock',
    quantity      integer not null,
    avg_cost      numeric not null,
    realized_pnl  numeric not null default 0,
    updated_at    timestamptz not null default now(),
    primary key (account_id, ticker)
);

create table if not exists paper_equity (
    id               uuid primary key default gen_random_uuid(),
    account_id       uuid not null references paper_accounts(id) on delete cascade,
    asof_ts          timestamptz not null,
    cash             numeric not null,
    unsettled_cash   numeric not null default 0,
    positions_value  numeric not null,
    total_equity     numeric not null,
    benchmark_value  numeric,                          -- XU100 aynı kasayla
    stale_price_count integer not null default 0,      -- bayat fiyatla işaretli MTM
    is_official      boolean not null default false,    -- EOD resmi mi, intraday tahmini mi
    created_at       timestamptz not null default now(),
    unique (account_id, asof_ts)
);

-- Observer karne sözleşmesi (plan F0.6 / teslimat #6): ML tahmini hangi horizon ve
-- hangi gerçekleşen fiyatla değerlendirilecek; observer sonucu para P&L'i gibi gösterilmez.
create table if not exists paper_predictions (
    id              uuid primary key default gen_random_uuid(),
    run_id          uuid references paper_runs(id) on delete cascade,
    season_id       text references arena_seasons(season_id) on delete cascade,
    ticker          text not null,
    signal_asof     timestamptz not null,
    horizon_days    integer not null default 10,
    model_version   text,
    p_up            numeric,
    skill           numeric,                           -- brier_skill_score
    recent_skill    numeric,
    auc             numeric,
    brier           numeric,
    n_samples       integer,
    quality_passed  boolean,
    rejection_reasons jsonb,
    realized_asof   timestamptz,                       -- horizon dolduğunda
    realized_price  numeric,
    realized_up     boolean,                           -- gerçekleşen yön (karne)
    metadata        jsonb,
    created_at      timestamptz not null default now(),
    unique (season_id, ticker, signal_asof, model_version)
);

-- ── Güvenlik (RLS) — tek kullanıcılı kurulum (schema.sql ile aynı ilke) ──────
alter table arena_seasons     enable row level security;
alter table paper_accounts    enable row level security;
alter table paper_runs        enable row level security;
alter table paper_orders      enable row level security;
alter table paper_fills       enable row level security;
alter table paper_cash_ledger enable row level security;
alter table paper_positions   enable row level security;
alter table paper_equity      enable row level security;
alter table paper_predictions enable row level security;
