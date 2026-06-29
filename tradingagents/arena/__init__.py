"""Paper Trading Arena — BIST hisseleri için eşit kasayla yarışan kâğıt hesaplar.

Plan: ``docs/PAPER_TRADING_ARENA_PLAN.md``. Felsefe (Codex'in "fail-cheap" kararı):
arena tek satır gerçek altyapı yazılmadan önce **edge kapısı** geçilmeli — yani
sinyaller maliyet sonrası XU100 al-tut'u geçiyor mu? Bu paket o soruyu yanıtlar.

Katmanlar:
  - :mod:`config`    — ortak execution fiziği (komisyon/slippage/buffer/settlement)
  - :mod:`profiles`  — 4 para hesabı + 1 ML observer (kurallar, değişmez snapshot)
  - :mod:`metrics`   — getiri/Sharpe/max-drawdown/CAGR (saf)
  - :mod:`engine`    — portföy backtest motoru (T+1 fill, OHLC stop/hedef, kill-switch)
  - :mod:`replay`    — tarihsel edge kapısı: her profil vs XU100 (survivorship uyarılı)

UI: ``app_pages/arena_page.py``. Supabase şeması: ``storage/schema_arena.sql``.
Hiçbiri LLM/ücret harcamaz; tamamı deterministik ve yereldir.
"""

from tradingagents.arena.config import DEFAULT_EXECUTION, ExecutionConfig  # noqa: F401
from tradingagents.arena.profiles import PROFILES, Profile  # noqa: F401
