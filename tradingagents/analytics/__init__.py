"""Deterministik teknik & temel analiz motoru (LLM yok, maliyet yok).

Bu paket, hem Streamlit'te bağımsız "Teknik Analiz" sayfalarını besler hem de
AI analistlerine (Market Analyst) önceden hesaplanmış, sayısal bir "teknik
brif" enjekte eder. Tamamı saf pandas/numpy — harici TA kütüphanesi (TA-Lib
vb.) gerektirmez, böylece kurulum her platformda sorunsuz kalır.

Modüller:
  - :mod:`indicators`         — RSI, MACD, ADX, ATR, Bollinger, SMA/EMA (ortak çekirdek)
  - :mod:`candlesticks`       — mum formasyonları (yutan, çekiç, doji, yıldız…)
  - :mod:`patterns`           — grafik formasyonları (OBO, ikili tepe/dip, üçgen…)
  - :mod:`support_resistance` — pivot kümelemeli destek/direnç + Fibonacci
  - :mod:`seasonality`        — ay/gün bazlı sezonsallık istatistikleri
  - :mod:`regime`             — volatilite/trend rejimi + TL stres göstergesi
  - :mod:`composite`          — tüm sinyallerin ağırlıklı kompozit skoru
  - :mod:`fundamental_score`  — Piotroski-tarzı temel sağlamlık skoru (yfinance)
"""

from tradingagents.analytics.composite import (  # noqa: F401
    compute_composite,
    build_technical_brief,
    CompositeResult,
)
from tradingagents.analytics.ratio_score import (  # noqa: F401
    compute_ratio_score,
    format_ratio_brief,
    RatioScoreResult,
)
from tradingagents.analytics.combined import (  # noqa: F401
    combined_signal,
    CombinedResult,
)
