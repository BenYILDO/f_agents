"""Makine öğrenmesi katmanı — sinyal isabetini ölçen/öngören modeller.

Deterministik analiz motorunun (indikatörler + bileşen skorları) ürettiği
özelliklerden, hissenin ileriye dönük (N gün) yön olasılığını tahmin eden bir
sınıflandırıcı eğitir ve zaman-serisi bölmeli backtest ile isabetini raporlar.

Tasarım: model, hissenin KENDİ geçmişi üzerinde talep anında (Streamlit'te)
eğitilir — önceden eğitilmiş ağırlık paketlenmez (her hissenin/rejimin
dinamiği farklıdır ve veri canlı çekilir). scikit-learn dışında ağır bağımlılık
yoktur.
"""

from tradingagents.ml.model import (  # noqa: F401
    SignalModelResult,
    train_signal_model,
)
