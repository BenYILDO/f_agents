"""Makine öğrenmesi katmanı — sinyal isabetini ölçen/öngören modeller.

Deterministik analiz motorunun (indikatörler + bileşen skorları) ürettiği
özelliklerden bir sınıflandırıcı eğitir ve zaman-serisi bölmeli backtest ile
isabetini raporlar. Etiket varsayılanı **üçlü-bariyer** (S1): "N gün sonra
yukarı mı?" değil, "bu barda açılan ATR stop/hedefli işlem maliyet-sonrası
(XU100-relatif) kazanır mı?" — model, motorun gerçekten sorduğu soruyu öğrenir.

Tasarım: model, hissenin KENDİ geçmişi üzerinde talep anında (Streamlit'te)
eğitilir — önceden eğitilmiş ağırlık paketlenmez (her hissenin/rejimin
dinamiği farklıdır ve veri canlı çekilir). scikit-learn dışında ağır bağımlılık
yoktur.
"""

from tradingagents.ml.features import (  # noqa: F401
    make_labels_triple_barrier,
    triple_barrier_outcomes,
)
from tradingagents.ml.model import (  # noqa: F401
    SignalModelResult,
    train_signal_model,
)
