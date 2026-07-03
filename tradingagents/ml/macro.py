"""Makro özellik çerçevesi (S3) — BIST'in gerçek sürücüleri modele girer.

Literatür BIST100 için USDTRY, altın ve endeks rejimini ana açıklayıcılar olarak
gösterir (plan §S3); projede bunlar kapı/gösterge olarak vardı ama modele
**özellik** olarak girmiyordu. Bu modül tarih-indeksli tek bir makro çerçeve
üretir; havuz paneli bunu tarih üzerinden join'ler (aynı gün tüm hisselere aynı
makro satır düşer).

Nedensellik notu: tüm kolonlar yalnız o güne KADARKİ veriyle hesaplanır
(pct_change / rolling — geriye bakan pencereler). Rejim için HMM'in tam-örneklem
parametreleri kullanılmaz (parametre sızıntısı olur); nedensel vekiller kullanılır:
MA200 üstü bayrağı + gerçekleşen volatilite (replay'in rejim proxy'siyle tutarlı).

Eksik seri sistemi kırmaz: verilmeyen kaynak kolonları nötr (0.0) doldurulur —
mevcut graceful-degrade deseni.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MACRO_COLUMNS = [
    "xu_ret_20",        # XU100 20 günlük momentum
    "xu_above_ma200",   # XU100 200 günlük ortalamanın üstünde mi (rejim vekili)
    "xu_vol_20",        # XU100 20 günlük gerçekleşen vol (yıllıklandırılmış)
    "usdtry_ret_20",    # USDTRY 20 günlük ivme (TL stresi)
    "usdtry_vol_20",    # USDTRY 20 günlük gerçekleşen vol
    "gold_ret_20",      # Altın (USD) 20 günlük momentum
]


def _ret_vol(close: pd.Series, idx: pd.Index) -> tuple[pd.Series, pd.Series]:
    """Bir kapanış serisinden (idx'e ffill hizalı) 20g momentum + yıllık vol."""
    c = close.sort_index()
    c = c[~c.index.duplicated(keep="last")].reindex(idx, method="ffill")
    ret20 = c.pct_change(20)
    vol20 = c.pct_change().rolling(20).std() * np.sqrt(252)
    return ret20, vol20


def build_macro_frame(
    xu_close: pd.Series,
    usdtry_close: pd.Series | None = None,
    gold_close: pd.Series | None = None,
) -> pd.DataFrame:
    """Tarih-indeksli makro özellik çerçevesi (kolonlar: ``MACRO_COLUMNS``).

    ``xu_close`` zorunludur (takvimi de o belirler); USDTRY/altın verilmezse o
    kolonlar nötr 0.0 kalır. İlk ~200 bar (MA200 ısınması) NaN'dır ve panelde
    düşer — özellik ısınmalarıyla aynı davranış.
    """
    xu = xu_close.sort_index()
    xu = xu[~xu.index.duplicated(keep="last")]
    idx = xu.index
    out = pd.DataFrame(index=idx)

    out["xu_ret_20"] = xu.pct_change(20)
    ma200 = xu.rolling(200).mean()
    out["xu_above_ma200"] = (xu > ma200).astype(float).where(ma200.notna())
    out["xu_vol_20"] = xu.pct_change().rolling(20).std() * np.sqrt(252)

    if usdtry_close is not None and len(usdtry_close) > 21:
        r, v = _ret_vol(usdtry_close, idx)
        out["usdtry_ret_20"], out["usdtry_vol_20"] = r, v
    else:
        out["usdtry_ret_20"] = 0.0
        out["usdtry_vol_20"] = 0.0

    if gold_close is not None and len(gold_close) > 21:
        r, _ = _ret_vol(gold_close, idx)
        out["gold_ret_20"] = r
    else:
        out["gold_ret_20"] = 0.0

    return out[MACRO_COLUMNS].replace([np.inf, -np.inf], np.nan)
