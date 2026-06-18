"""Pozisyon boyutu — kesirli Kelly + volatilite hedefleme.

Ne kadar alınacağını matematiksel olarak belirler:

  - :func:`kelly_fraction` — Kelly (1956): kazanma olasılığı ve ödül/risk oranından
    sermayenin optimal payı. Tam Kelly oynaktır → :func:`fractional_kelly` (½-Kelly).
  - :func:`vol_target_weight` — Moreira & Muir (2017): pozisyonu öngörülen
    volatiliteyle ters ölçekle (vol yüksekken küçült) → daha yüksek Sharpe.
  - :func:`position_size` — ikisini birleştirip nihai sermaye payını üretir (tavanlı).

Saf/ağsız.
"""

from __future__ import annotations

from dataclasses import dataclass


def kelly_fraction(p_win: float, win_loss_ratio: float) -> float:
    """Kelly payı: f* = p − (1−p)/b. Negatifse 0 (pozisyon açma)."""
    if win_loss_ratio <= 0:
        return 0.0
    f = p_win - (1 - p_win) / win_loss_ratio
    return max(0.0, f)


def fractional_kelly(p_win: float, win_loss_ratio: float, fraction: float = 0.25) -> float:
    """Kesirli Kelly — tam Kelly'nin ``fraction`` katı.

    Varsayılan **çeyrek Kelly (0.25)**: tam Kelly rejim değişiminde/parametre
    hatasında iflas riski taşır; modern fonların standardı çeyrek Kelly'dir
    (López de Prado; Thorp). Daha düşük oynaklık, kontrollü düşüş.
    """
    return kelly_fraction(p_win, win_loss_ratio) * fraction


def vol_target_weight(forecast_vol: float, target_vol: float = 0.15,
                      max_leverage: float = 1.5) -> float:
    """Volatilite hedefleme ölçek katsayısı: target/forecast (tavanlı)."""
    if forecast_vol <= 0:
        return 0.0
    return min(max_leverage, target_vol / forecast_vol)


@dataclass
class SizingResult:
    weight: float                 # nihai sermaye payı [0, max_weight]
    kelly: float                  # tam Kelly
    kelly_used: float             # kesirli Kelly
    vol_scalar: float             # vol hedefleme katsayısı
    note: str = ""


def position_size(
    p_win: float,
    win_loss_ratio: float,
    forecast_vol: float,
    target_vol: float = 0.15,
    kelly_mult: float = 0.25,
    max_weight: float = 0.20,
) -> SizingResult:
    """Çeyrek Kelly × vol-hedefleme → nihai pozisyon ağırlığı (sermaye payı)."""
    k = kelly_fraction(p_win, win_loss_ratio)
    kf = k * kelly_mult
    vs = vol_target_weight(forecast_vol, target_vol)
    weight = max(0.0, min(max_weight, kf * vs))
    note = (f"Kelly={k:.2f} → çeyrek-Kelly={kf:.2f} × vol-ölçek={vs:.2f} "
            f"→ ağırlık %{weight*100:.1f} (tavan %{max_weight*100:.0f})")
    return SizingResult(round(weight, 4), round(k, 4), round(kf, 4),
                        round(vs, 4), note)
