"""Analiz orkestrasyonu — deterministik motorları tek bir snapshot'a bağlar.

  - :mod:`trust` — güven katmanının saf (ağsız, test edilebilir) çekirdeği:
    çoklu-yöntem mutabakatı, sinyal istikrarı (flip-flop), sinyal karnesi.
  - :mod:`run`   — bir hisseyi (ya da evreni) analiz edip ``analysis_snapshots``
    satırı üretir. Hem "ekleyince anında analiz" / "şimdi analiz et" butonu hem
    de saat başı zamanlayıcı (GitHub Actions) bu tek kaynağı kullanır.
"""
