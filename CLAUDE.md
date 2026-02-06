# Energy Forecast

Uludağ elektrik dağıtım bölgesi (Bursa, Balıkesir, Yalova, Çanakkale) saatlik elektrik tüketimi tahmin sistemi.
CatBoost + Prophet + TFT ensemble, 48 saat ileri (T + T+1), FastAPI serving, AWS deploy.

## Referans Proje
Eski proje: C:\Users\pc\Desktop\distributed-energy-forecasting\
Bu projeyi KOD KOPYALAMADAN referans olarak kullan.
Mantığı oku, anla, temiz yeniden yaz.

## Forecast Akışı
- Müşteri T-1 günü 23:00'a kadar tüketim verisi verir (Excel)
- Model 48 saat tahmin üretir: T günü (Gün İçi) + T+1 günü (Gün Öncesi)
- Kullanıcı web UI'da seçer: sadece T+1 (GOP) veya T+T+1 (GOP+GİP)
- Aynı gün güncellenmiş hava durumu ile tekrar talep yapılabilir

## Komutlar
```
make install          # uv sync
make test             # pytest -x --tb=short
make lint             # ruff check + mypy --strict
make format           # ruff format
make train-catboost   # CatBoost eğitimi
make train-prophet    # Prophet eğitimi
make train-tft        # TFT eğitimi
make train-ensemble   # Ensemble eğitimi
make prepare-data     # Feature pipeline çalıştır
make serve            # FastAPI başlat (uvicorn)
```

## Kod Stili
- Type hints ZORUNLU her yerde (mypy strict)
- Docstring: Google style, kısa ve öz
- Loguru logger — ASLA print()
- line-length: 100
- Ruff formatter (black yerine)
- Conventional Commits — ASLA commit mesajında AI/Claude referansı YAPMA

## Mimari Kurallar
- Her feature engineer BaseFeatureEngineer'dan türer (sklearn uyumlu)
- Config değişikliği = configs/*.yaml güncelle, KOD DEĞİŞMESİN
- Her model BaseForecaster ABC'den türer (train, predict, save, load)
- Feature pipeline tüm modlarda aynı feature'ları üretir
- Ham EPIAS değerleri pipeline çıkışında her zaman DROP

## Data Leakage Kuralları (ASLA İHLAL ETME)
- Consumption/EPIAS lag feature'larda min_lag=48 saat
- Ham EPIAS değerleri (FDPP, Real_Time_Consumption, DAM_Purchase, Bilateral, Load_Forecast) training'den önce DROP
- Rolling/expanding window: .shift(1) SONRA .rolling()
- Expanding min_periods >= 48
- TimeSeriesSplit: ASLA random shuffle, has_time=true

## Leakage OLMAYAN Durumlar (Karıştırma)
- Solar feature'lar (lead dahil): Astronomik hesaplama, deterministik — her modda OK
- Weather forecast (T, T+1): Tahmin anında OpenMeteo'dan mevcut — her modda OK
- Bu ikisi hem training hem prediction'da kullanılabilir

## Veri Kuralları
- Bölge: Uludağ (Bursa + Balıkesir + Yalova + Çanakkale toplam tüketimi)
- Birim: MWh (saatlik)
- Zaman dilimi: Europe/Istanbul (UTC+3)
- Frekans: Saatlik (24 değer/gün)
- Hava durumu: 4 şehir ağırlıklı ortalama (Bursa %60, Balıkesir %24, Yalova %10, Çanakkale %6)
- Tatiller: data/static/turkish_holidays.parquet
- EPIAS cache: data/external/epias/{year}.parquet
- CatBoost kategorik kolonlar: configs/models/catboost.yaml

## Detaylı Bilgi
@SPEC.md — Proje spesifikasyonu (forecast akışı, model mimarisi, API tasarımı)
@PROJECT_KNOWLEDGE.md — Eski proje detayları (feature listesi, config değerleri, referans)
@docs/plans/ — Milestone planları

## Milestone Durumu
- [x] M0: Proje iskeleti
- [x] M1: Config sistemi
- [x] M2: Data pipeline (ingestion + EPIAS/OpenMeteo clients)
- [x] M3: Feature engineering (5 modül + pipeline orkestratör)
- [ ] M4: Leakage audit
- [ ] M5: CatBoost training (TSCV + Optuna + MLflow)
- [ ] M6: Prophet training
- [ ] M7: 2-model ensemble (Faz 1 tamamlanır)
- [ ] M8: TFT training
- [ ] M9: 3-model ensemble (Faz 2 tamamlanır)
- [ ] M10: API serving (FastAPI + auth)
- [ ] M11: Docker + CI/CD (AWS deploy)
