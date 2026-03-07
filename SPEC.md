# SPEC.md — Energy Forecast

> Proje anayasası. Claude Code ve geliştirici bu dosyayı tek kaynak olarak referans alır.
> Son güncelleme: 2026-03-06

---

## 1. Proje Tanımı

**Ne:** Uludağ elektrik dağıtım bölgesi (Bursa, Balıkesir, Yalova, Çanakkale) için
saatlik elektrik tüketimi tahmin sistemi.

**Neden:** Enerji dağıtım şirketleri Gün Öncesi Planı (GOP) ve Gün İçi Planı (GİP)
için ertesi günün saatlik tüketim tahminlerine ihtiyaç duyar. Doğru tahmin, dengesizlik
maliyetlerini azaltır ve şebeke planlamasını iyileştirir.

**Kimin için:** Enerji dağıtım/tedarik şirketleri. Web arayüzü üzerinden talep edilen
tahminler üretilir.

---

## 2. Forecast Akışı

### 2.1 Temel Kavramlar

```
T     = Bugün (tahmin talep edilen gün)
T-1   = Dün (son verinin geldiği gün)
T+1   = Yarın (ana tahmin hedefi)
```

### 2.2 Veri ve Tahmin Akışı

```
MÜŞTERİ VERİR                           MODEL TAHMİN EDER
─────────────                            ──────────────────
Geçmiş tüketim:                          T günü   00:00 → 23:00  (24 saat, Gün İçi)
... → T-1 günü 23:00                     T+1 günü 00:00 → 23:00  (24 saat, Gün Öncesi)
                                         ─────────────────────────
                                         Toplam: 48 saatlik tahmin
```

### 2.3 Kullanıcı Seçimi (Web UI Toggle)

| Seçim | Ne Alır | Kullanım Amacı |
|-------|---------|----------------|
| Sadece T+1 | 24 değer (yarın 00:00-23:00) | GOP — Gün Öncesi Planı |
| T + T+1 | 48 değer (bugün + yarın) | GOP + GİP — Gün Öncesi + Gün İçi |

Model her zaman 48 saat üretir. Kullanıcıya gösterilecek kısım toggle ile belirlenir.
> Not: forecast_type parametresi şu an devre dışı (TODO). Filtreleme ileride implement edilecek.

### 2.4 Güncelleme Senaryosu

Aynı gün içinde birden fazla talep yapılabilir:

```
Saat 10:00 → Talep #1
  - Geçmiş veri: ... → T-1 23:00
  - Hava durumu tahmini: O anki OpenMeteo forecast
  - Çıktı: T + T+1 tahminleri

Saat 15:00 → Talep #2 (güncelleme)
  - Geçmiş veri: Aynı (... → T-1 23:00, değişmez)
  - Hava durumu tahmini: Güncellenmiş OpenMeteo forecast
  - Çıktı: Güncellenmiş T + T+1 tahminleri
```

Hava durumu tahmini zamanla iyileşir → daha sonraki talepler daha doğru tahmin üretir.

### 2.5 Zaman Gap'i

```
Son veri noktası:     T-1 23:00
İlk tahmin noktası:   T   00:00
Gap:                  1 saat (minimum)
En uzak tahmin:       T+1 23:00
Maksimum gap:         48 saat
```

Bu yapı `min_lag=48` kuralını doğal olarak karşılar — tahmin edilen en uzak nokta,
son veri noktasından 48 saat sonradır.

---

## 3. Veri Kaynakları

### 3.1 Tüketim Verisi (Ana Input)

| Özellik | Değer |
|---------|-------|
| Kaynak | Müşteri tarafından Excel (.xlsx) upload |
| İçerik | Bölge toplam tüketimi (4 şehir birleşik) |
| Kolonlar | `date`, `time`, `consumption` |
| Birim | MWh (saatlik) |
| Frekans | Saatlik (24 değer/gün) |
| `time` formatı | 0-23 arası integer |
| Son veri noktası | T-1 günü saat 23:00 |

### 3.2 EPİAŞ Piyasa Verileri

| Özellik | Değer |
|---------|-------|
| Kaynak | EPİAŞ Transparency Platform REST API |
| Authentication | Username/password → JWT token |
| Cache | Yıllık parquet dosyaları: `data/external/epias/epias_market_{year}.parquet` |
| File pattern | Config-driven (EpiasApiConfig.file_pattern) |
| Retry | 3 deneme, exponential backoff (tenacity) |
| Rate limit | 10 saniye delay (configurable) |

**Değişkenler:**

| Kısaltma | Tam Adı | Durum |
|----------|---------|-------|
| ~~FDPP~~ | ~~Final Daily Production Plan~~ | **DEPRECATED** — API'den artık çekilemiyor |
| Real_Time_Consumption | — | Aktif |
| DAM_Purchase | Day-Ahead Market Purchase | Aktif |
| Bilateral_Agreement_Purchase | — | Aktif |
| Load_Forecast | — | Aktif |

**Kritik kural:** Ham EPİAŞ değerleri doğrudan feature olarak KULLANILMAZ.
Sadece türetilmiş versiyonları (lag, rolling) kullanılır. Ham değerler pipeline'da drop edilir.

### 3.3 Hava Durumu (OpenMeteo)

| Özellik | Değer |
|---------|-------|
| Kaynak | Open-Meteo API (ücretsiz) |
| Yaklaşım | 4 şehir ağırlıklı ortalama (sayısal kolonlar) |
| Authentication | Yok (açık API) |
| Cache | SQLite backend: `data/external/weather_cache.sqlite` |
| Validation | Pandera WeatherSchema |

**Lokasyon Ağırlıkları:**

| Şehir | Ağırlık | Enlem | Boylam |
|-------|---------|-------|--------|
| Bursa | %60 | 40.183 | 29.050 |
| Balıkesir | %24 | 39.653 | 27.886 |
| Yalova | %10 | 40.655 | 29.272 |
| Çanakkale | %6 | 40.146 | 26.402 |

**Ağırlıklandırma kuralı:**
- Sayısal değişkenler (temperature, humidity, wind...): Ağırlıklı ortalama, NaN-safe renormalize (eksik şehir ağırlığı dışlanır, kalanlar normalize edilir)
- weather_code (WMO 4677 kategorik): Ağırlıklı ortalama YAPILMAZ → dominant city stratejisi (en yüksek ağırlıklı şehrin kodu, NaN fallback)
- weather_group: weather_code'dan türetilen 8-grup string categorical (clear, cloudy, fog, drizzle, rain, snow, showers, thunderstorm)

**Değişkenler:**
temperature_2m, relative_humidity_2m, dew_point_2m, apparent_temperature,
precipitation, snow_depth, weather_code, surface_pressure,
wind_speed_10m, wind_direction_10m, shortwave_radiation

**Hava durumu verisi kullanım kuralı:**

| Durum | Veri Tipi | Kaynak |
|-------|-----------|--------|
| Training (geçmiş) | Historical weather | OpenMeteo Archive API |
| Prediction (gelecek) | Weather forecast | OpenMeteo Forecast API |
| Dataset hazırlama | Historical + Forecast concat | Her ikisi birlikte çekilir |

Weather forecast ve solar hesaplamaları (lead dahil) data leakage DEĞİLDİR:
- **Solar:** Astronomik hesaplama, deterministik — kesin bilinir
- **Weather forecast:** Tahmin anında OpenMeteo'dan mevcut — bilinebilir bilgi

### 3.4 Statik Veriler

| Dosya | İçerik | Format | Zorunlu |
|-------|--------|--------|---------|
| `data/static/turkish_holidays.parquet` | Türk resmi tatilleri | Parquet | Evet |
| `data/external/profile/profile_coef_{year}.parquet` | EPİAŞ profil katsayıları | Parquet | Evet |

---

## 4. Feature Engineering

### 4.1 Feature Modülleri

5 feature engineer modülü, her biri `BaseFeatureEngineer`'dan türer:

| Modül | Kaynak Veri | Ana Feature Grupları |
|-------|-------------|---------------------|
| Calendar | `date` kolonu | Datetime, cyclical, holiday, Ramazan, business hours, solar position |
| Consumption | `consumption` kolonu | Lag, rolling, EWMA, momentum, volatilite, quantile, interaction |
| Weather | OpenMeteo verileri | HDD/CDD, comfort index, extremes, rolling, severity, weather_group |
| Solar | pvlib hesaplamaları | GHI/DNI/DHI, POA, clearness index, cloud proxy |
| EPIAS | EPİAŞ piyasa verileri | Lag, rolling, expanding (türetilmiş değerler) |

**Toplam feature sayısı:** ~153

### 4.2 Data Leakage Kuralları

Bu kurallar ihlal edildiğinde model gerçek performansından yüksek sonuç verir
ama production'da başarısız olur. KESİNLİKLE UYULMALIDIR:

| # | Kural | Uygulama |
|---|-------|----------|
| 1 | min_lag = 48 saat | Tüm consumption ve EPİAŞ lag feature'larında (config + Pydantic ge=48 validator) |
| 2 | Shift before roll | `.shift(1).rolling()` — önce kaydır, sonra pencerele |
| 3 | Expanding min_periods | `min_periods >= 48` |
| 4 | Ham EPİAŞ değerleri | Feature pipeline çıkışında DROP edilir |
| 5 | Temporal split | ASLA random shuffle — `has_time=true`, shuffle=True → ValueError |
| 6 | consumption duplicate | Pipeline merge sonrası duplicate kolon kontrolü |
| 7 | Selective forward-fill | ffill/bfill SADECE weather kolonları — consumption/EPIAS'a dokunma |

**Leakage OLMAYAN durumlar (known future inputs):**

| Feature | Neden Leakage Değil |
|---------|---------------------|
| Solar (lead dahil tümü) | Astronomik hesaplama, deterministik — herhangi bir tarih/saat için kesin bilinir |
| Weather forecast (T, T+1) | Tahmin anında OpenMeteo'dan mevcut — tahmin ama bilinebilir bilgi |

Bu feature'lar hem training hem prediction'da kullanılabilir.

### 4.3 Dataset Hazırlama (Unified Pipeline)

Feature pipeline training ve prediction için AYNI şekilde çalışır. Ayrı "mode" yoktur.

**Kritik kural — tek seferde pipeline:**

```
1. Excel yükle (son satır T-1 23:00)
2. 48 boş satır ekle (T + T+1, consumption=NaN)
3. EPIAS + Weather merge et (historical + forecast weather birleşik)
4. Feature pipeline TEK SEFERDE çalıştır
5. Split:
   - historical = df[:-48] → training için
   - forecast = df[-48:] → prediction için
6. Pandera schema validation (ConsumptionSchema, EpiasSchema, WeatherSchema)
```

Bu sayede uzun lag'ler (consumption_lag_720 gibi) forecast satırlarında da doğru hesaplanır.

**Çıktı dosyaları:**
- `data/processed/features_historical.parquet` (~48K satır, ~153 feature)
- `data/processed/features_forecast.parquet` (48 satır, ~153 feature)

---

## 5. Model Mimarisi

### 5.1 Üç Model Ensemble

```
Input (feature-engineered DataFrame)
        │
        ├──→ CatBoost (gradient boosting)  ─→ 48 saatlik tahmin
        ├──→ Prophet  (trend+seasonality)  ─→ 48 saatlik tahmin
        └──→ TFT      (attention-based DL) ─→ 48 saatlik tahmin
                │
                ▼
        Ensemble (weighted average)
        prediction = w₁·CB + w₂·P + w₃·TFT
                │
                ▼
        48 saatlik final tahmin (PREDICTION_COL = "consumption_mwh")
        ├── T   00:00-23:00 (Gün İçi)
        └── T+1 00:00-23:00 (Gün Öncesi)
```

### 5.2 CatBoost

| Parametre | Değer | Not |
|-----------|-------|-----|
| Task type | CPU | GPU opsiyonel |
| Iterations | 1000-3000 | Optuna ile optimize |
| Learning rate | 0.01-0.1 | Log scale |
| Depth | 4-7 | Optuna ile optimize |
| Loss function | RMSE / MAE / MAPE | Optuna ile seç |
| Early stopping | 50 rounds | Validation metric'e göre |
| has_time | true | Zaman sırası korunur |
| Kategorik kolonlar | 28 adet, configs/models/catboost.yaml'da tanımlı | 6 grup: Time, Holiday, Interaction, Time-period, Weather, Season/Solar |

**Güçlü yanı:** Feature etkileşimleri (tatil × saat × mevsim), tabular data'da en iyi performans.

### 5.3 Prophet

| Parametre | Değer |
|-----------|-------|
| Seasonality mode | multiplicative (sabit, Optuna'dan çıkarıldı) |
| Daily Fourier order | 15 |
| Weekly Fourier order | 8 |
| Yearly Fourier order | 12 |
| Holidays | TR resmi tatilleri + Ramazan (3-tier: bayram -1/+1, resmi 0/+1, ramazan 0/0) |
| Regressors | 14 adet (asagida) |
| Changepoint prior scale | 0.001-1.0 (Optuna ile optimize) |
| n_changepoints | 15-50 (Optuna ile optimize) |

**Prophet Regressors (14 adet, R2 güncel):**

| # | Regressor | Mode |
|---|-----------|------|
| 1 | consumption_lag_168 | multiplicative |
| 2 | consumption_lag_48 | multiplicative |
| 3 | consumption_lag_720 | multiplicative |
| 4 | temperature_2m | multiplicative |
| 5 | apparent_temperature | multiplicative |
| 6 | relative_humidity_2m | additive |
| 7 | shortwave_radiation | multiplicative |
| 8 | wth_cdd | multiplicative |
| 9 | wth_hdd | multiplicative |
| 10 | is_weekend | multiplicative |
| 11 | is_sunday | multiplicative |
| 12 | is_holiday | multiplicative |
| 13 | is_business_hours | multiplicative |
| 14 | sol_elevation | multiplicative |

**Güçlü yanı:** Trend + mevsimsellik yapısal ayrıştırma, tatil etkilerini doğal modelleme.

### 5.4 TFT (Temporal Fusion Transformer — NeuralForecast)

| Parametre | Değer |
|-----------|-------|
| Framework | NeuralForecast (nixtla) |
| Hidden size | 128 (prod) |
| Attention heads (n_head) | 2 |
| RNN layers (n_rnn_layers) | 1 |
| Dropout | 0.1 |
| Encoder length | 168 saat (7 gün) |
| Prediction length (h) | 48 saat |
| max_steps | 10000 (prod) |
| windows_batch_size | 2048 (prod) |
| Loss | MAE |
| Covariates | futr_exog_list (known), hist_exog_list (unknown) |

**Güçlü yanı:** Attention-based interpretability, GPU utilization %96+,
hangi feature'ın hangi saatte önemli olduğunu gösterir.

### 5.5 Ensemble

İki mod desteklenir:

**Stacking (varsayılan):**
- CatBoost meta-learner (depth=2) OOF val predictions üzerinde eğitilir
- Context features: hour, day_of_week, is_weekend, month, is_holiday
- Temporal 80/20 split ile meta-learner validation

**Weighted Average (fallback):**
- `scipy.optimize.minimize` SLSQP (constraint: Σwᵢ = 1, wᵢ ≥ 0)
- Metrik: MAPE(y, Σwᵢ·predᵢ) — blended predictions üzerinden
- NOT: MAPE(blended) ≠ Σwᵢ·MAPE(predᵢ)

**Ortak:**
- Graceful degradation: Bir model başarısız olursa kalanlarla devam eder
- Stacking < 2 model ile çalışamaz → otomatik weighted_average'a düşer

**Faz Planı:**

| Faz | Modeller | Durum |
|-----|----------|-------|
| Faz 1 | CatBoost + Prophet | ✅ Tamamlandı |
| Faz 2 | CatBoost + Prophet + TFT | ✅ Tamamlandı (3-model production HPO pending) |

---

## 6. Training Stratejisi

### 6.1 Cross-Validation

Time Series Cross-Validation (TSCV) — expanding window:

```
Split 1: [████████ train ████████][val][test]
Split 2: [█████████ train █████████][val][test]
Split 3: [██████████ train ██████████][val][test]
...
Split N: [█████████████████ train █████████████████][val][test]
```

| Parametre | Değer |
|-----------|-------|
| n_splits | 12 |
| val_period | 1 ay |
| test_period | 1 ay |
| Shuffle | HAYIR (ValueError ile enforce) |

### 6.2 Hyperparameter Tuning

- Optuna ile parametreler optimize edilir
- Objective: Validation MAPE ortalaması (tüm split'ler üzerinden)
- Storage: n_trials > 3 → SQLite (crash recovery), n_trials ≤ 3 → in-memory

| Ayar | Değer |
|------|-------|
| n_trials | 50+ (CatBoost), 30 (Prophet), 20 (TFT) |
| iterations | 1000-3000 |

### 6.3 Evaluation Metrikleri

| Metrik | Tip | Açıklama |
|--------|-----|----------|
| **MAPE** | Birincil | Mean Absolute Percentage Error (%) |
| MAE | Standart | Mean Absolute Error |
| RMSE | Standart | Root Mean Squared Error |
| R² | Standart | Coefficient of determination |
| SMAPE | Ek | Symmetric MAPE (%) |
| WMAPE | Ek | Weighted MAPE (%) |
| MBE | Ek | Mean Bias Error (over/under-prediction) |

### 6.4 Experiment Tracking

MLflow ile tüm eğitimler izlenir:
- Hyperparametreler
- Metrikler (per-split ve aggregate)
- Model artifact'ları
- Feature importance

### 6.5 Hızlı Test

Ayrı smoke test config/script yok — tek pipeline, tek YAML. Hızlı validation için
model YAML'larında (configs/models/hyperparameters.yaml, tft.yaml) değerleri geçici
olarak düşür, çalıştır, sonra geri al:

```bash
# hyperparameters.yaml'da n_trials: 1, n_splits: 2 yap, sonra:
uv run python -m energy_forecast.training.run --model catboost --no-mlflow

# Bitince değerleri production'a geri al (n_trials: 50, n_splits: 12)
# Veya: git checkout -- configs/  ile tüm config değişikliklerini geri al
```

---

## 7. Serving / API

### 7.1 Genel Yapı

- **Framework:** FastAPI
- **Deploy:** AWS (Docker container)
- **Tetik:** Web UI'dan kullanıcı talebi (on-demand)
- **Authentication:** Bearer token (API_KEY), /health hariç tüm endpoint'ler
- **Timing-safe:** secrets.compare_digest() ile key karşılaştırma
- **CORS:** Config-driven origins (settings.api.cors_origins)
- **Timezone:** Tüm datetime UTC+3 (Europe/Istanbul)
- **Pattern:** Async job processing

### 7.2 Kullanıcı Akışı

```
1. Kullanıcı web UI'ya girer
2. Excel dosyası upload eder (tüketim verisi → T-1 23:00'a kadar)
3. "Tahmin Üret" butonuna tıklar
4. Backend (async):
   a. Job oluştur → job_id döndür
   b. Background'da:
      - Excel'i parse et → DataFrame
      - EPİAŞ verisini çek (cache veya API)
      - OpenMeteo hava durumu tahminini çek (historical + forecast)
      - Feature pipeline çalıştır
      - Ensemble prediction üret (48 saat)
   c. Job tamamlanınca sonucu sakla
5. Frontend job_id ile status polling yapar
6. Tamamlanınca sonuç döner (JSON + Excel download)
```

### 7.3 API Endpoints

| Method | Path | Auth | Açıklama |
|--------|------|------|----------|
| GET | `/health` | Hayır | Sağlık kontrolü |
| GET | `/models` | Evet | Aktif model bilgileri |
| POST | `/predict` | Evet | Yeni tahmin job'ı oluştur |
| GET | `/jobs/{job_id}` | Evet | Job durumunu sorgula |
| GET | `/jobs/{job_id}/result` | Evet | Tamamlanan job sonucunu al |
| DELETE | `/jobs/{job_id}` | Evet | Job'ı iptal et |
| GET | `/docs` | Hayır | OpenAPI dokümantasyonu |

### 7.4 Job Oluşturma

```
POST /predict
Content-Type: multipart/form-data
Authorization: Bearer {API_KEY}

file: consumption.xlsx
```

**Response:**
```json
{
  "job_id": "abc123",
  "status": "pending",
  "created_at": "2025-01-01T10:00:00+03:00"
}
```

### 7.5 Job Status

```
GET /jobs/{job_id}
Authorization: Bearer {API_KEY}
```

**Response:**
```json
{
  "job_id": "abc123",
  "status": "completed",
  "created_at": "2025-01-01T10:00:00+03:00",
  "completed_at": "2025-01-01T10:00:25+03:00",
  "progress": 100
}
```

### 7.6 Job Result

```
GET /jobs/{job_id}/result
Authorization: Bearer {API_KEY}
```

**Response:**
```json
{
  "success": true,
  "predictions": [
    {"datetime": "2025-01-01T00:00:00+03:00", "consumption_mwh": 1245.3, "period": "intraday"},
    {"datetime": "2025-01-01T01:00:00+03:00", "consumption_mwh": 1198.7, "period": "intraday"},
    {"datetime": "2025-01-02T00:00:00+03:00", "consumption_mwh": 1267.1, "period": "day_ahead"}
  ],
  "metadata": {
    "model": "ensemble_v1",
    "weights": {"catboost": 0.55, "prophet": 0.25, "tft": 0.20},
    "last_data_point": "2024-12-31T23:00:00+03:00",
    "weather_updated_at": "2025-01-01T10:00:00+03:00",
    "latency_ms": 12450
  },
  "statistics": {
    "count": 48,
    "mean": 1230.5,
    "min": 980.2,
    "max": 1456.8
  },
  "download_url": "/files/{job_id}.xlsx"
}
```

---

## 8. Deployment

### 8.1 AWS Hedef Mimarisi

```
┌─ AWS ───────────────────────────────────┐
│                                          │
│   ECR (Container Registry)               │
│       └── energy-forecast:latest         │
│                                          │
│   ECS / App Runner                       │
│       └── FastAPI container              │
│           ├── /predict endpoint          │
│           ├── /health endpoint           │
│           └── Model files (S3'den yükle) │
│                                          │
│   S3                                     │
│       ├── models/ (.cbm, .pkl, .ckpt)    │
│       ├── reports/                        │
│       └── data/external/                  │
│                                          │
│   RDS (opsiyonel)                        │
│       └── PostgreSQL (MLflow backend)    │
│                                          │
└──────────────────────────────────────────┘
```

### 8.2 Docker

- Multi-stage build: builder (uv install) → runtime (slim)
- Base image: `python:3.11-slim`
- Model dosyaları S3'den runtime'da yüklenir
- Health check: `/health` endpoint

### 8.3 CI/CD

GitHub Actions (.github/workflows/ci.yml):
- Push/PR → ruff check + mypy + pytest (-m "not slow")
- Tag → build Docker image → push ECR → deploy

---

## 9. Konfigürasyon

### 9.1 YAML Dosyaları

```
configs/
├── settings.yaml           # Genel: logging, lokasyon, timezone, paths
├── pipeline.yaml           # Pipeline: hangi modüller aktif, merge stratejisi
├── data_loader.yaml        # Veri yükleme: Excel format, validation
├── openmeteo.yaml          # Hava durumu: lokasyonlar, ağırlıklar, değişkenler
├── api.yaml                # API: CORS, rate limit
├── models/
│   ├── catboost.yaml       # CatBoost: 28 kategorik kolon, eğitim parametreleri
│   ├── prophet.yaml        # Prophet: seasonality, holidays, 14 regressor
│   ├── tft.yaml            # TFT: NeuralForecast architecture, training
│   ├── ensemble.yaml       # Ensemble: ağırlıklar, aktif modeller
│   └── hyperparameters.yaml # Optuna arama uzayı (tüm modeller)
└── features/
    ├── calendar.yaml       # Calendar feature parametreleri
    ├── consumption.yaml    # Consumption: lag değerleri, window boyutları
    ├── epias.yaml          # EPİAŞ: lag değerleri, window boyutları
    ├── solar.yaml          # Solar: lokasyon, panel parametreleri
    └── weather.yaml        # Weather: threshold'lar, window boyutları
```

### 9.2 Environment Variables (.env)

```bash
# API
APP_ENV=production          # development | production
API_HOST=0.0.0.0
API_PORT=8000
API_KEY=secret_key

# EPİAŞ
EPIAS_USERNAME=xxx
EPIAS_PASSWORD=xxx

# Email
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=xxx
SMTP_PASSWORD=xxx

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# AWS
AWS_S3_BUCKET=energy-forecast-models
AWS_REGION=eu-west-1

# Database (opsiyonel)
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db
```

---

## 10. Kalite Standartları

### 10.1 Kod Kalitesi

| Araç | Konfigürasyon |
|------|--------------|
| Formatter | Ruff format (line-length: 100) |
| Linter | Ruff check |
| Type checker | MyPy strict mode |
| Commit format | Conventional Commits (commitizen) |
| Pre-commit | Ruff + MyPy + commitizen + trailing whitespace |
| CI | GitHub Actions (ruff + mypy + pytest) |

### 10.2 Test Gereksinimleri

| Katman | Gereksinim |
|--------|-----------|
| Unit tests | Her public fonksiyon/class test edilmeli |
| Integration tests | End-to-end pipeline (Excel → prediction) |
| Hızlı test | YAML'da değer düşür → run.py çalıştır → geri al |
| Coverage hedefi | %90+ (mevcut: %91) |
| Mock kuralı | Sadece external API'ler mock'lanır (EPİAŞ, OpenMeteo) |
| Test sayısı | 594 (aktif) |

### 10.3 Git Workflow

```
main (protected)
  └── feat/M{N}-{açıklama}  (feature branch per milestone)
       └── Merge via --no-ff
```

Commit format: `feat(scope): description` / `fix(scope): description`
**ASLA commit mesajında AI/Claude referansı YAPMA.**

---

## 11. Teknik Kısıtlar

### 11.1 Performans

| Metrik | Hedef | Mevcut |
|--------|-------|--------|
| Tahmin üretme süresi | < 30 saniye | Ölçülüyor (latency_ms metadata'da) |
| API response time | < 60 saniye | — |
| CatBoost Val MAPE | < %5 | Yeni HPO sonrası güncellenecek |
| Prophet Val MAPE | < %6 | Yeni HPO sonrası güncellenecek |
| Ensemble Val MAPE | < %3 (hedef) | Yeni HPO sonrası güncellenecek |

### 11.2 Güvenilirlik

- Graceful degradation: Ensemble'da bir model fail ederse kalanlarla devam
- API rate limiting: Aynı kullanıcıdan 1 talep/dakika
- Input validation: Pandera schema ile Excel doğrulama (ConsumptionSchema, EpiasSchema, WeatherSchema)
- Error handling: Structured error responses (JSON), JobNotFoundError → 404, Exception → 500
- Model integrity: Prophet pickle SHA256 hash verification

---

## 12. Known Issues

| Sorun | Açıklama | Workaround |
|-------|----------|------------|
| FDPP deprecated | EPIAS API'den artık çekilemiyor | Diğer 4 değişken kullanılıyor |
| EPIAS duplicate timestamps | Yıllık cache dosyalarında duplicate satırlar | `df[~df.index.duplicated(keep='first')]` ile temizle |
| Windows cp1254 codec | Unicode box-drawing karakterler encode edilemiyor | ASCII karakterler kullan |
| Prophet cmdstanpy | Bazı ortamlarda kurulum sorunu | `pip install cmdstanpy` sonra `cmdstanpy.install_cmdstan()` |

---

## 13. Domain Sözlüğü

| Terim | Açıklama |
|-------|----------|
| GOP | Gün Öncesi Planı — T+1 günü tahmini |
| GİP | Gün İçi Planı — T günü tahmini (güncelleme) |
| EPİAŞ | Enerji Piyasaları İşletme A.Ş. |
| EPDK | Enerji Piyasası Düzenleme Kurumu |
| TSCV | Time Series Cross-Validation |
| HDD | Heating Degree Days — ısıtma derece günleri |
| CDD | Cooling Degree Days — soğutma derece günleri |
| GHI | Global Horizontal Irradiance |
| DNI | Direct Normal Irradiance |
| DHI | Diffuse Horizontal Irradiance |
| POA | Plane of Array (güneş paneli yüzeyi ışınımı) |
| WMO 4677 | World Meteorological Organization hava durumu kod standardı |
| FDPP | Final Daily Production Plan (DEPRECATED) |
| Uludağ Bölgesi | Bursa + Balıkesir + Yalova + Çanakkale dağıtım bölgesi |

---

## 14. Referanslar

| Dosya | İçerik |
|-------|--------|
| `CLAUDE.md` | Claude Code context ve kurallar |
| `PROJECT_KNOWLEDGE.md` | Eski projenin detaylı teknik analizi |
| `docs/plans/M{N}.md` | Milestone planları |
| `docs/data-flow.md` | Veri akış haritası (müşteri döngüsü, DB/Drive/dosya) |
| `docs/decisions/ADR-*.md` | Architecture Decision Records |
| Eski proje | `~/projects/distributed-energy-forecasting/` (read-only referans) |