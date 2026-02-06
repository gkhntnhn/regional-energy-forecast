# ADR-001: Data Leakage Audit

> Tarih: 2026-02-07
> Durum: Kabul Edildi
> Kapsam: `src/energy_forecast/features/` altindaki tum .py dosyalari

---

## Ozet

M4 milestone kapsaminda feature engineering pipeline'inin data leakage kurallarina
uygunlugu denetlenmistir. 6 kontrol noktasi incelenmis, **tumu GECMISTIR**.
1 adet bilgilendirme uyarisi tespit edilmistir.

---

## === DATA LEAKAGE AUDIT RAPORU ===

### Taranan Dosyalar

| # | Dosya | Satir |
|---|-------|-------|
| 1 | `src/energy_forecast/features/base.py` | 48 |
| 2 | `src/energy_forecast/features/calendar.py` | 232 |
| 3 | `src/energy_forecast/features/consumption.py` | 143 |
| 4 | `src/energy_forecast/features/custom.py` | 176 |
| 5 | `src/energy_forecast/features/epias.py` | 124 |
| 6 | `src/energy_forecast/features/pipeline.py` | 99 |
| 7 | `src/energy_forecast/features/solar.py` | 88 |
| 8 | `src/energy_forecast/features/weather.py` | 149 |
| 9 | `src/energy_forecast/config/settings.py` | 870 |
| 10 | `configs/features/consumption.yaml` | 50 |
| 11 | `configs/features/epias.yaml` | 37 |
| 12 | `configs/pipeline.yaml` | 14 |

---

### 1. Consumption Lag Kontrolu

**KURAL:** Tum consumption lag feature'larin minimum degeri >= 48 saat olmali.

| Konum | Kontrol | Deger | Sonuc |
|-------|---------|-------|-------|
| `consumption.yaml:4` | `min_lag` config | 48 | OK |
| `consumption.yaml:6-11` | Lag values | [48, 72, 96, 168, 336, 720] | Tumu >= 48 |
| `consumption.py:42` | Runtime `min_lag` | `self.config["lags"]["min_lag"]` (48) | OK |
| `consumption.py:58-66` | LagFeatures `periods` | `lag_values` = [48..720] | OK |
| `consumption.py:77-87` | WindowFeatures `periods` | `min_lag` = 48 | OK |
| `consumption.py:96-107` | ExpandingWindowFeatures `periods` | `min_lag` = 48 | OK |
| `consumption.py:115-121` | EwmaFeatures `periods` | `min_lag` = 48 | OK |
| `consumption.py:125-131` | MomentumFeatures `min_lag` | `min_lag` = 48 | OK |
| `consumption.py:135-141` | QuantileFeatures `periods` | `min_lag` = 48 | OK |
| `settings.py:270` | Pydantic guard | `ge=48` | OK |
| `settings.py:273-279` | Validator `_all_lags_ge_min` | `lag < 48 => raise` | OK |

**Sonuc:** GECTI

---

### 2. EPIAS Lag Kontrolu

**KURAL:** Tum EPIAS lag feature'larin minimum degeri >= 48 saat olmali.

| Konum | Kontrol | Deger | Sonuc |
|-------|---------|-------|-------|
| `epias.yaml:15` | `min_lag` config | 48 | OK |
| `epias.yaml:17-19` | Lag values | [48, 72, 168] | Tumu >= 48 |
| `epias.py:39` | Runtime `min_lag` | `self.config["lags"]["min_lag"]` (48) | OK |
| `epias.py:62-70` | LagFeatures `periods` | `lag_values` = [48, 72, 168] | OK |
| `epias.py:88-98` | WindowFeatures `periods` | `min_lag` = 48 | OK |
| `epias.py:112-122` | ExpandingWindowFeatures `periods` | `min_lag` = 48 | OK |
| `settings.py:441` | Pydantic guard | `ge=48` | OK |
| `settings.py:444-451` | Validator `_all_lags_ge_min` | `lag < 48 => raise` | OK |

**Sonuc:** GECTI

---

### 3. Ham EPIAS Degerleri Drop Kontrolu

**KURAL:** Pipeline cikisinda su kolonlar DROP edilmeli:
FDPP, Real_Time_Consumption, DAM_Purchase, Bilateral_Agreement_Purchase, Load_Forecast

| Konum | Kontrol | Deger | Sonuc |
|-------|---------|-------|-------|
| `epias.py:51-52` | `drop_raw` logic | `df.drop(columns=available_vars)` | OK |
| `epias.yaml:36` | `drop_raw` config | `true` | OK |
| `settings.py:492` | Pydantic default | `drop_raw: bool = True` | OK |
| `pipeline.py:84-89` | Post-pipeline validation | raw EPIAS kolon kontrolu | OK |
| `pipeline.yaml:11` | `drop_raw_epias` config | `true` | OK |

Drop edilen degiskenler (`epias.yaml:7-12`):
- FDPP
- Real_Time_Consumption
- DAM_Purchase
- Bilateral_Agreement_Purchase
- Load_Forecast

**Cift katmanli koruma:**
1. `EpiasFeatureEngineer.transform()` — feature uretim sirasinda drop
2. `FeaturePipeline._validate_output()` — pipeline cikisinda tekrar kontrol

**Sonuc:** GECTI

---

### 4. Rolling Window Kontrolu (Shift Before Roll)

**KURAL:** `.rolling()` oncesinde `.shift()` uygulanmali. `shift` olmadan rolling = o anki gozlemi dahil eder = leakage.

#### feature-engine WindowFeatures (dahili shift)

feature-engine `WindowFeatures(periods=N)` parametresi, rolling window'u N period
geriye kaydirmis veri uzerinde hesaplar. Bu `.shift(N).rolling(window)` ile esdegerdir.

| Konum | `periods` | Aciklama | Sonuc |
|-------|-----------|----------|-------|
| `consumption.py:81` | `min_lag` = 48 | Consumption rolling | OK |
| `epias.py:92` | `min_lag` = 48 | EPIAS rolling | OK |
| `weather.py:74` | `1` | Weather rolling (leakage degil) | OK |

#### feature-engine ExpandingWindowFeatures (dahili shift)

| Konum | `periods` | `min_periods` | Sonuc |
|-------|-----------|---------------|-------|
| `consumption.py:100` | `min_lag` = 48 | 48 | OK |
| `epias.py:117` | `min_lag` = 48 | 48 | OK |

#### Custom transformerlar

| Konum | Yontem | Sonuc |
|-------|--------|-------|
| `custom.py:53` | `.ewm().mean().shift(periods)` — EWMA sonra shift(48) | OK |
| `custom.py:90-95` | `shift(min_lag)` ve `shift(min_lag + period)` — tum degerler shifted | OK |
| `custom.py:133-137` | `shift(periods)` SONRA `.rolling().quantile()` — shift once | OK |

**Not:** `weather.py:128-129` — `temp - temp.shift(3)` ve `temp - temp.shift(24)`:
Weather verisi "known future input" olarak tanimlanmistir (SPEC.md Bolum 4.2).
Tahmin aninda OpenMeteo'dan mevcut oldugu icin leakage DEGILDIR.

**Sonuc:** GECTI

---

### 5. Expanding Window Kontrolu

**KURAL:** `min_periods >= 48` olmali.

| Konum | `min_periods` | Guard | Sonuc |
|-------|---------------|-------|-------|
| `consumption.yaml:34` | 48 | — | OK |
| `consumption.py:98` | `exp_cfg["min_periods"]` = 48 | — | OK |
| `settings.py:299` | `ExpandingConfig.min_periods` | `ge=48` | OK |
| `settings.py:302-308` | Validator `_min_periods_ge_48` | `< 48 => raise` | OK |
| `epias.yaml:32` | 48 | — | OK |
| `epias.py:114` | `exp_cfg["min_periods"]` = 48 | — | OK |
| `settings.py:464` | `EpiasExpandingConfig.min_periods` | `ge=48` | OK |
| `settings.py:467-473` | Validator `_min_periods_ge_48` | `< 48 => raise` | OK |

**Sonuc:** GECTI

---

### 6. Consumption Duplicate / Target Leakage Kontrolu

**KURAL:** Pipeline cikisinda target kolon (`consumption`) feature olarak sizmamali.
Duplicate kolon kontrolu yapilmali.

| Konum | Kontrol | Sonuc |
|-------|---------|-------|
| `pipeline.py:77-81` | `df.columns.duplicated()` kontrolu | OK |
| `pipeline.py:92-94` | DatetimeIndex preserved kontrolu | OK |
| `pipeline.yaml:13` | `check_duplicate_columns: true` | OK |

**Not:** `consumption` kolonu pipeline cikisinda KALIR (`drop_original=False`).
Bu beklenen davranistir — model egitim kodunun X (features) ve y (target=consumption)
ayrimini yapmasi gerekir. Pipeline bu kolonu birakarak target'i korur.

**Sonuc:** GECTI (uyari ile)

---

## Leakage OLMAYAN Durumlar (Dogrulama)

Asagidaki feature'lar "known future inputs" olarak onaylanmistir:

| Feature | Dosya | Neden Leakage Degil |
|---------|-------|---------------------|
| Solar position, GHI/DNI/DHI, POA | `solar.py:48-68` | Astronomik, deterministik |
| Solar lead features (`shift(-h)`) | `solar.py:84-85` | Astronomik, deterministik |
| Solar clearness index, cloud proxy | `solar.py:71-74` | Astronomik, deterministik |
| Solar daylight hours | `solar.py:77-79` | Astronomik, deterministik |
| Weather rolling (`periods=1`) | `weather.py:70-80` | Tahmin aninda OpenMeteo'dan mevcut |
| Weather temp change | `weather.py:128-129` | Tahmin aninda OpenMeteo'dan mevcut |
| Weather extreme flags | `weather.py:87-101` | Tahmin aninda OpenMeteo'dan mevcut |
| Weather HDD/CDD | `weather.py:46-53` | Tahmin aninda OpenMeteo'dan mevcut |
| Calendar features (tumu) | `calendar.py:28-42` | Deterministik, zaman bilgisi |

---

## Sonuc Tablosu

```
=== DATA LEAKAGE AUDIT RAPORU ===
Tarih: 2026-02-07

GECEN KONTROLLER:
- [1] Consumption Lag: Tum lag degerleri >= 48, Pydantic ge=48 guard + validator
- [2] EPIAS Lag: Tum lag degerleri >= 48, Pydantic ge=48 guard + validator
- [3] Ham EPIAS Drop: Cift katmanli koruma (engineer drop + pipeline validate)
- [4] Rolling Window: feature-engine periods= parametresi + custom shift() guvenli
- [5] Expanding Window: min_periods=48 her yerde, Pydantic guard + validator
- [6] Duplicate/Target: Duplicate kolon kontrolu aktif, DatetimeIndex kontrolu aktif

BASARISIZ KONTROLLER:
  (yok)

UYARILAR:
- [6] consumption kolonu pipeline cikisinda kalir (drop_original=False).
  Bu beklenen davranistir — model egitim kodu X/y ayrimini yapmalidir.
  M5 (CatBoost training) milestone'unda bu ayrim dogrulanmalidir.

Toplam: 6 gecen, 0 basarisiz, 1 uyari
```

---

## Koruma Katmanlari Ozeti

Sistem 3 katmanli leakage korumasina sahiptir:

1. **Config katmani** (`settings.py`): Pydantic `ge=48` field constraint + `field_validator`
   ile min_lag ve min_periods degerleri config yukleme sirasinda dogrulanir.

2. **Feature katmani** (`consumption.py`, `epias.py`, `custom.py`): Her transformer
   `periods` / `min_lag` / `shift` parametrelerini config'den alir ve uygular.

3. **Pipeline katmani** (`pipeline.py`): `_validate_output()` ile ham EPIAS kolonlari,
   duplicate kolonlar ve DatetimeIndex son kez kontrol edilir.

---

## Oneriler (M5+ icin)

1. M5 CatBoost training'de `consumption` kolonunun target olarak ayrilip
   feature'lardan cikarildigini dogrulayin.
2. TimeSeriesSplit implementasyonunda `shuffle=False` kontrolunu ekleyin
   (`settings.py:711` — config'de zaten `False`).
3. Egitim sonrasi feature importance raporunda ham veri sizdiran
   suspekt feature'lari kontrol edin.
