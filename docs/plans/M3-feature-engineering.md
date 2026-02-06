# M3: Feature Engineering — Implementation Plan

## 1. Kapsam

**Milestone:** M3 — Feature Engineering (5 modül + pipeline orkestratör)
**Bağımlılık:** M0 ✅, M1 ✅, M2 ✅
**Hedef:** 5 feature engineer modülünü (Calendar, Consumption, Weather, Solar, EPIAS)
implement et. feature-engine kütüphanesi temel transformer olarak kullanılır;
domain-specific hesaplamalar (pvlib solar, HDD/CDD, WMO severity) custom sklearn-uyumlu
transformer'larla yazılır. FeaturePipeline orkestratör ile birleştirilir.

---

## 2. Mevcut Durum

| Dosya | Durum | Satır |
|-------|-------|-------|
| `src/energy_forecast/features/base.py` | Tamamlandı | 48 |
| `src/energy_forecast/features/calendar.py` | Stub (NotImplementedError) | 32 |
| `src/energy_forecast/features/consumption.py` | Stub (NotImplementedError) | 35 |
| `src/energy_forecast/features/weather.py` | Stub (NotImplementedError) | 35 |
| `src/energy_forecast/features/solar.py` | Stub (NotImplementedError) | 35 |
| `src/energy_forecast/features/epias.py` | Stub (NotImplementedError) | 35 |
| `src/energy_forecast/features/pipeline.py` | Stub (NotImplementedError) | 33 |
| `src/energy_forecast/features/__init__.py` | BaseFeatureEngineer + FeaturePipeline export | 7 |
| `configs/features/*.yaml` | Tüm 5 feature config tamamlandı | ~95 |
| `configs/pipeline.yaml` | Tamamlandı | ~8 |
| `tests/unit/test_features/__init__.py` | Boş (M3 için hazır) | 0 |
| `src/energy_forecast/config/settings.py` | Tüm feature Pydantic modelleri mevcut | 856 |

---

## 3. Referans Proje Analizi

Eski proje: `C:\Users\pc\Desktop\distributed-energy-forecasting\`

### 3.1 Kaynak → Hedef Eşleme

| Eski Dosya | Eski Satır | Yeni Dosya | Hedef Satır | Not |
|------------|-----------|------------|-------------|-----|
| `scripts/calendar_features.py` | 1124 | `features/calendar.py` | ~280 | DatetimeFeatures + CyclicalFeatures ile sadeleşir |
| `scripts/consumption_features.py` | 1306 | `features/consumption.py` | ~250 | LagFeatures + WindowFeatures + ExpandingWindowFeatures |
| `scripts/weather_features.py` | 808 | `features/weather.py` | ~250 | WindowFeatures + custom HDD/CDD/severity |
| `scripts/solar_features.py` | 1195 | `features/solar.py` | ~300 | Tamamen custom (pvlib) |
| `scripts/epias_features.py` | 557 | `features/epias.py` | ~180 | LagFeatures + WindowFeatures + ExpandingWindowFeatures |
| `src/data/preprocessing.py` | 856 | `features/pipeline.py` | ~180 | Orkestratör |
| — | — | `features/custom.py` | ~200 | EWMA, momentum, quantile, HDD/CDD custom transformers |
| **Toplam** | **5846** | | **~1640** | **%72 kod azalması** |

### 3.2 feature-engine vs Custom Karar Matrisi

| İşlem | feature-engine Transformer | Custom Gerekiyor mu? |
|-------|--------------------------|---------------------|
| Datetime extraction (hour, dow, month...) | `DatetimeFeatures` | Hayır |
| Cyclical sin/cos encoding | `CyclicalFeatures` | Hayır |
| Consumption lag (shift 48+) | `LagFeatures(periods=[48,72,...])` | Hayır |
| Consumption rolling window | `WindowFeatures(window=W, periods=48)` | Hayır |
| Consumption expanding | `ExpandingWindowFeatures(periods=48, min_periods=48)` | Hayır |
| Consumption EWMA | — | **Evet** (EwmaFeatures) |
| Consumption momentum | — | **Evet** (MomentumFeatures) |
| Consumption quantile | — | **Evet** (QuantileFeatures) |
| Weather rolling | `WindowFeatures(window=[6,12,24])` | Hayır |
| HDD/CDD | — | **Evet** (DegreeDayFeatures) |
| Weather extreme flags | — | **Evet** (inline in weather.py) |
| Weather severity (WMO) | — | **Evet** (inline in weather.py) |
| Solar (pvlib tümü) | — | **Evet** (SolarFeatureEngineer) |
| EPIAS lag | `LagFeatures(periods=[48,72,168])` | Hayır |
| EPIAS rolling | `WindowFeatures(window=W, periods=48)` | Hayır |
| EPIAS expanding | `ExpandingWindowFeatures(periods=48)` | Hayır |
| EPIAS raw drop | — | **Evet** (inline in epias.py) |
| Holiday/Ramazan/bridge | — | **Evet** (inline in calendar.py) |
| Business hours/season | — | **Evet** (inline in calendar.py) |

### 3.3 Taşınmayacak / Sadeleştirilecek

- ✗ Manual pandas shift/rolling → **feature-engine LagFeatures/WindowFeatures**
- ✗ 100+ feature per module → **Config-driven essential features (~30-50 per module)**
- ✗ Memory optimization (float32) → **Gereksiz**
- ✗ WindowFeatures wrapper class (eski proje) → **feature-engine native**
- ✗ Advanced volatility (MAD, IQR, stability) → **M5'te Optuna karar verir**
- ✗ Distribution features (skew, kurtosis) → **CatBoost otomatik öğrenir**
- ✗ Autocorrelation, regime detection → **M5'te değerlendirilir**
- ✗ WMO 93-satır mapping → **4-level severity grouping yeterli**

---

## 4. feature-engine API Özeti

### 4.1 LagFeatures

```python
from feature_engine.timeseries.forecasting import LagFeatures

lag = LagFeatures(
    variables=["consumption"],       # Hangi kolonlar
    periods=[48, 72, 96, 168],       # shift(N) — tümü >= min_lag
    freq=None,                       # Alternatif: "48h" gibi
    sort_index=True,                 # DatetimeIndex sıralansın
    missing_values="raise",          # NaN varsa hata
    drop_original=False,             # Orijinal kolonu koru
    drop_na=False,                   # NaN satırları silme
)
# Çıktı kolonları: consumption_lag_48, consumption_lag_72, ...
```

### 4.2 WindowFeatures

```python
from feature_engine.timeseries.forecasting import WindowFeatures

win = WindowFeatures(
    variables=["consumption"],
    window=24,                       # Rolling window boyutu
    min_periods=None,                # pandas min_periods
    functions=["mean", "std"],       # Aggregation fonksiyonları
    periods=48,                      # shift(48) — LEAKAGE KORUMA
    freq=None,
    sort_index=True,
    missing_values="raise",
    drop_original=False,
    drop_na=False,
)
# İşlem: series.rolling(24).mean().shift(48)
# Çıktı: consumption_window_24_mean, consumption_window_24_std
```

**Leakage güvenliği:** `periods=48` sayesinde rolling sonucu 48 saat ileri kaydırılır.
Time t'de sadece t-48 ve öncesinin istatistiklerini görür. ✅

### 4.3 ExpandingWindowFeatures

```python
from feature_engine.timeseries.forecasting import ExpandingWindowFeatures

exp = ExpandingWindowFeatures(
    variables=["consumption"],
    min_periods=48,                  # >= 48 enforced
    functions=["mean", "std"],
    periods=48,                      # shift(48)
    sort_index=True,
    missing_values="raise",
    drop_original=False,
    drop_na=False,
)
# Çıktı: consumption_expanding_mean, consumption_expanding_std
```

### 4.4 DatetimeFeatures

```python
from feature_engine.datetime import DatetimeFeatures

dt = DatetimeFeatures(
    variables="index",               # DatetimeIndex'ten çıkar
    features_to_extract=[            # İstenen bileşenler
        "hour", "day_of_week", "day_of_month",
        "day_of_year", "week", "month", "quarter", "year",
    ],
    drop_original=False,
)
# Çıktı: hour, day_of_week, day_of_month, ...
```

### 4.5 CyclicalFeatures

```python
from feature_engine.creation import CyclicalFeatures

cyc = CyclicalFeatures(
    variables=["hour", "day_of_week", "month", "day_of_year"],
    max_values={
        "hour": 24,
        "day_of_week": 7,
        "month": 12,
        "day_of_year": 365,
    },
    drop_original=False,
)
# Formül: sin(2π × x / max_value), cos(2π × x / max_value)
# Çıktı: hour_sin, hour_cos, day_of_week_sin, ...
```

---

## 5. Implementation Detayları

### 5.1 `src/energy_forecast/features/base.py` — DEĞİŞMEZ

```python
class BaseFeatureEngineer(ABC, BaseEstimator, TransformerMixin):
    def __init__(self, config: dict[str, Any]) -> None
    def fit(self, X, y=None) -> self
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame
```

### 5.2 `src/energy_forecast/features/custom.py` — Yeni Dosya (~200 satır)

feature-engine'de karşılığı olmayan domain-specific custom transformer'lar.
Hepsi `BaseEstimator, TransformerMixin` ile sklearn-uyumlu.

```python
class EwmaFeatures(BaseEstimator, TransformerMixin):
    """EWMA features with min_lag shift. feature-engine'de yok.

    Args:
        variables: Lag uygulanacak kolonlar.
        spans: EWMA span değerleri (config: [24, 48, 168]).
        periods: Shift miktarı (min_lag=48).

    İşlem: series.ewm(span=S, adjust=False).mean().shift(periods)
    Çıktı: {var}_ewma_{span}
    """

class MomentumFeatures(BaseEstimator, TransformerMixin):
    """Momentum (velocity) ve pct_change features.

    Args:
        variables: Kaynak kolon.
        min_lag: Minimum lag (48).
        momentum_periods: Momentum periyotları (config: [24, 168]).

    Formüller:
        momentum_P = series.shift(min_lag) - series.shift(min_lag + P)
        pct_change_P = momentum_P / series.shift(min_lag + P) * 100
    Çıktı: {var}_momentum_{P}, {var}_pct_change_{P}
    """

class QuantileFeatures(BaseEstimator, TransformerMixin):
    """Rolling quantile features with min_lag shift.

    Args:
        variables: Kaynak kolon.
        quantiles: Quantile değerleri (config: [0.25, 0.50, 0.75]).
        window: Rolling window boyutu (config: 168).
        periods: Shift miktarı (min_lag=48).

    İşlem: series.shift(periods).rolling(window).quantile(q)
    Çıktı: {var}_q{int(q*100)}_{window}
    """

class DegreeDayFeatures(BaseEstimator, TransformerMixin):
    """HDD/CDD (Heating/Cooling Degree Days) hesaplaması.

    Args:
        temp_variable: Sıcaklık kolonu adı (default: "temperature_2m").
        hdd_base: HDD baz sıcaklığı (config: 18.0).
        cdd_base: CDD baz sıcaklığı (config: 24.0).

    Formüller:
        HDD = max(hdd_base - temperature, 0)
        CDD = max(temperature - cdd_base, 0)
    Çıktı: wth_hdd, wth_cdd
    """
```

Tüm parametreler constructor'da alınır, config'ten okunur. Hardcoded değer YOK.

### 5.3 `src/energy_forecast/features/calendar.py` — CalendarFeatureEngineer (~280 satır)

**Input:** DatetimeIndex'li DataFrame
**Output:** Aynı DataFrame + calendar feature kolonları

**Kullanılan feature-engine transformer'lar:**
- `DatetimeFeatures(variables="index", features_to_extract=[...])` → datetime bileşenleri
- `CyclicalFeatures(variables=[...], max_values={...})` → sin/cos encoding

**Custom (inline) logic:**
- Holiday features (parquet'tan oku, is_holiday, proximity)
- Ramadan flags
- Bridge days
- Business hours, peak, weekend
- Seasonal flags (heating/cooling season)

```python
class CalendarFeatureEngineer(BaseFeatureEngineer):
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # 1. DatetimeFeatures — feature-engine
        dt_features = self._extract_datetime(df)
        extract_list = self.config["datetime"]["extract"]
        dt_transformer = DatetimeFeatures(
            variables="index",
            features_to_extract=extract_list,  # Config'ten
            drop_original=False,
        )
        df = dt_transformer.fit_transform(df)

        # 2. CyclicalFeatures — feature-engine
        cyclical_cfg = self.config["cyclical"]
        cyc_vars = list(cyclical_cfg.keys())
        max_vals = {k: v["period"] for k, v in cyclical_cfg.items()}
        cyc_transformer = CyclicalFeatures(
            variables=cyc_vars,
            max_values=max_vals,
            drop_original=False,
        )
        df = cyc_transformer.fit_transform(df)

        # 3. Custom — domain-specific
        df = self._add_holiday_features(df)      # parquet read + merge
        df = self._add_business_features(df)      # is_weekend, is_peak, season
        return df
```

**Feature çıktısı (~33 feature):**

| Grup | Kaynak | Feature'lar | Sayı |
|------|--------|-------------|------|
| Datetime | DatetimeFeatures | hour, day_of_week, day_of_month, day_of_year, week, month, quarter, year | 8 |
| Cyclical | CyclicalFeatures | hour_sin/cos, day_of_week_sin/cos, month_sin/cos, day_of_year_sin/cos | 8 |
| Holiday | Custom | is_holiday, holiday_name, is_ramadan, days_until_holiday, days_since_holiday, is_bridge_day | 6 |
| Business | Custom | is_weekend, is_monday, is_friday, is_business_hours, is_peak, season, is_heating_season, is_cooling_season | 8 |
| **Toplam** | | | **~30** |

**Private methods:**
```python
def _add_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame
def _add_business_features(self, df: pd.DataFrame) -> pd.DataFrame
def _load_holidays(self) -> pd.DataFrame  # parquet read with graceful fallback
```

### 5.4 `src/energy_forecast/features/consumption.py` — ConsumptionFeatureEngineer (~250 satır)

**Input:** DatetimeIndex + `consumption` kolonu
**Output:** Aynı DataFrame + consumption feature kolonları

**LEAKAGE KURALLARI:**
- LagFeatures: `periods=[48,72,...] `— tümü >= 48
- WindowFeatures: `periods=48` — shift AFTER roll (matematiksel eşdeğer)
- ExpandingWindowFeatures: `periods=48, min_periods=48`
- Custom EWMA/momentum/quantile: shift(min_lag) uygulanır

**Kullanılan feature-engine transformer'lar:**
- `LagFeatures(variables=["consumption"], periods=[48,72,96,168,336,720])`
- `WindowFeatures(variables=["consumption"], window=W, periods=48, functions=[...])`
  → Her window değeri (24,48,168,336,720) için ayrı instance
- `ExpandingWindowFeatures(variables=["consumption"], periods=48, min_periods=48)`

**Custom transformer'lar (custom.py'den):**
- `EwmaFeatures(variables=["consumption"], spans=[24,48,168], periods=48)`
- `MomentumFeatures(variables=["consumption"], min_lag=48, momentum_periods=[24,168])`
- `QuantileFeatures(variables=["consumption"], quantiles=[0.25,0.50,0.75], window=168, periods=48)`

```python
class ConsumptionFeatureEngineer(BaseFeatureEngineer):
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        min_lag = self.config["lags"]["min_lag"]  # 48

        # 1. LagFeatures — feature-engine
        lag = LagFeatures(
            variables=["consumption"],
            periods=self.config["lags"]["values"],  # [48,72,96,168,336,720]
            sort_index=True,
            missing_values="ignore",
            drop_original=False,
        )
        df = lag.fit_transform(df)

        # 2. WindowFeatures — feature-engine (per window size)
        for w in self.config["rolling"]["windows"]:  # [24,48,168,336,720]
            win = WindowFeatures(
                variables=["consumption"],
                window=w,
                functions=self.config["rolling"]["functions"],  # [mean,std,min,max]
                periods=min_lag,  # 48 — LEAKAGE KORUMA
                sort_index=True,
                missing_values="ignore",
                drop_original=False,
            )
            df = win.fit_transform(df)

        # 3. ExpandingWindowFeatures — feature-engine
        exp = ExpandingWindowFeatures(
            variables=["consumption"],
            min_periods=self.config["expanding"]["min_periods"],  # 48
            functions=self.config["expanding"]["functions"],  # [mean, std]
            periods=min_lag,  # 48
            sort_index=True,
            missing_values="ignore",
            drop_original=False,
        )
        df = exp.fit_transform(df)

        # 4. Custom — EWMA, momentum, quantile
        ewma = EwmaFeatures(
            variables=["consumption"],
            spans=self.config["ewma"]["spans"],  # [24,48,168]
            periods=min_lag,
        )
        df = ewma.fit_transform(df)

        momentum = MomentumFeatures(
            variables=["consumption"],
            min_lag=min_lag,
            momentum_periods=self.config["momentum"]["periods"],  # [24,168]
        )
        df = momentum.fit_transform(df)

        quantile = QuantileFeatures(
            variables=["consumption"],
            quantiles=self.config["quantile"]["quantiles"],  # [0.25,0.50,0.75]
            window=self.config["quantile"]["window"],  # 168
            periods=min_lag,
        )
        df = quantile.fit_transform(df)

        return df
```

**Feature çıktısı (~38 feature):**

| Grup | Kaynak | Çıktı Pattern | Sayı |
|------|--------|---------------|------|
| Lag | LagFeatures | `consumption_lag_{48,72,96,168,336,720}` | 6 |
| Rolling | WindowFeatures | `consumption_window_{W}_{func}` (5 window × 4 func) | 20 |
| Expanding | ExpandingWindowFeatures | `consumption_expanding_{func}` | 2 |
| EWMA | EwmaFeatures | `consumption_ewma_{24,48,168}` | 3 |
| Momentum | MomentumFeatures | `consumption_momentum_{24,168}`, `consumption_pct_change_{24,168}` | 4 |
| Quantile | QuantileFeatures | `consumption_q{25,50,75}_168` | 3 |
| **Toplam** | | | **~38** |

### 5.5 `src/energy_forecast/features/weather.py` — WeatherFeatureEngineer (~250 satır)

**Input:** DatetimeIndex + 11 weather kolonu
**Output:** Aynı DataFrame + weather feature kolonları

**NOT:** Weather features LEAKAGE DEĞİLDİR.
Weather rolling'de `periods=1` (default) yeterli — gelecek veriye erişim yok.

**Kullanılan feature-engine transformer'lar:**
- `WindowFeatures(variables=weather_vars, window=[6,12,24], functions=[mean,min,max])`

**Custom (inline + custom.py):**
- `DegreeDayFeatures(temp_variable, hdd_base, cdd_base)` → HDD/CDD
- Extreme flags (binary threshold comparison)
- Severity mapping (WMO code → 0-3 scale)
- Temperature change (diff)

```python
class WeatherFeatureEngineer(BaseFeatureEngineer):
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # 1. DegreeDayFeatures — custom transformer
        dd = DegreeDayFeatures(
            temp_variable="temperature_2m",
            hdd_base=self.config["thresholds"]["hdd_base"],  # 18.0
            cdd_base=self.config["thresholds"]["cdd_base"],  # 24.0
        )
        df = dd.fit_transform(df)

        # 2. WindowFeatures — feature-engine (weather rolling)
        weather_roll_vars = ["temperature_2m", "wth_hdd", "wth_cdd"]
        for w in self.config["rolling"]["windows"]:  # [6, 12, 24]
            win = WindowFeatures(
                variables=weather_roll_vars,
                window=w,
                functions=self.config["rolling"]["functions"],  # [mean,min,max]
                periods=1,  # Default — weather leakage değil
                sort_index=True,
                missing_values="ignore",
                drop_original=False,
            )
            df = win.fit_transform(df)

        # 3. Custom — extreme flags, severity, temp change
        df = self._add_extreme_flags(df)
        df = self._add_severity(df)
        df = self._add_temp_change(df)
        return df
```

**Feature çıktısı (~24 feature):**

| Grup | Kaynak | Çıktı | Sayı |
|------|--------|-------|------|
| Degree days | DegreeDayFeatures | `wth_hdd`, `wth_cdd` | 2 |
| Rolling | WindowFeatures | `temperature_2m_window_{W}_{func}`, `wth_hdd_window_{W}_{func}`, `wth_cdd_window_{W}_{func}` | 27 (3 var × 3 win × 3 func) |
| Extremes | Custom | `wth_extreme_cold`, `wth_extreme_hot`, `wth_extreme_wind`, `wth_heavy_precip` | 4 |
| Severity | Custom | `wth_severity`, `wth_is_severe` | 2 |
| Temp change | Custom | `wth_temp_change_3h`, `wth_temp_change_24h` | 2 |
| **Toplam (rolling dahil kısıtlama ile)** | | | **~20-25** |

Rolling variable listesi config ile daraltılabilir. İlk fazda sadece `temperature_2m`
rolling yeterli olabilir — 3 window × 3 func = 9 feature. Gerekirse HDD/CDD rolling eklenir.

**Private methods:**
```python
def _add_extreme_flags(self, df: pd.DataFrame) -> pd.DataFrame
def _add_severity(self, df: pd.DataFrame) -> pd.DataFrame
def _add_temp_change(self, df: pd.DataFrame) -> pd.DataFrame
```

### 5.6 `src/energy_forecast/features/solar.py` — SolarFeatureEngineer (~300 satır)

**Input:** DatetimeIndex'li DataFrame
**Output:** Aynı DataFrame + solar feature kolonları

**NOT:** Solar features LEAKAGE DEĞİLDİR — deterministik astronomik hesaplama.

**Tamamen custom** — pvlib API'si feature-engine ile entegre değil.
Tüm parametreler config'ten okunur (`SolarConfig`).

```python
class SolarFeatureEngineer(BaseFeatureEngineer):
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        loc_cfg = self.config["location"]
        panel_cfg = self.config["panel"]

        # pvlib Location
        location = pvlib.location.Location(
            latitude=loc_cfg["latitude"],     # config: 40.183
            longitude=loc_cfg["longitude"],   # config: 29.050
            tz=loc_cfg["timezone"],           # config: Europe/Istanbul
            altitude=loc_cfg["altitude"],     # config: 100
        )

        # tz-localize (pvlib requires tz-aware index)
        times = df.index.tz_localize(loc_cfg["timezone"])

        # 1. Solar position
        solar_pos = location.get_solarposition(times)
        df["sol_elevation"] = solar_pos["apparent_elevation"].values
        df["sol_azimuth"] = solar_pos["azimuth"].values

        # 2. Clear sky irradiance
        clearsky = location.get_clearsky(times)
        df["sol_ghi"] = clearsky["ghi"].values
        df["sol_dni"] = clearsky["dni"].values
        df["sol_dhi"] = clearsky["dhi"].values

        # 3. POA irradiance (tilted panel)
        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=panel_cfg["tilt"],       # config: 35
            surface_azimuth=panel_cfg["azimuth"], # config: 180
            solar_zenith=solar_pos["apparent_zenith"],
            solar_azimuth=solar_pos["azimuth"],
            dni=clearsky["dni"], ghi=clearsky["ghi"], dhi=clearsky["dhi"],
        )
        df["sol_poa_global"] = poa["poa_global"].values

        # 4. Clearness index & cloud proxy
        extra = pvlib.irradiance.get_extra_radiation(times)
        df["sol_clearness_index"] = (clearsky["ghi"] / extra).clip(0, 1).values
        df["sol_cloud_proxy"] = (1 - df["sol_clearness_index"]).clip(0, 1)

        # 5. Daylight
        df["sol_is_daylight"] = (df["sol_elevation"] > 0).astype(int)
        df["sol_daylight_hours"] = (
            df["sol_is_daylight"].resample("D").sum().reindex(df.index, method="ffill")
        )

        # 6. Lead features (NOT leakage — deterministic)
        if self.config["lead"]["enabled"]:
            for h in self.config["lead"]["hours"]:  # config: [1, 2, 3]
                df[f"sol_ghi_lead_{h}"] = df["sol_ghi"].shift(-h)

        # Filter to config feature list
        return self._filter_features(df)
```

**Feature çıktısı (~13 feature):**

| Grup | Kaynak | Feature'lar | Sayı |
|------|--------|-------------|------|
| Position | pvlib | sol_elevation, sol_azimuth | 2 |
| Irradiance | pvlib | sol_ghi, sol_dni, sol_dhi | 3 |
| POA | pvlib | sol_poa_global | 1 |
| Clearness | pvlib | sol_clearness_index, sol_cloud_proxy | 2 |
| Daylight | Custom | sol_is_daylight, sol_daylight_hours | 2 |
| Lead | Custom | sol_ghi_lead_1, sol_ghi_lead_2, sol_ghi_lead_3 | 3 |
| **Toplam** | | | **~13** |

**Private methods:**
```python
def _filter_features(self, df: pd.DataFrame) -> pd.DataFrame
    # config["features"] listesinde olmayan sol_ kolonları drop
```

### 5.7 `src/energy_forecast/features/epias.py` — EpiasFeatureEngineer (~180 satır)

**Input:** DatetimeIndex + 5 EPIAS kolonu
**Output:** Aynı DataFrame + derived features (**RAW KOLONLAR DROP**)

**LEAKAGE KURALLARI:**
- LagFeatures: `periods=[48,72,168]` — tümü >= 48
- WindowFeatures: `periods=48`
- ExpandingWindowFeatures: `periods=48, min_periods=48`
- Raw values DROP: `drop_raw=True` config flag

**Kullanılan feature-engine transformer'lar:**
- `LagFeatures(variables=epias_vars, periods=[48,72,168])`
- `WindowFeatures(variables=epias_vars, window=W, periods=48, functions=[...])`
- `ExpandingWindowFeatures(variables=epias_vars, periods=48, min_periods=48)`

```python
class EpiasFeatureEngineer(BaseFeatureEngineer):
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        raw_vars = self.config["variables"]  # [FDPP, RTC, DAM, Bilateral, Forecast]
        min_lag = self.config["lags"]["min_lag"]  # 48

        # Mevcut EPIAS kolonlarını filtrele
        available_vars = [v for v in raw_vars if v in df.columns]
        if not available_vars:
            logger.warning("No EPIAS columns found, skipping")
            return df

        # 1. LagFeatures — feature-engine
        lag = LagFeatures(
            variables=available_vars,
            periods=self.config["lags"]["values"],  # [48, 72, 168]
            sort_index=True,
            missing_values="ignore",
            drop_original=False,
        )
        df = lag.fit_transform(df)

        # 2. WindowFeatures — feature-engine
        for w in self.config["rolling"]["windows"]:  # [24, 48, 168]
            win = WindowFeatures(
                variables=available_vars,
                window=w,
                functions=self.config["rolling"]["functions"],  # [mean, std]
                periods=min_lag,
                sort_index=True,
                missing_values="ignore",
                drop_original=False,
            )
            df = win.fit_transform(df)

        # 3. ExpandingWindowFeatures — feature-engine
        exp = ExpandingWindowFeatures(
            variables=available_vars,
            min_periods=self.config["expanding"]["min_periods"],  # 48
            functions=self.config["expanding"]["functions"],  # [mean]
            periods=min_lag,
            sort_index=True,
            missing_values="ignore",
            drop_original=False,
        )
        df = exp.fit_transform(df)

        # 4. Drop raw columns
        if self.config.get("drop_raw", True):
            df = df.drop(columns=available_vars, errors="ignore")

        return df
```

**Feature çıktısı:**

| Grup | Kaynak | Pattern | Sayı |
|------|--------|---------|------|
| Lag | LagFeatures | `{VAR}_lag_{48,72,168}` | 15 (5 var × 3 lag) |
| Rolling | WindowFeatures | `{VAR}_window_{W}_{func}` | 30 (5 var × 3 win × 2 func) |
| Expanding | ExpandingWindowFeatures | `{VAR}_expanding_mean` | 5 (5 var × 1 func) |
| **Toplam derived** | | | **~50** |
| Raw DROP | | FDPP, RTC, DAM, Bilateral, Forecast | **-5** |

### 5.8 `src/energy_forecast/features/pipeline.py` — FeaturePipeline (~180 satır)

```python
class FeaturePipeline:
    """Tüm feature engineering modüllerini orkestre eder."""

    MODULE_MAP: ClassVar[dict[str, type[BaseFeatureEngineer]]] = {
        "calendar": CalendarFeatureEngineer,
        "consumption": ConsumptionFeatureEngineer,
        "weather": WeatherFeatureEngineer,
        "solar": SolarFeatureEngineer,
        "epias": EpiasFeatureEngineer,
    }

    def __init__(self, config: Settings) -> None:
        self._settings = config
        self._pipeline_config = config.pipeline
        self._engineers: list[tuple[str, BaseFeatureEngineer]] = []
        self._build_engineers()

    def _build_engineers(self) -> None:
        """Config'teki modül listesinden engineer'ları oluştur."""
        for module_name in self._pipeline_config.modules:
            if module_name not in self.MODULE_MAP:
                msg = f"Unknown feature module: {module_name}"
                raise ValueError(msg)
            engineer_cls = self.MODULE_MAP[module_name]
            feature_config = getattr(self._settings.features, module_name)
            engineer = engineer_cls(feature_config.model_dump())
            self._engineers.append((module_name, engineer))

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pipeline'ı çalıştır.

        Sıra:
            1. Input'u kopyala
            2. Her engineer için: df = engineer.fit_transform(df)
            3. Output'u doğrula
            4. Return
        """
        result = df.copy()
        for name, engineer in self._engineers:
            logger.info("Running {} feature engineer", name)
            result = engineer.fit_transform(result)
            logger.info("{} complete, shape: {}", name, result.shape)

        self._validate_output(result)
        return result

    def _validate_output(self, df: pd.DataFrame) -> None:
        """Post-pipeline doğrulama.

        Kontroller:
            - Duplicate kolon yok
            - Raw EPIAS kolonu yok (drop_raw_epias=True ise)
            - DatetimeIndex korunmuş
        """

    def get_feature_names(self) -> list[str]:
        """Tüm üretilen feature kolon isimlerini döndür."""
```

### 5.9 `src/energy_forecast/features/__init__.py` — Güncelleme

```python
from energy_forecast.features.base import BaseFeatureEngineer
from energy_forecast.features.calendar import CalendarFeatureEngineer
from energy_forecast.features.consumption import ConsumptionFeatureEngineer
from energy_forecast.features.custom import (
    DegreeDayFeatures,
    EwmaFeatures,
    MomentumFeatures,
    QuantileFeatures,
)
from energy_forecast.features.epias import EpiasFeatureEngineer
from energy_forecast.features.pipeline import FeaturePipeline
from energy_forecast.features.solar import SolarFeatureEngineer
from energy_forecast.features.weather import WeatherFeatureEngineer

__all__ = [
    "BaseFeatureEngineer",
    "CalendarFeatureEngineer",
    "ConsumptionFeatureEngineer",
    "DegreeDayFeatures",
    "EpiasFeatureEngineer",
    "EwmaFeatures",
    "FeaturePipeline",
    "MomentumFeatures",
    "QuantileFeatures",
    "SolarFeatureEngineer",
    "WeatherFeatureEngineer",
]
```

---

## 6. Dosya Planı

### 6.1 Düzenlenecek Dosyalar

| Dosya | Mevcut → Hedef | Değişiklik |
|-------|----------------|------------|
| `src/energy_forecast/features/calendar.py` | 32 → ~280 | Tam implementasyon |
| `src/energy_forecast/features/consumption.py` | 35 → ~250 | Tam implementasyon |
| `src/energy_forecast/features/weather.py` | 35 → ~250 | Tam implementasyon |
| `src/energy_forecast/features/solar.py` | 35 → ~300 | Tam implementasyon |
| `src/energy_forecast/features/epias.py` | 35 → ~180 | Tam implementasyon |
| `src/energy_forecast/features/pipeline.py` | 33 → ~180 | Tam implementasyon |
| `src/energy_forecast/features/__init__.py` | 7 → ~25 | Tüm exports |
| `tests/conftest.py` | 151 → ~280 | Feature test fixtures |
| `pyproject.toml` | dependencies | `feature-engine>=1.8`, `openmeteo-requests>=1.3`, `requests-cache>=1.2`, `retry-requests>=2.0` ekle |
| `src/energy_forecast/data/openmeteo_client.py` | 317 → ~250 | SDK refactor (httpx→openmeteo-requests) |
| `src/energy_forecast/config/settings.py` | 856 → ~900 | OpenMeteoConfig + GeocodingConfig güncellemeleri |
| `configs/openmeteo.yaml` | 25 → ~35 | SDK uyumlu config + geocoding + historical_forecast endpoint |
| `tests/unit/test_data/test_openmeteo_client.py` | 244 → ~300 | SDK testleri (yeniden yazım) |

### 6.2 Yeni Oluşturulacak Dosyalar

| Dosya | Tahmini Boyut | İçerik |
|-------|--------------|--------|
| `src/energy_forecast/features/custom.py` | ~200 | EwmaFeatures, MomentumFeatures, QuantileFeatures, DegreeDayFeatures |
| `tests/unit/test_features/test_calendar.py` | ~250 | CalendarFeatureEngineer testleri |
| `tests/unit/test_features/test_consumption.py` | ~350 | ConsumptionFeatureEngineer testleri |
| `tests/unit/test_features/test_weather.py` | ~250 | WeatherFeatureEngineer testleri |
| `tests/unit/test_features/test_solar.py` | ~250 | SolarFeatureEngineer testleri |
| `tests/unit/test_features/test_epias.py` | ~250 | EpiasFeatureEngineer testleri |
| `tests/unit/test_features/test_pipeline.py` | ~200 | FeaturePipeline testleri |
| `tests/unit/test_features/test_custom.py` | ~200 | Custom transformer testleri |

---

## 7. Test Stratejisi

### 7.1 Yeni Fixtures (`tests/conftest.py`)

```python
@pytest.fixture()
def sample_consumption_df() -> pd.DataFrame:
    """30 gün × 24 saat = 720 satır, consumption + DatetimeIndex.
    Lag ve rolling feature'lar için yeterli uzunluk.
    """

@pytest.fixture()
def sample_weather_df() -> pd.DataFrame:
    """7 gün × 24 saat = 168 satır, 11 weather kolon + DatetimeIndex."""

@pytest.fixture()
def sample_epias_df() -> pd.DataFrame:
    """30 gün × 24 saat = 720 satır, 5 EPIAS kolon + DatetimeIndex."""

@pytest.fixture()
def sample_full_df() -> pd.DataFrame:
    """consumption + 11 weather + 5 EPIAS birleşik. Pipeline testleri için."""

@pytest.fixture()
def calendar_config(default_settings: Settings) -> dict[str, Any]:
    return default_settings.features.calendar.model_dump()

# consumption_config, weather_config, solar_config, epias_config aynı pattern
```

### 7.2 `test_custom.py` (~10 test)

| # | Test | Tip |
|---|------|-----|
| 1 | `test_ewma_shift_applied` | EWMA sonucu min_lag kadar shift edilmiş |
| 2 | `test_ewma_spans_from_config` | Config'teki span değerleri kullanılır |
| 3 | `test_ewma_column_naming` | `{var}_ewma_{span}` pattern |
| 4 | `test_momentum_calculation` | lag48 - lag72 == momentum_24 |
| 5 | `test_pct_change_calculation` | momentum / previous × 100 |
| 6 | `test_quantile_shift_applied` | Quantile sonucu shift(min_lag) |
| 7 | `test_quantile_range` | 0 ≤ q25 ≤ q50 ≤ q75 |
| 8 | `test_degree_day_hdd` | T=10°C, base=18 → HDD=8 |
| 9 | `test_degree_day_cdd` | T=30°C, base=24 → CDD=6 |
| 10 | `test_degree_day_zero` | T=20°C → HDD=0, CDD=0 |

### 7.3 `test_calendar.py` (~15 test)

| # | Test | Tip |
|---|------|-----|
| 1 | `test_datetime_features_extracted` | hour, dow, month vb. eklendi |
| 2 | `test_cyclical_sin_cos_range` | [-1, 1] arasında |
| 3 | `test_cyclical_hour_midnight` | hour=0 → sin=0, cos=1 |
| 4 | `test_cyclical_hour_noon` | hour=12 → sin≈0, cos=-1 |
| 5 | `test_holiday_flag_known_date` | 2024-01-01 → is_holiday=1 |
| 6 | `test_holiday_flag_normal_day` | 2024-01-02 → is_holiday=0 |
| 7 | `test_holiday_file_missing_graceful` | Dosya yoksa → warning, all 0 |
| 8 | `test_is_weekend_saturday` | Cumartesi → 1 |
| 9 | `test_is_weekend_monday` | Pazartesi → 0 |
| 10 | `test_business_hours_within` | saat 10 → 1 |
| 11 | `test_business_hours_outside` | saat 22 → 0 |
| 12 | `test_peak_hours` | saat 18 → 1 |
| 13 | `test_season_winter` | Ocak → heating=1 |
| 14 | `test_season_summer` | Temmuz → cooling=1 |
| 15 | `test_config_driven_extraction` | Custom features_to_extract → subset |

### 7.4 `test_consumption.py` (~18 test)

| # | Test | Tip |
|---|------|-----|
| 1 | `test_lag_features_created` | consumption_lag_48 vb. mevcut |
| 2 | `test_lag_values_correct` | lag_48 == consumption.shift(48) |
| 3 | `test_lag_min_lag_enforced` | Tüm lag'lar >= 48 |
| 4 | `test_window_features_created` | consumption_window_24_mean mevcut |
| 5 | `test_window_leakage_safe` | periods=48 → shift uygulanmış |
| 6 | `test_window_all_sizes` | [24,48,168,336,720] her biri var |
| 7 | `test_window_all_functions` | mean, std, min, max |
| 8 | `test_expanding_created` | consumption_expanding_mean mevcut |
| 9 | `test_expanding_min_periods` | min_periods=48 uygulanıyor |
| 10 | `test_expanding_shift_applied` | periods=48 → shift edilmiş |
| 11 | `test_ewma_created` | consumption_ewma_24 mevcut |
| 12 | `test_momentum_created` | consumption_momentum_24 mevcut |
| 13 | `test_quantile_created` | consumption_q25_168 mevcut |
| 14 | `test_no_leakage_any_feature` | Hiçbir feature t'den sonrasına bakmaz |
| 15 | `test_nan_in_early_rows` | İlk 48+ satırda NaN |
| 16 | `test_config_driven_lags` | Custom lag listesi → o lag'lar |
| 17 | `test_missing_column_raises` | consumption yoksa → ValueError |
| 18 | `test_feature_engine_integration` | feature-engine transformer'lar çalışır |

### 7.5 `test_weather.py` (~12 test)

| # | Test | Tip |
|---|------|-----|
| 1 | `test_hdd_calculation` | T=10 → HDD=8 |
| 2 | `test_cdd_calculation` | T=30 → CDD=6 |
| 3 | `test_hdd_zero_warm` | T=25 → HDD=0 |
| 4 | `test_cdd_zero_cold` | T=10 → CDD=0 |
| 5 | `test_extreme_cold_flag` | T=-5 → 1 |
| 6 | `test_extreme_hot_flag` | T=40 → 1 |
| 7 | `test_rolling_temp_mean` | WindowFeatures doğru sonuç |
| 8 | `test_severity_mapping` | code=95 → severity=3 |
| 9 | `test_temp_change` | T(t)-T(t-3) == change_3h |
| 10 | `test_missing_column_graceful` | Eksik kolon → skip |
| 11 | `test_config_thresholds` | Custom threshold → farklı sonuç |
| 12 | `test_feature_engine_window` | WindowFeatures entegrasyonu doğru |

### 7.6 `test_solar.py` (~12 test)

| # | Test | Tip |
|---|------|-----|
| 1 | `test_elevation_midday_positive` | Öğlen → elevation > 0 |
| 2 | `test_elevation_midnight_zero` | Gece → elevation ≤ 0 |
| 3 | `test_ghi_daytime_positive` | Gündüz → ghi > 0 |
| 4 | `test_ghi_nighttime_zero` | Gece → ghi ≈ 0 |
| 5 | `test_clearness_range` | 0 ≤ Kt ≤ 1 |
| 6 | `test_cloud_proxy_inverse` | cloud = 1 - clearness |
| 7 | `test_is_daylight_flag` | Gündüz=1, gece=0 |
| 8 | `test_poa_calculated` | POA > 0 gündüz |
| 9 | `test_lead_features_shift` | ghi_lead_1 == ghi.shift(-1) |
| 10 | `test_lead_disabled` | enabled=false → lead yok |
| 11 | `test_config_location_used` | Custom lat/lon → farklı sonuç |
| 12 | `test_timezone_handling` | tz-naive input doğru çalışır |

### 7.7 `test_epias.py` (~14 test)

| # | Test | Tip |
|---|------|-----|
| 1 | `test_lag_features_created` | FDPP_lag_48 vb. mevcut |
| 2 | `test_lag_values_correct` | lag_48 == FDPP.shift(48) |
| 3 | `test_lag_min_lag_enforced` | Tümü >= 48 |
| 4 | `test_window_leakage_safe` | periods=48 |
| 5 | `test_window_all_variables` | 5 variable × 3 window |
| 6 | `test_expanding_shift` | periods=48 |
| 7 | `test_raw_dropped` | drop_raw=true → 5 ham kolon yok |
| 8 | `test_raw_kept_disabled` | drop_raw=false → kolonlar kalır |
| 9 | `test_no_leakage` | Hiçbir feature gelecek bilgi kullanmaz |
| 10 | `test_missing_variable_graceful` | Eksik kolon → skip |
| 11 | `test_all_five_variables` | 5 variable tümü işlenir |
| 12 | `test_nan_early_rows` | İlk 48+ satırda NaN |
| 13 | `test_config_driven` | Custom variable list → subset |
| 14 | `test_feature_engine_integration` | feature-engine transformer'lar çalışır |

### 7.8 `test_pipeline.py` (~12 test)

| # | Test | Tip |
|---|------|-----|
| 1 | `test_runs_all_modules` | 5 modül çalışır |
| 2 | `test_adds_features` | Output kolon > input kolon |
| 3 | `test_preserves_index` | DatetimeIndex korunur |
| 4 | `test_no_duplicate_columns` | Tekrar yok |
| 5 | `test_drops_raw_epias` | Raw EPIAS yok |
| 6 | `test_config_modules` | Sadece config'teki modüller |
| 7 | `test_subset_modules` | [calendar, weather] → 2 modül |
| 8 | `test_validate_output` | validate_output=true → kontrol |
| 9 | `test_feature_names` | get_feature_names() doğru |
| 10 | `test_empty_modules` | [] → input = output |
| 11 | `test_unknown_module_raises` | ["unknown"] → ValueError |
| 12 | `test_settings_integration` | Settings objesi ile çalışır |

### 7.9 Test Sayı Özeti

| Test Dosyası | Test Sayısı |
|-------------|------------|
| test_custom.py | 10 |
| test_calendar.py | 15 |
| test_consumption.py | 18 |
| test_weather.py | 12 |
| test_solar.py | 12 |
| test_epias.py | 14 |
| test_pipeline.py | 12 |
| test_openmeteo_client.py (SDK refactor) | 12 (yeniden yazım) |
| **Yeni M3 toplam** | **~105** |
| Mevcut (M1+M2, OpenMeteo testleri hariç) | ~73 |
| **Genel toplam** | **~178** |

---

## 8. Implementasyon Sırası

```
Adım 0:  OpenMeteo SDK Refactor (M2 client yeniden yazımı)
         0a: pyproject.toml → openmeteo-requests, requests-cache, retry-requests ekle
         0b: configs/openmeteo.yaml → SDK uyumlu yapıya güncelle
         0c: config/settings.py → OpenMeteoConfig, GeocodingConfig güncellemeleri
         0d: openmeteo_client.py → SDK ile yeniden yaz
         0e: tests/unit/test_data/test_openmeteo_client.py → SDK testleri
         0f: uv sync + lint + test
Adım 1:  pyproject.toml → feature-engine>=1.8 ekle, uv sync
Adım 2:  conftest.py fixtures (tüm test dosyaları buna bağımlı)
Adım 3:  custom.py + test_custom.py
         → EwmaFeatures, MomentumFeatures, QuantileFeatures, DegreeDayFeatures
         → Bağımsız, diğer modüllerden önce test edilebilir
Adım 4:  calendar.py + test_calendar.py
         → DatetimeFeatures + CyclicalFeatures (feature-engine)
         → Holiday/business (custom)
         → Leakage riski YOK
Adım 5:  weather.py + test_weather.py
         → WindowFeatures (feature-engine) + DegreeDayFeatures (custom)
         → Leakage riski YOK
Adım 6:  solar.py + test_solar.py
         → Tamamen custom (pvlib)
         → Leakage riski YOK
Adım 7:  consumption.py + test_consumption.py
         → LagFeatures + WindowFeatures + ExpandingWindowFeatures (feature-engine)
         → EwmaFeatures + MomentumFeatures + QuantileFeatures (custom)
         → YÜKSEK leakage riski — dikkatli implementasyon
Adım 8:  epias.py + test_epias.py
         → LagFeatures + WindowFeatures + ExpandingWindowFeatures (feature-engine)
         → Raw drop (custom)
         → YÜKSEK leakage riski
Adım 9:  pipeline.py + test_pipeline.py
         → Tüm modüller tamamlandıktan sonra
Adım 10: __init__.py güncelle
Adım 11: lint + mypy pass
         → ruff check --fix + ruff format
         → mypy --strict
```

---

## 9. Teknik Kararlar

### 9.1 feature-engine Leakage Güvenliği

feature-engine'in `WindowFeatures(periods=N)` işlemi:
```
series.rolling(window).agg(func).shift(periods)
```

Bu, `series.shift(periods).rolling(window).agg(func)` ile **matematiksel olarak eşdeğerdir**
(stateless aggregation'lar için). Her iki durumda da time t'de sadece t-periods ve
öncesinin istatistikleri görülür. **Leakage riski YOK.** ✅

### 9.2 feature-engine missing_values="ignore"

Consumption ve EPIAS DataFrame'lerinde forecast horizon satırları (T, T+1) NaN olabilir.
`missing_values="raise"` (default) bu durumda hata verir.
`missing_values="ignore"` ile NaN'lı satırlar tolere edilir.

### 9.3 feature-engine Kolon İsimlendirmesi

| Transformer | Pattern | Örnek |
|-------------|---------|-------|
| LagFeatures | `{var}_lag_{period}` | `consumption_lag_48` |
| WindowFeatures | `{var}_window_{size}_{func}` | `consumption_window_24_mean` |
| ExpandingWindowFeatures | `{var}_expanding_{func}` | `consumption_expanding_mean` |
| CyclicalFeatures | `{var}_sin`, `{var}_cos` | `hour_sin`, `hour_cos` |
| DatetimeFeatures | `{component}` | `hour`, `day_of_week` |

Bu isimlendirme zaten yeterince açıklayıcı. Ek prefix (cons_, epias_) gereksiz —
kolon adındaki variable ismi (consumption, FDPP) zaten kaynağı gösterir.

Custom transformer'larda isimlendirme:
- EWMA: `{var}_ewma_{span}` → `consumption_ewma_24`
- Momentum: `{var}_momentum_{period}` → `consumption_momentum_24`
- Quantile: `{var}_q{pct}_{window}` → `consumption_q25_168`
- HDD/CDD: `wth_hdd`, `wth_cdd`
- Solar: `sol_` prefix tüm solar feature'larda

### 9.4 pyproject.toml Güncellemesi

```toml
# dependencies bölümüne eklenmesi gereken:
"feature-engine>=1.8",
"openmeteo-requests>=1.3",
"requests-cache>=1.2",
"retry-requests>=2.0",

# Kaldırılabilecek (OpenMeteo SDK refactor sonrası):
# "httpx>=0.27"  → Eğer başka yerde kullanılmıyorsa kaldır
# "tenacity>=8.2" → EPIAS client hala kullanıyor, KALSIN
```

mypy overrides'a da eklenmeli:
```toml
[[tool.mypy.overrides]]
module = [
    # ... mevcut
    "feature_engine.*",
    "openmeteo_requests.*",
    "openmeteo_sdk.*",
    "requests_cache.*",
    "retry_requests.*",
]
ignore_missing_imports = true
```

### 9.5 Config → Transformer Parametre Eşleme

Tüm feature-engine ve custom transformer parametreleri config'ten okunur.
HARDCODED DEĞER YOK. Örnek:

```python
# Config (consumption.yaml):
# lags:
#   min_lag: 48
#   values: [48, 72, 96, 168, 336, 720]

# Transformer oluşturma:
lag = LagFeatures(
    variables=["consumption"],
    periods=self.config["lags"]["values"],  # YAML'dan
)
```

### 9.6 pvlib Timezone Handling

pvlib tz-aware DatetimeIndex gerektirir. Projede tz-naive convention var.
```python
# Solar engineer'da:
times = df.index.tz_localize(self.config["location"]["timezone"])
# pvlib hesaplamaları...
# Sonra .values ile tz-naive DataFrame'e geri yaz
```

### 9.7 Holiday File Graceful Fallback

Holiday parquet dosyası yoksa (test ortamı):
- `logger.warning()` ile uyar
- is_holiday = 0 for all rows
- Test'te `tmp_path` ile mock holiday file oluşturulur

### 9.8 WMO Severity Mapping (Basitleştirilmiş)

```python
def _map_severity(code: float) -> int:
    """WMO weather code → 4-level severity. Config'ten eşik değerleri."""
    if code < 4: return 0      # Clear/Cloudy
    if code < 70: return 1     # Fog/Drizzle/Rain
    if code < 90: return 2     # Snow/Freezing
    return 3                   # Thunderstorm
```

---

## 10. Bağımlılık Kontrolü

| Paket | pyproject.toml | Durum | Kullanım |
|-------|---------------|-------|----------|
| pandas | `>=2.1` | ✅ Mevcut | DataFrame operations |
| numpy | `>=1.26` | ✅ Mevcut | Numerical |
| scikit-learn | `>=1.4` | ✅ Mevcut | BaseEstimator mixin |
| pvlib | `>=0.11` | ✅ Mevcut | Solar calculations |
| loguru | `>=0.7` | ✅ Mevcut | Logging |
| pyarrow | `>=15.0` | ✅ Mevcut | Holiday parquet |
| holidays | `>=0.40` | ✅ Mevcut | (M2'de eklendi) |
| **feature-engine** | **YOK** | ❌ **EKLENECEK** | **LagFeatures, WindowFeatures, CyclicalFeatures, DatetimeFeatures** |
| **openmeteo-requests** | **YOK** | ❌ **EKLENECEK** | **OpenMeteo resmi SDK (FlatBuffers response)** |
| **requests-cache** | **YOK** | ❌ **EKLENECEK** | **HTTP response cache (CachedSession)** |
| **retry-requests** | **YOK** | ❌ **EKLENECEK** | **HTTP retry logic (backoff)** |

---

## 11. Leakage Audit Checklist (M4 hazırlık)

- [ ] LagFeatures periods tümü >= 48 (consumption + EPIAS)
- [ ] WindowFeatures periods=48 (consumption + EPIAS)
- [ ] ExpandingWindowFeatures periods=48, min_periods>=48
- [ ] EwmaFeatures periods=48
- [ ] MomentumFeatures min_lag=48
- [ ] QuantileFeatures periods=48
- [ ] Raw EPIAS kolonları pipeline çıkışında YOK
- [ ] Solar features lead dahil — LEAKAGE DEĞİL (deterministik)
- [ ] Weather features — LEAKAGE DEĞİL (forecast data)
- [ ] Calendar features — LEAKAGE DEĞİL
- [ ] Pipeline output'ta duplicate kolon YOK

---

## 12. Çıkış Kriterleri

- [ ] `make test` → ~93 yeni + 83 mevcut = ~176 test geçer
- [ ] `make lint` → `ruff check` + `mypy --strict` temiz
- [ ] `feature-engine>=1.8` pyproject.toml'da ve yüklü
- [ ] CalendarFeatureEngineer: DatetimeFeatures + CyclicalFeatures + custom holidays
- [ ] ConsumptionFeatureEngineer: ~38 feature, tümü leakage-safe
- [ ] WeatherFeatureEngineer: HDD/CDD + WindowFeatures rolling + severity
- [ ] SolarFeatureEngineer: pvlib entegre, ~13 feature
- [ ] EpiasFeatureEngineer: ~50 derived, 5 raw DROP
- [ ] FeaturePipeline: 5 modülü orkestre, output validate
- [ ] Custom transformers: sklearn-uyumlu (fit/transform)
- [ ] Hiçbir feature min_lag < 48 ihlali yapmaz
- [ ] Tüm parametreler config'ten okunur, hardcoded değer YOK
- [ ] OpenMeteo SDK refactor: `openmeteo-requests` + `requests-cache` + `retry-requests`
- [ ] Geocoding API entegrasyonu: şehir adı → otomatik koordinat (config flag)
- [ ] Historical Forecast API desteği (3. endpoint)
- [ ] Mevcut OpenMeteo testleri SDK formatına göre güncellenmiş

---

## 13. Commit Stratejisi

İki commit (SDK refactor ayrı, feature engineering ayrı):

**Commit 1: SDK Refactor**
```
refactor(data): migrate OpenMeteo client to official SDK

- Replace httpx with openmeteo-requests (FlatBuffers response)
- Replace custom SQLite cache with requests-cache CachedSession
- Replace tenacity retry with retry-requests
- Add Historical Forecast API endpoint support
- Add Geocoding API integration (configurable)
- Update OpenMeteoConfig with backoff_factor, geocoding settings
- Rewrite OpenMeteo tests for SDK mock format
```

**Commit 2: Feature Engineering**
```
feat(features): implement feature engineering pipeline with feature-engine

- feature-engine integration: LagFeatures, WindowFeatures,
  ExpandingWindowFeatures, CyclicalFeatures, DatetimeFeatures
- Custom transformers: EwmaFeatures, MomentumFeatures,
  QuantileFeatures, DegreeDayFeatures (sklearn-compatible)
- Calendar: datetime extraction + cyclical + holiday + business
- Consumption: lag, rolling, expanding, EWMA, momentum, quantile
  with min_lag=48 leakage protection
- Weather: HDD/CDD + rolling + severity + extreme flags
- Solar: pvlib GHI/DNI/DHI, POA, clearness, lead features
- EPIAS: lag, rolling, expanding with raw value dropping
- Pipeline orchestrator with output validation
- ~105 unit tests covering all modules and custom transformers
```

---

## 14. OpenMeteo SDK Refactor

### 14.1 Mevcut Durum vs SDK

Mevcut M2 implementasyonu (`openmeteo_client.py`, 317 satır):
- **HTTP:** `httpx.Client` ile manuel GET request
- **Parse:** `response.json()["hourly"]` → JSON dict parse
- **Cache:** Custom SQLite table (`weather_cache` tablo, key-value)
- **Retry:** `tenacity` decorator (`@retry`, exponential backoff)
- **Response format:** JSON dict `{"hourly": {"time": [...], "temperature_2m": [...]}}`

Resmi SDK (`openmeteo-requests`):
- **HTTP:** `openmeteo_requests.Client` (wrapper around `requests.Session`)
- **Parse:** FlatBuffers binary → `.Hourly()`, `.Variables(i)`, `.ValuesAsNumpy()`
- **Cache:** `requests_cache.CachedSession` (SQLite/filesystem/memory, otomatik)
- **Retry:** `retry_requests.retry(session, retries=N, backoff_factor=X)`
- **Response format:** Typed objects, numpy array çıktı

**Sonuç:** Mevcut client SDK ile UYUMSUZ. Tam yeniden yazım gerekli.

### 14.2 SDK Response Format

```python
import openmeteo_requests
import requests_cache
from retry_requests import retry

# Setup: cache + retry + client
cache_session = requests_cache.CachedSession(
    ".cache",
    expire_after=3600,  # config: cache.ttl_hours * 3600
)
retry_session = retry(
    cache_session,
    retries=5,           # config: api.retry_attempts
    backoff_factor=0.2,  # config: api.backoff_factor
)
om = openmeteo_requests.Client(session=retry_session)

# API call
params = {
    "latitude": 40.183,
    "longitude": 29.050,
    "hourly": ["temperature_2m", "relative_humidity_2m", ...],
    "timezone": "Europe/Istanbul",
    "start_date": "2024-01-01",
    "end_date": "2024-01-31",
}
responses = om.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)
# responses: list[WeatherApiResponse]

response = responses[0]
hourly = response.Hourly()

# Variable erişimi — index sırası params["hourly"] ile aynı
temperature = hourly.Variables(0).ValuesAsNumpy()  # numpy array
humidity = hourly.Variables(1).ValuesAsNumpy()

# Zaman dizisi
import numpy as np
import pandas as pd
hourly_time = pd.date_range(
    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
    freq=pd.Timedelta(seconds=hourly.Interval()),
    inclusive="left",
)
```

### 14.3 Yeniden Yazım Planı: `openmeteo_client.py`

```python
class OpenMeteoClient:
    """Open-Meteo SDK client with caching and retry.

    httpx → openmeteo_requests.Client
    Custom SQLite cache → requests_cache.CachedSession
    tenacity → retry_requests.retry
    JSON parsing → FlatBuffers .Hourly().Variables(i).ValuesAsNumpy()
    """

    def __init__(self, config: OpenMeteoConfig, region: RegionConfig) -> None:
        self.config = config
        self.region = region

        # Cache session
        cache_session = requests_cache.CachedSession(
            cache_name=str(Path(config.cache.path).with_suffix("")),  # .db uzantısız
            backend=config.cache.backend,    # "sqlite"
            expire_after=config.cache.ttl_hours * 3600,
        )

        # Retry session
        retry_session = retry(
            cache_session,
            retries=config.api.retry_attempts,  # 3
            backoff_factor=config.api.backoff_factor,  # 0.2 (yeni config)
        )

        # OpenMeteo client
        self._client = openmeteo_requests.Client(session=retry_session)

    def close(self) -> None:
        """Close the session."""
        # requests_cache.CachedSession has .close()
        pass  # Session garbage collected

    # Public API: fetch_historical, fetch_forecast — imza AYNI kalır
    # Return tipi AYNI: pd.DataFrame (hourly DatetimeIndex + weather columns)

    def fetch_historical(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical weather — weighted average across cities."""
        city_dfs = []
        for city in self.region.cities:
            df = self._fetch_single_location(
                url=self.config.api.base_url_historical,
                latitude=city.latitude,
                longitude=city.longitude,
                start_date=start_date,
                end_date=end_date,
            )
            city_dfs.append((city, df))
        return self._compute_weighted_average(city_dfs)

    def fetch_historical_forecast(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical forecast (reanalysis + forecast blend). YENİ ENDPOINT."""
        city_dfs = []
        for city in self.region.cities:
            df = self._fetch_single_location(
                url=self.config.api.base_url_historical_forecast,
                latitude=city.latitude,
                longitude=city.longitude,
                start_date=start_date,
                end_date=end_date,
            )
            city_dfs.append((city, df))
        return self._compute_weighted_average(city_dfs)

    def fetch_forecast(self, forecast_days: int = 2) -> pd.DataFrame:
        """Fetch weather forecast — imza AYNI."""
        # ... aynı pattern

    def _fetch_single_location(self, url: str, ...) -> pd.DataFrame:
        """SDK ile tek lokasyon fetch.

        Önemli değişiklikler:
        - self._client.weather_api(url, params=params) → FlatBuffers response
        - _parse_response() → FlatBuffers parse (JSON değil)
        - Cache ve retry otomatik (session seviyesinde)
        - tenacity @retry decorator KALDIRILDI
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": self.config.variables,  # list[str]
            "timezone": "Europe/Istanbul",
        }
        if start_date and end_date:
            params["start_date"] = start_date
            params["end_date"] = end_date
        if forecast_days is not None:
            params["forecast_days"] = forecast_days

        try:
            responses = self._client.weather_api(url, params=params)
        except Exception as exc:
            msg = f"OpenMeteo API error: {exc}"
            raise OpenMeteoApiError(msg) from exc

        return self._parse_sdk_response(responses[0])

    def _parse_sdk_response(self, response: Any) -> pd.DataFrame:
        """FlatBuffers response → DataFrame.

        SDK response format:
        - response.Hourly() → hourly block
        - hourly.Variables(i).ValuesAsNumpy() → numpy array per variable
        - hourly.Time() / hourly.TimeEnd() / hourly.Interval() → timestamps (epoch)
        """
        hourly = response.Hourly()
        if hourly is None:
            raise OpenMeteoApiError("No hourly data in response")

        # Zaman dizisi oluştur
        times = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
        # UTC → Europe/Istanbul
        times = times.tz_convert("Europe/Istanbul").tz_localize(None)

        # Variables → DataFrame
        columns = {}
        for i, var_name in enumerate(self.config.variables):
            values = hourly.Variables(i).ValuesAsNumpy()
            columns[var_name] = values

        df = pd.DataFrame(columns, index=times)
        df.index.name = "datetime"
        return df

    # _compute_weighted_average() → DEĞİŞMEZ
```

### 14.4 Geocoding API Entegrasyonu

**Endpoint:** `https://geocoding-api.open-meteo.com/v1/search`

```
GET /v1/search?name=Bursa&count=1&language=tr&format=json
→ {
    "results": [{
      "id": 745042,
      "name": "Bursa",
      "latitude": 40.19559,
      "longitude": 29.06013,
      "elevation": 155.0,
      "timezone": "Europe/Istanbul",
      "country_code": "TR",
      ...
    }]
  }
```

**Plan:**
- Yeni `GeocodingConfig` Pydantic modeli: `enabled: bool`, `api_url: str`
- `configs/openmeteo.yaml`'a `geocoding:` bölümü ekle
- `OpenMeteoClient`'a `resolve_coordinates()` metodu ekle
- `CityConfig`'te `latitude/longitude` opsiyonel → eğer geocoding enabled ve koordinat
  verilmemişse, city name ile otomatik çöz
- İlk çağrıda koordinatları cache'le (requests-cache otomatik yapar)

**Config değişikliği (openmeteo.yaml):**
```yaml
api:
  base_url_historical: "https://archive-api.open-meteo.com/v1/archive"
  base_url_historical_forecast: "https://historical-forecast-api.open-meteo.com/v1/forecast"
  base_url_forecast: "https://api.open-meteo.com/v1/forecast"
  timeout: 30
  retry_attempts: 3
  backoff_factor: 0.2

geocoding:
  enabled: false          # Varsayılan kapalı — mevcut hardcoded koordinatlar çalışmaya devam
  api_url: "https://geocoding-api.open-meteo.com/v1/search"
  language: "tr"
  count: 1

variables:
  - temperature_2m
  - relative_humidity_2m
  # ... (11 değişken aynı)

cache:
  backend: "sqlite"
  path: "data/external/weather_cache.db"
  ttl_hours: 6
```

**Pydantic model değişiklikleri (`settings.py`):**

```python
class GeocodingConfig(BaseModel, frozen=True):
    """Geocoding API configuration."""
    enabled: bool = False
    api_url: str = "https://geocoding-api.open-meteo.com/v1/search"
    language: str = "tr"
    count: int = 1

class OpenMeteoApiConfig(BaseModel, frozen=True):
    base_url_historical: str = "https://archive-api.open-meteo.com/v1/archive"
    base_url_historical_forecast: str = (
        "https://historical-forecast-api.open-meteo.com/v1/forecast"
    )
    base_url_forecast: str = "https://api.open-meteo.com/v1/forecast"
    timeout: int = 30
    retry_attempts: int = 3
    backoff_factor: float = 0.2  # YENİ

class OpenMeteoConfig(BaseModel, frozen=True):
    api: OpenMeteoApiConfig = OpenMeteoApiConfig()
    variables: list[str] = [...]
    cache: WeatherCacheConfig = WeatherCacheConfig()
    geocoding: GeocodingConfig = GeocodingConfig()  # YENİ
```

### 14.5 API Endpoints (3 endpoint)

| Endpoint | URL | Kullanım |
|----------|-----|----------|
| Historical | `archive-api.open-meteo.com/v1/archive` | Eğitim verisi (geçmiş hava) |
| Historical Forecast | `historical-forecast-api.open-meteo.com/v1/forecast` | **YENİ** — Geçmiş tahminler (reanalysis blend) |
| Forecast | `api.open-meteo.com/v1/forecast` | Prediction (T ve T+1 hava tahmini) |
| Geocoding | `geocoding-api.open-meteo.com/v1/search` | Şehir adı → koordinat çözümleme |

**Historical Forecast Avantajı:**
- Son 5 güne kadar veri sağlar (Historical API'nin 5 gün gecikmesi var)
- Tahmin ile gerçek verinin blend'i — daha güncel
- Training data gap'ini kapatır

### 14.6 SDK Variable Mapping

SDK FlatBuffers response'ta değişkenler index sırasıyla erişilir:
`hourly.Variables(i).ValuesAsNumpy()` — `i` sırası `params["hourly"]` listesindeki
sıra ile aynıdır.

| Config Variable | SDK Variable (openmeteo_sdk) | Enum | Altitude |
|----------------|------------------------------|------|----------|
| temperature_2m | Variable.temperature + Altitude(2m) | 47 | 2m |
| relative_humidity_2m | Variable.relative_humidity + Altitude(2m) | 29 | 2m |
| dew_point_2m | Variable.dew_point + Altitude(2m) | 8 | 2m |
| apparent_temperature | Variable.apparent_temperature | 1 | — |
| precipitation | Variable.precipitation | 24 | — |
| snow_depth | Variable.snow_depth | 35 | — |
| weather_code | Variable.weather_code | 56 | — |
| surface_pressure | Variable.surface_pressure | 45 | — |
| wind_speed_10m | Variable.wind_speed + Altitude(10m) | 59 | 10m |
| wind_direction_10m | Variable.wind_direction + Altitude(10m) | 57 | 10m |
| shortwave_radiation | Variable.shortwave_radiation | 32 | — |

**NOT:** SDK'da variable erişimi index sırası ile yapılır — config variables listesindeki
sıra params["hourly"] listesine birebir aktarılır. Enum değerlerine doğrudan
erişim gerekmez (string isimler yeterli).

### 14.7 Test Stratejisi (SDK)

Mevcut 10 test tamamen yeniden yazılacak. Mock stratejisi değişiyor:
- **Eski:** `patch.object(client._client, "get", ...)` → MagicMock JSON response
- **Yeni:** `patch.object(client._client, "weather_api", ...)` → Mock FlatBuffers response

```python
class MockHourly:
    """Mock openmeteo_sdk Hourly response."""
    def __init__(self, variables: list[np.ndarray], time_start: int, time_end: int):
        self._variables = variables
        self._time_start = time_start
        self._time_end = time_end

    def Variables(self, index: int) -> MockVariable:
        return MockVariable(self._variables[index])

    def Time(self) -> int:
        return self._time_start

    def TimeEnd(self) -> int:
        return self._time_end

    def Interval(self) -> int:
        return 3600  # hourly

class MockVariable:
    def __init__(self, values: np.ndarray):
        self._values = values

    def ValuesAsNumpy(self) -> np.ndarray:
        return self._values

class MockWeatherResponse:
    def __init__(self, hourly: MockHourly):
        self._hourly = hourly

    def Hourly(self) -> MockHourly:
        return self._hourly
```

**Yeni testler (~12 test):**

| # | Test | Değişiklik |
|---|------|-----------|
| 1 | `test_returns_dataframe` | SDK mock ile |
| 2 | `test_datetime_index` | UTC→Istanbul dönüşümü |
| 3 | `test_weighted_average_correct` | Aynı mantık, farklı mock |
| 4 | `test_config_variables_used` | SDK variable index sırası |
| 5 | `test_api_error_raises` | SDK exception handling |
| 6 | `test_no_hourly_raises` | Hourly() returns None |
| 7 | `test_forecast_returns_dataframe` | forecast_days parametresi |
| 8 | `test_historical_forecast_endpoint` | **YENİ** — 3. endpoint |
| 9 | `test_cache_session_configured` | requests-cache session doğru init |
| 10 | `test_retry_session_configured` | retry-requests session doğru init |
| 11 | `test_geocoding_resolve` | **YENİ** — Geocoding API mock |
| 12 | `test_geocoding_disabled` | **YENİ** — enabled=false → skip |

### 14.8 Weather Feature Uyumu

SDK refactor sonrası `WeatherFeatureEngineer` input'u DEĞİŞMEZ.
`OpenMeteoClient.fetch_historical()` hala aynı DataFrame döndürür:
- DatetimeIndex (hourly, tz-naive Europe/Istanbul)
- 11 weather kolon (temperature_2m, relative_humidity_2m, ...)

SDK response format dahili olarak farklı (FlatBuffers → numpy → DataFrame) ama
public API output'u aynı kalır. Weather feature modülünde değişiklik GEREKMEZ.

### 14.9 Breaking Changes ve Geçiş

| Bileşen | Eski | Yeni | Breaking? |
|---------|------|------|-----------|
| HTTP client | `httpx.Client` | `openmeteo_requests.Client` | Internal |
| Cache | Custom SQLite (`weather_cache` table) | `requests_cache.CachedSession` (otomatik SQLite) | Cache file format değişir |
| Retry | `tenacity @retry` decorator | `retry_requests.retry(session)` | Internal |
| Response parse | `response.json()["hourly"]` | `response.Hourly().Variables(i).ValuesAsNumpy()` | Internal |
| Public API | `fetch_historical(start, end) → DataFrame` | AYNI | ✅ Uyumlu |
| Config | `OpenMeteoApiConfig` | + `backoff_factor`, + `base_url_historical_forecast` | Additive |

**Geçiş notları:**
- Mevcut `data/external/weather_cache.db` custom SQLite formatı requests-cache formatıyla
  uyumsuz → ilk çalıştırmada cache sıfırlanır (veya eski dosya silinir)
- `httpx` bağımlılığı EPIAS client tarafından hala kullanılıyor → pyproject.toml'dan KALDIRILMAZ
- `tenacity` bağımlılığı EPIAS client tarafından hala kullanılıyor → KALDIRILMAZ

---

## 15. Kapsam Dışı

- M4: Leakage Audit (feature'ların otomatik doğrulanması)
- M5: CatBoost training (feature importance ile gereksiz feature eleme)
- Interaction features → M5 feature selection sonrası
- Advanced volatility (MAD, IQR) → M5 Optuna karar verir
- Distribution features (skew, kurtosis) → CatBoost otomatik öğrenir
