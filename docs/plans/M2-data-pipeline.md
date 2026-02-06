# M2: Data Pipeline — Implementation Plan

## 1. Kapsam

**Milestone:** M2 — Data pipeline (ingestion + EPIAS/OpenMeteo clients + scripts)
**Bağımlılık:** M0 ✅, M1 ✅ (Config sistemi tamamlandı)
**Hedef:** Excel tüketim verisi yükleme, EPIAS piyasa verisi çekme, OpenMeteo hava durumu
verisi çekme, profil katsayıları çekme, tatil verisi üretme — config-driven, type-safe,
cache destekli data ingestion katmanı + CLI scripts.

## 2. Mevcut Durum

| Dosya | Durum | Satır |
|-------|-------|-------|
| `src/energy_forecast/data/loader.py` | Placeholder (NotImplementedError) | 30 |
| `src/energy_forecast/data/epias_client.py` | Placeholder (NotImplementedError) | 31 |
| `src/energy_forecast/data/openmeteo_client.py` | Placeholder (NotImplementedError) | 40 |
| `src/energy_forecast/data/__init__.py` | Sadece DataLoader export | 6 |
| `configs/data_loader.yaml` | Tamamlandı | 22 |
| `configs/openmeteo.yaml` | Tamamlandı | 25 |
| `configs/features/epias.yaml` | Tamamlandı (source, cache_dir, variables) | 37 |
| `src/energy_forecast/config/settings.py` | Tamamlandı — tüm Pydantic modelleri mevcut | 856 |
| `scripts/` | Dizin yok — oluşturulacak | — |

## 3. Referans Proje Analizi

Eski proje: `C:\Users\pc\Desktop\distributed-energy-forecasting\`

### 3.1 Kaynak → Hedef Eşleme

| Eski Dosya | Eski Satır | Yeni Dosya | Hedef Satır |
|------------|-----------|------------|-------------|
| `src/data/ingestion.py` | 421 | `data/loader.py` | ~250 |
| `src/utils/epias_client.py` | 950 | `data/epias_client.py` | ~400 |
| `src/utils/openmeteo_client.py` | 679 | `data/openmeteo_client.py` | ~350 |
| `scripts/generate_turkish_holidays.py` | 229 | `scripts/generate_holidays.py` | ~150 |
| `scripts/backfill_epias_cache.py` | 225 | `scripts/backfill_epias.py` | ~100 |
| `scripts/fetch_profile_coefficients.py` | 717 | `scripts/fetch_profile_coefficients.py` | ~300 |

### 3.2 Taşınacak Mantık (KOD KOPYALAMADAN yeniden yaz)

**DataIngestion → DataLoader:**
- Excel okuma: `date` + `time` kolon merge → `pd.to_datetime(date) + pd.to_timedelta(time, unit="h")`
- Continuous time series: `pd.date_range(start, periods, freq="h")` → left merge
- Validation: required columns, time range [0-23], min rows, consumption > 0
- Forecast horizon: Son data noktasından +48 saat NaN dolgu

**EpiasClient:**
- Auth: `POST /cas/v1/tickets` → TGT token (1 saat TTL)
- 5 endpoint'ten veri çekme (FDPP, Real_Time_Consumption, DAM_Purchase, Bilateral, Load_Forecast)
- Yıl bazlı parquet cache: `data/external/epias/{year}.parquet`
- Inner merge on date
- Rate limit: 10s delay, 3 retry + exponential backoff (tenacity)

**OpenMeteoClient:**
- Historical: `archive-api.open-meteo.com/v1/archive`
- Forecast: `api.open-meteo.com/v1/forecast`
- 4 şehir ağırlıklı ortalama (Bursa %60, Balıkesir %24, Yalova %10, Çanakkale %6)
- 11 hava durumu değişkeni
- SQLite cache (configurable TTL)

**generate_holidays.py:**
- `holidays.country_holidays('TR')` ile 2015-2044 arası Türk tatilleri
- Çakışma çözümü (aynı günde birden fazla tatil → priority listesi)
- Türkçe → İngilizce isim standardizasyonu
- Çıktı: `data/static/turkish_holidays.parquet` (date, holiday_name, raw_holiday_name)

**backfill_epias.py:**
- EpiasClient üzerinden 2020-current year arası tüm yılları cache'le
- Skip: zaten cache'de olan yıllar
- Rapor: success/skip/fail sayıları

**fetch_profile_coefficients.py:**
- Aynı EPIAS auth mekanizması
- Uludağ dağıtım şirketi ID → profil grupları keşfi
- Her profil grubu × her ay × her yıl için katsayı çekme
- Konut/Sanayi/Ticari/Tarım LV+MV aggregate
- Aydınlatma kolonlarının birleştirilmesi
- Çıktı: `data/external/profile/{year}.parquet`

### 3.3 Taşınmayacak / İyileştirilecek

- ✗ String date format (`"YYYY-MM-DD HH:MM"`) → **DatetimeIndex kullan**
- ✗ Memory optimization (float32 downcast) → **gereksiz, modern pandas yeterli**
- ✗ CacheManager sınıfı → **EPIAS inline parquet cache, OpenMeteo inline SQLite**
- ✗ Aşırı derin try/except → **sadece external API'lerde, temiz exception hierarchy**
- ✗ WMO code mapping (93 satır hardcoded) → **M3 weather features scope'unda**
- ✗ Private method erişimi (`_fetch_year_data`) → **public backfill API**
- ✗ Hardcoded yıl aralıkları → **CLI argümanları veya config'ten**
- ✗ Profile group keyword matching fragility → **explicit mapping dict + unknown grubu loglama**
- ✗ `print()` kullanımı → **loguru**

## 4. Dosya Planı

### 4.1 Düzenlenecek Dosyalar

| Dosya | Mevcut → Hedef | Değişiklik |
|-------|----------------|------------|
| `src/energy_forecast/data/loader.py` | 30 → ~250 | Tam implementasyon |
| `src/energy_forecast/data/epias_client.py` | 31 → ~400 | Tam implementasyon |
| `src/energy_forecast/data/openmeteo_client.py` | 40 → ~350 | Tam implementasyon |
| `src/energy_forecast/data/__init__.py` | 6 → ~15 | Public API güncelleme |
| `tests/conftest.py` | 40 → ~120 | Data fixtures |
| `pyproject.toml` | dependencies | `holidays>=0.40` ekle |

### 4.2 Yeni Oluşturulacak Dosyalar

| Dosya | Tahmini Boyut | İçerik |
|-------|--------------|--------|
| `src/energy_forecast/data/schemas.py` | ~80 | Pandera DataFrame schemas |
| `src/energy_forecast/data/exceptions.py` | ~40 | Custom exception sınıfları |
| `scripts/__init__.py` | 1 | Package marker |
| `scripts/generate_holidays.py` | ~150 | Türk tatilleri parquet üretici |
| `scripts/backfill_epias.py` | ~100 | EPIAS yıllık cache backfill |
| `scripts/fetch_profile_coefficients.py` | ~300 | EPIAS profil katsayıları çekici |
| `tests/unit/test_data/__init__.py` | 1 | Package init |
| `tests/unit/test_data/test_loader.py` | ~300 | DataLoader unit testleri |
| `tests/unit/test_data/test_epias_client.py` | ~250 | EpiasClient testleri (mocked) |
| `tests/unit/test_data/test_openmeteo_client.py` | ~250 | OpenMeteoClient testleri (mocked) |
| `tests/unit/test_data/test_schemas.py` | ~100 | Schema validation testleri |
| `tests/unit/test_scripts/__init__.py` | 1 | Package init |
| `tests/unit/test_scripts/test_generate_holidays.py` | ~100 | Holiday generation testleri |

## 5. Implementation Detayları

### 5.1 `src/energy_forecast/data/exceptions.py`

```python
class DataError(Exception):
    """Base exception for data module."""

class DataValidationError(DataError):
    """Input data validation failed."""

class EpiasApiError(DataError):
    """EPIAS API request failed."""

class EpiasAuthError(EpiasApiError):
    """EPIAS authentication failed."""

class OpenMeteoApiError(DataError):
    """OpenMeteo API request failed."""
```

### 5.2 `src/energy_forecast/data/schemas.py`

Pandera schemas for input/output validation:

```python
class RawExcelSchema(pa.DataFrameModel):
    """Raw Excel input after column rename."""
    date: Series[str]
    time: Series[int] = pa.Field(ge=0, le=23)
    consumption: Series[float] = pa.Field(ge=0.0, le=10000.0)

class ConsumptionSchema(pa.DataFrameModel):
    """Processed consumption DataFrame."""
    # DatetimeIndex + consumption column

class EpiasSchema(pa.DataFrameModel):
    """EPIAS market data (5 variables)."""
    # DatetimeIndex + 5 EPIAS columns

class WeatherSchema(pa.DataFrameModel):
    """Weighted-average weather data (11 variables)."""
    # DatetimeIndex + 11 weather columns
```

### 5.3 `src/energy_forecast/data/loader.py` — DataLoader

```python
class DataLoader:
    """Loads and validates consumption data from Excel files."""

    def __init__(self, config: DataLoaderConfig) -> None: ...

    def load_excel(self, path: Path) -> pd.DataFrame:
        """Load Excel → validate → merge date+time → DatetimeIndex → continuous series.

        Steps:
            1. pd.read_excel(path) — openpyxl engine
            2. _validate_columns() — date, time, consumption mevcut mu
            3. _validate_raw_data() — Pandera RawExcelSchema
            4. _merge_datetime() — date + timedelta(time, "h")
            5. _create_continuous_index() — pd.date_range, reindex
            6. _set_datetime_index() — DatetimeIndex, freq="h"
            7. Return: DataFrame[DatetimeIndex, consumption]
        """

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Check required columns exist, raise DataValidationError."""

    def _merge_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge date + time → single datetime column.

        Formula: pd.to_datetime(date) + pd.to_timedelta(time, unit='h')
        """

    def _create_continuous_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill gaps with NaN for missing hours.

        Uses pd.date_range(min, max, freq='h') → reindex.
        """

    def extend_for_forecast(
        self, df: pd.DataFrame, horizon_hours: int = 48,
    ) -> pd.DataFrame:
        """Extend DataFrame with NaN rows for forecast period.

        Adds horizon_hours rows after last data point.
        """
```

**Config:** `settings.data_loader` (DataLoaderConfig)

### 5.4 `src/energy_forecast/data/epias_client.py` — EpiasClient

```python
class EpiasClient:
    """EPIAS Transparency Platform API client with year-based caching."""

    AUTH_URL: ClassVar[str] = "https://giris.epias.com.tr/cas/v1/tickets"
    BASE_URL: ClassVar[str] = "https://seffaflik.epias.com.tr/electricity-service/v1"

    def __init__(
        self,
        username: str,
        password: str,
        cache_dir: Path = Path("data/external/epias"),
        rate_limit_seconds: float = 10.0,
    ) -> None: ...

    def authenticate(self) -> str:
        """Get TGT token from EPIAS CAS.

        POST AUTH_URL → TGT (cached for 1 hour).
        """

    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch all EPIAS variables for date range.

        Steps:
            1. Determine year range from dates
            2. Per year: load_cache() or fetch_year()
            3. Concat + filter to [start, end]
            4. Return DataFrame[DatetimeIndex, 5 EPIAS columns]
        """

    def fetch_year(self, year: int) -> pd.DataFrame:
        """Fetch full year from API, save cache. PUBLIC for backfill script."""

    def load_cache(self, year: int) -> pd.DataFrame | None:
        """Load cached parquet for year, or None."""

    def save_cache(self, year: int, df: pd.DataFrame) -> None:
        """Save year data as parquet."""

    def _fetch_variable(
        self, endpoint: str, start_date: str, end_date: str,
    ) -> pd.DataFrame:
        """Fetch single EPIAS variable. Retry + rate limit."""
```

**EPIAS Endpoint Map:**

| Variable | Endpoint | Response Key |
|----------|----------|-------------|
| FDPP | `/generation/data/dpp` | `toplam` |
| Real_Time_Consumption | `/consumption/data/realtime-consumption` | `consumption` |
| DAM_Purchase | `/markets/dam/data/clearing-quantity` | `matchedBids` |
| Bilateral_Agreement_Purchase | `/markets/bilateral-contracts/data/bilateral-contracts-bid-quantity` | `quantity` |
| Load_Forecast | `/consumption/data/load-estimation-plan` | `lep` |

**Config:**
- `settings.env.epias_username`, `settings.env.epias_password`
- `settings.features.epias.source.cache_dir`
- `settings.features.epias.variables`

### 5.5 `src/energy_forecast/data/openmeteo_client.py` — OpenMeteoClient

```python
class OpenMeteoClient:
    """Open-Meteo weather API client with multi-location weighted average."""

    def __init__(self, config: OpenMeteoConfig, region: RegionConfig) -> None: ...

    def fetch_historical(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical weather, return weighted average.

        Steps:
            1. Per city: GET archive API → hourly data
            2. Weighted average: Σ(city.weight × city_data)
            3. Return DataFrame[DatetimeIndex, 11 weather columns]
        """

    def fetch_forecast(self, forecast_days: int = 2) -> pd.DataFrame:
        """Fetch weather forecast for T and T+1.

        Same logic as historical, different base URL.
        """

    def _fetch_single_location(
        self, base_url: str, latitude: float, longitude: float,
        start_date: str | None = None, end_date: str | None = None,
        forecast_days: int | None = None,
    ) -> pd.DataFrame:
        """Fetch weather for single (lat, lon)."""

    def _compute_weighted_average(
        self, city_dfs: list[tuple[CityConfig, pd.DataFrame]],
    ) -> pd.DataFrame:
        """Weighted average: Σ(weight_i × value_i) for each variable."""

    def _parse_response(self, data: dict[str, Any]) -> pd.DataFrame:
        """Parse Open-Meteo JSON → DataFrame.

        Keys: data['hourly']['time'] → index, data['hourly'][var] → columns.
        """
```

**Cache:** SQLite backend, key=`f"{lat}_{lon}_{start}_{end}"`, TTL from config.
**Config:** `settings.openmeteo` + `settings.region`

### 5.6 `scripts/generate_holidays.py`

```python
"""Generate Turkish holiday parquet file (2015-2044).

Usage: python scripts/generate_holidays.py

Output: data/static/turkish_holidays.parquet
"""

HOLIDAY_NAME_MAPPING: dict[str, str] = {
    "Yılbaşı": "New Year",
    "Ulusal Egemenlik ve Çocuk Bayramı": "National Sovereignty",
    "Emek ve Dayanışma Günü": "Labour Day",
    "Atatürk'ü Anma, Gençlik ve Spor Bayramı": "Youth and Sports Day",
    "Zafer Bayramı": "Victory Day",
    "Cumhuriyet Bayramı": "Republic Day",
    # Ramazan + Kurban bayramları (tahmini dahil)
    ...
}

HOLIDAY_PRIORITY: list[str] = [
    "Republic Day", "Victory Day", "National Sovereignty", ...
]

def generate_holiday_catalog(
    start_year: int = 2015, end_year: int = 2044,
) -> pd.DataFrame:
    """Generate holiday catalog with collision resolution.

    Returns DataFrame: date, holiday_name (categorical), raw_holiday_name.
    """

def main() -> None:
    """Generate and save holiday parquet."""
```

**Yeni bağımlılık:** `holidays>=0.40` (pyproject.toml'a eklenecek)

### 5.7 `scripts/backfill_epias.py`

```python
"""Backfill EPIAS year-based cache.

Usage: python scripts/backfill_epias.py [--start-year 2020] [--end-year 2025]

Output: data/external/epias/{year}.parquet
"""

def main(start_year: int = 2020, end_year: int | None = None) -> None:
    """Backfill missing years.

    Steps:
        1. Load config, create EpiasClient
        2. For each year: check cache → skip or fetch_year()
        3. Report summary (success/skip/fail counts)
    """
```

**Kullanır:** `EpiasClient.fetch_year()` (PUBLIC method) + `EpiasClient.load_cache()`

### 5.8 `scripts/fetch_profile_coefficients.py`

```python
"""Fetch EPIAS profile coefficients for Uludağ distribution.

Usage: python scripts/fetch_profile_coefficients.py [--start-year 2020] [--end-year 2025]

Output: data/external/profile/{year}.parquet
"""

# Uses EpiasClient.authenticate() for shared auth

PROFILE_ENDPOINTS = {
    "distribution": "/consumption/data/multiple-factor-distribution",
    "meter_type": "/consumption/data/multiple-factor-meter-reading-type",
    "profile_groups": "/consumption/data/multiple-factor-profile-group",
    "coefficients": "/consumption/data/multiple-factor",
}

PROFILE_NAME_MAPPING: dict[str, str] = {
    # Turkish profile group keywords → English column names
    # Explicit mapping instead of fragile keyword matching
}

def authenticate(username: str, password: str) -> str:
    """Get TGT token (shared with EpiasClient pattern)."""

def get_distribution_id(tgt: str, target: str = "ULUDAĞ") -> int:
    """Find distribution company ID by name."""

def get_profile_groups(tgt: str, distribution_id: int, period: str) -> list[dict]:
    """List all subscriber profile groups."""

def fetch_coefficients_for_year(
    tgt: str, year: int, distribution_id: int,
    meter_type_id: int, profile_groups: list[dict],
) -> pd.DataFrame:
    """Fetch all profile groups for a year, merge + aggregate."""

def normalize_profile_name(raw_name: str) -> str:
    """Map Turkish profile group name to English column name."""

def add_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Create LV+MV aggregate columns (residential, industrial, etc.)."""

def main(start_year: int = 2020, end_year: int | None = None) -> None:
    """Orchestrate full workflow."""
```

**Çıktı schema:**
```
date: datetime
profile_residential: float (aggregate LV+MV)
profile_residential_lv: float
profile_residential_mv: float
profile_industrial: float (aggregate)
profile_industrial_lv: float
profile_industrial_mv: float
profile_commercial: float (aggregate)
profile_commercial_lv: float
profile_commercial_mv: float
profile_agricultural_irrigation: float (aggregate)
profile_agricultural_irrigation_lv: float
profile_agricultural_irrigation_mv: float
profile_lighting: float (combined from city variants)
profile_government: float
profile_other: float
```

### 5.9 `src/energy_forecast/data/__init__.py`

```python
from energy_forecast.data.loader import DataLoader
from energy_forecast.data.epias_client import EpiasClient
from energy_forecast.data.openmeteo_client import OpenMeteoClient
from energy_forecast.data.exceptions import (
    DataError, DataValidationError,
    EpiasApiError, EpiasAuthError, OpenMeteoApiError,
)

__all__ = [
    "DataLoader", "EpiasClient", "OpenMeteoClient",
    "DataError", "DataValidationError",
    "EpiasApiError", "EpiasAuthError", "OpenMeteoApiError",
]
```

## 6. Test Stratejisi

### 6.1 Fixtures (`tests/conftest.py` güncellemesi)

```python
@pytest.fixture()
def sample_excel_df() -> pd.DataFrame:
    """Minimal valid consumption DataFrame (3 days = 72 rows)."""

@pytest.fixture()
def sample_excel_path(tmp_path, sample_excel_df) -> Path:
    """Write sample consumption to Excel file."""

@pytest.fixture()
def sample_epias_response() -> dict[str, Any]:
    """Mock EPIAS API JSON response for single variable."""

@pytest.fixture()
def sample_openmeteo_response() -> dict[str, Any]:
    """Mock OpenMeteo API JSON response with 11 variables."""

@pytest.fixture()
def data_loader_config(settings) -> DataLoaderConfig:
    """DataLoaderConfig from project settings."""

@pytest.fixture()
def openmeteo_config(settings) -> OpenMeteoConfig:
    """OpenMeteoConfig from project settings."""

@pytest.fixture()
def region_config(settings) -> RegionConfig:
    """RegionConfig from project settings."""
```

### 6.2 `tests/unit/test_data/test_loader.py` (~12 test)

| # | Test | Tip |
|---|------|-----|
| 1 | `test_load_valid_excel` | Happy path — 3 gün Excel → 72 satır DataFrame |
| 2 | `test_datetime_index_created` | DatetimeIndex, freq="h" |
| 3 | `test_date_time_merge_correct` | `2024-01-01` + `15` → `2024-01-01 15:00` |
| 4 | `test_continuous_index_fills_gaps` | Eksik saatler NaN ile doldurulur |
| 5 | `test_missing_column_raises` | `date` kolonu yoksa → DataValidationError |
| 6 | `test_invalid_time_range_raises` | time=25 → DataValidationError |
| 7 | `test_negative_consumption_raises` | consumption=-10 → DataValidationError |
| 8 | `test_empty_excel_raises` | 0 satır → DataValidationError |
| 9 | `test_extend_for_forecast` | +48 saat NaN satır eklenir |
| 10 | `test_config_column_names_used` | Custom column mapping çalışır |
| 11 | `test_max_missing_ratio_exceeded` | %5+ eksik → DataValidationError |
| 12 | `test_file_not_found_raises` | Var olmayan path → FileNotFoundError |

### 6.3 `tests/unit/test_data/test_epias_client.py` (~11 test)

| # | Test | Tip |
|---|------|-----|
| 1 | `test_authenticate_returns_token` | Mocked POST → TGT string |
| 2 | `test_authenticate_failure_raises` | 401 → EpiasAuthError |
| 3 | `test_fetch_single_variable` | Mocked GET → DataFrame |
| 4 | `test_fetch_all_variables_merged` | 5 variable inner merge |
| 5 | `test_cache_hit_skips_api` | Cache varsa API çağrılmaz |
| 6 | `test_cache_miss_fetches_api` | Cache yoksa API çağrılır + cache yazılır |
| 7 | `test_rate_limit_delay` | İki istek arası >= rate_limit_seconds |
| 8 | `test_retry_on_server_error` | 500 → retry 3x → EpiasApiError |
| 9 | `test_date_range_filter` | Yıl verisi start-end aralığına filtrelenir |
| 10 | `test_output_schema_valid` | Çıktı EpiasSchema'ya uygun |
| 11 | `test_output_datetime_index` | DatetimeIndex doğru |

**Mock:** `httpx.Client` tamamen mock'lanır.

### 6.4 `tests/unit/test_data/test_openmeteo_client.py` (~10 test)

| # | Test | Tip |
|---|------|-----|
| 1 | `test_fetch_historical_returns_dataframe` | Mocked GET → DataFrame |
| 2 | `test_fetch_forecast_returns_dataframe` | Mocked forecast → DataFrame |
| 3 | `test_weighted_average_correct` | 4 şehir ağırlıklı ortalama doğruluğu |
| 4 | `test_all_11_variables_present` | 11 weather column mevcut |
| 5 | `test_datetime_index_hourly` | DatetimeIndex freq="h" |
| 6 | `test_api_error_raises` | 500 → OpenMeteoApiError |
| 7 | `test_retry_on_timeout` | Timeout → retry → başarılı |
| 8 | `test_config_variables_used` | Config'teki variable listesi kullanılır |
| 9 | `test_single_location_fetch` | Tek şehir fetch doğru |
| 10 | `test_output_schema_valid` | Çıktı WeatherSchema'ya uygun |

**Mock:** `httpx.Client` mock'lanır.

### 6.5 `tests/unit/test_data/test_schemas.py` (~5 test)

| # | Test | Tip |
|---|------|-----|
| 1 | `test_raw_excel_schema_valid` | Geçerli data → OK |
| 2 | `test_raw_excel_schema_invalid_time` | time=25 → SchemaError |
| 3 | `test_consumption_schema_valid` | Geçerli data → OK |
| 4 | `test_epias_schema_valid` | 5 kolon DatetimeIndex → OK |
| 5 | `test_weather_schema_valid` | 11 kolon DatetimeIndex → OK |

### 6.6 `tests/unit/test_scripts/test_generate_holidays.py` (~6 test)

| # | Test | Tip |
|---|------|-----|
| 1 | `test_generate_catalog_returns_dataframe` | Happy path |
| 2 | `test_catalog_has_required_columns` | date, holiday_name, raw_holiday_name |
| 3 | `test_holiday_name_is_categorical` | dtype == category |
| 4 | `test_collision_resolved` | Aynı günde 2 tatil → priority'ye göre tek seçilir |
| 5 | `test_known_holiday_mapped` | "Cumhuriyet Bayramı" → "Republic Day" |
| 6 | `test_year_range_correct` | start=2015, end=2044 → 30 yıl kapsanır |

**Mock:** Yok — `holidays` kütüphanesi offline çalışır.

## 7. Implementasyon Sırası

```
Adım 1:  pyproject.toml → holidays>=0.40 ekle, uv sync
Adım 2:  exceptions.py + schemas.py (bağımlılık yok, temiz başlangıç)
Adım 3:  loader.py (schemas + exceptions kullanır)
Adım 4:  epias_client.py (exceptions kullanır, auth + cache + fetch)
Adım 5:  openmeteo_client.py (exceptions + config kullanır)
Adım 6:  __init__.py güncelle (tüm exports)
Adım 7:  scripts/generate_holidays.py (bağımsız, offline)
Adım 8:  scripts/backfill_epias.py (EpiasClient gerektirir)
Adım 9:  scripts/fetch_profile_coefficients.py (EPIAS auth gerektirir)
Adım 10: conftest.py fixtures (test altyapısı)
Adım 11: test_schemas.py (en basit testler)
Adım 12: test_loader.py (Excel testleri)
Adım 13: test_epias_client.py (mock testler)
Adım 14: test_openmeteo_client.py (mock testler)
Adım 15: test_generate_holidays.py (offline testler)
Adım 16: lint + mypy pass (temizlik)
```

## 8. Teknik Kararlar

### 8.1 HTTP Client: httpx (sync)
- Zaten pyproject.toml'da mevcut (`>=0.27`)
- Sync şimdilik, M10'da async geçiş kolay
- Built-in timeout desteği

### 8.2 Retry: tenacity
- Zaten pyproject.toml'da mevcut (`>=8.2`)
- Exponential backoff, max retry, custom exception filter
- EPIAS rate limit: `tenacity.wait_fixed(10)` + `wait_exponential`

### 8.3 Cache
- **EPIAS market data:** Yıl bazlı parquet — `data/external/epias/{year}.parquet`
- **EPIAS profile coefficients:** Yıl bazlı parquet — `data/external/profile/{year}.parquet`
- **OpenMeteo:** SQLite backend, TTL-based — `data/external/weather_cache.db`
- Ayrı CacheManager YAZI**LMI**YOR — over-engineering

### 8.4 Validation: Pandera
- Zaten pyproject.toml'da mevcut (`>=0.18`)
- Schema validation at boundaries (input Excel, output DataFrames)

### 8.5 DatetimeIndex Standardı
- Tüm çıktılar: `pd.DatetimeIndex`, `freq="h"`
- Tz-naive standart (eski projede de bu pattern)
- Config'te `timezone: "Europe/Istanbul"` bilgi amaçlı tutulur

### 8.6 Profile Coefficients: Ayrı Script
- EpiasClient'a eklemek yerine ayrı script — farklı endpoint'ler, farklı auth flow
- `EpiasClient.authenticate()` pattern'ini paylaşır (aynı AUTH_URL)
- Profile name mapping YAML'a taşınabilir ama M2 scope'unda script-level dict yeterli

### 8.7 Holidays: `holidays` kütüphanesi
- Yeni bağımlılık: `holidays>=0.40`
- Offline çalışır, Hicri takvim hesaplaması dahil
- Parquet dosyası version control'e alınabilir (statik veri, nadiren değişir)

## 9. Bağımlılık Kontrolü

| Paket | pyproject.toml | Durum | Kullanım |
|-------|---------------|-------|----------|
| httpx | `>=0.27` | ✅ Mevcut | API requests |
| tenacity | `>=8.2` | ✅ Mevcut | Retry logic |
| pandera | `>=0.18` | ✅ Mevcut | Schema validation |
| openpyxl | `>=3.1` | ✅ Mevcut | Excel okuma |
| pyarrow | `>=15.0` | ✅ Mevcut | Parquet I/O |
| loguru | `>=0.7` | ✅ Mevcut | Logging |
| **holidays** | **YOK** | ❌ EKLENECEk | Türk tatilleri |

## 10. Çıkış Kriterleri

- [ ] `make test` → ~44 yeni test + 30 mevcut = ~74 test geçer
- [ ] `make lint` → `ruff check` + `mypy --strict` temiz
- [ ] `DataLoader.load_excel()` gerçek Excel dosyasıyla çalışır
- [ ] `EpiasClient.fetch()` mock ile test edilir, cache read/write çalışır
- [ ] `EpiasClient.fetch_year()` public method olarak backfill script'ten çağrılabilir
- [ ] `OpenMeteoClient.fetch_historical()` ve `fetch_forecast()` mock ile test edilir
- [ ] Ağırlıklı ortalama doğru: Bursa %60, Balıkesir %24, Yalova %10, Çanakkale %6
- [ ] Pandera schema validation input/output sınırlarında çalışır
- [ ] `scripts/generate_holidays.py` çalışır → `data/static/turkish_holidays.parquet`
- [ ] `scripts/backfill_epias.py` mock ile test edilebilir
- [ ] `scripts/fetch_profile_coefficients.py` mock ile test edilebilir
- [ ] Exception hierarchy tutarlı ve informative error messages

## 11. Commit Stratejisi

Bölünmüş commit'ler (tercih edilen):
```
feat(data): add exception hierarchy and Pandera schemas
feat(data): implement Excel consumption data loader
feat(data): implement EPIAS API client with year-based caching
feat(data): implement OpenMeteo client with weighted-average weather
feat(scripts): add Turkish holiday generator script
feat(scripts): add EPIAS cache backfill and profile coefficient scripts
test(data): add comprehensive unit tests for data pipeline
```

## 12. Kapsam Dışı (Sonraki Milestone'lar)

- Feature engineering (M3)
- End-to-end pipeline orchestrator (M3)
- Async HTTP (M10 — FastAPI serving)
- Real data ile integration test (M3 sonunda)
- Profile name mapping'i YAML'a taşıma (nice-to-have, gerekirse M3'te)
