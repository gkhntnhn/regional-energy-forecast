# M1: Config System — Implementation Plan

## 1. Kapsam

**Milestone:** M1 — Config sistemi
**Bağımlılık:** M0 (proje iskeleti) — tamamlandı ✅
**Hedef:** Tüm YAML config dosyalarını Pydantic V2 modelleriyle type-safe yükleyen,
valide eden ve projenin her yerinden erişilebilir kılan merkezi config sistemi.

## 2. Mevcut Durum

- 13 YAML config dosyası **zaten mevcut** (`configs/` altında)
- `Settings` class'ı **placeholder** (`NotImplementedError`)
- `config/__init__.py` sadece `Settings` export ediyor
- Bağımlılıklar (`pydantic`, `pydantic-settings`, `pyyaml`) pyproject.toml'da **mevcut**

## 3. Referans Proje Analizi

Eski proje (`distributed-energy-forecasting`) şu pattern'i kullanıyor:
- Pydantic V2 `BaseModel` + `BaseSettings` ile nested config class'lar
- `load_modular_config()` fonksiyonu: tüm YAML'ları okur → merge → Settings oluşturur
- `.env` dosyasından secret'lar `pydantic-settings` ile yüklenir
- Her config grubu ayrı bir nested model (LocationConfig, LoggingConfig, vb.)

**Taşınacak mantık:** Modüler YAML loader + nested Pydantic validation pattern
**Taşınmayacak:** Database, Redis config'leri (bu projede yok), aşırı derin nesting

## 4. Dosya Planı

### 4.1 Düzenlenecek Dosyalar

| Dosya | Mevcut | İşlem |
|-------|--------|-------|
| `src/energy_forecast/config/settings.py` | 30 satır placeholder | ~350 satır — tüm config modelleri + loader |
| `src/energy_forecast/config/__init__.py` | 5 satır | ~15 satır — public API export |
| `tests/conftest.py` | 26 satır | +20 satır — config fixtures |

### 4.2 Yeni Oluşturulacak Dosyalar

| Dosya | Tahmini Boyut | İçerik |
|-------|--------------|--------|
| `tests/unit/test_config.py` | ~250 satır | Config unit testleri |

## 5. Implementation Detayları

### 5.1 `src/energy_forecast/config/settings.py`

Tek dosyada tüm config modelleri + loader. Modüller:

#### A) City & Region Config
```python
class CityConfig(BaseModel):
    name: str
    weight: float = Field(ge=0.0, le=1.0)
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)

class RegionConfig(BaseModel):
    name: str
    cities: list[CityConfig]

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> Self: ...
```

#### B) Logging Config
```python
class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: str
    rotation: str = "10 MB"
    retention: str = "30 days"
```

#### C) Forecast Config
```python
class ForecastConfig(BaseModel):
    horizon_hours: int = Field(default=48, ge=1)
    frequency: str = "1h"
    min_lag: int = Field(default=48, ge=1)
```

#### D) Pipeline Config
```python
class PipelineConfig(BaseModel):
    modules: list[str]
    merge_strategy: Literal["left", "inner", "outer"] = "left"
    drop_raw_epias: bool = True
    validate_output: bool = True
    check_duplicate_columns: bool = True
```

#### E) Data Loader Config
```python
class ExcelColumnsConfig(BaseModel):
    date: str = "date"
    time: str = "time"
    consumption: str = "consumption"

class ValidationConfig(BaseModel):
    min_consumption: float = Field(default=0.0, ge=0.0)
    max_consumption: float = Field(default=10000.0, gt=0.0)
    max_missing_ratio: float = Field(default=0.05, ge=0.0, le=1.0)
    expected_frequency: str = "1h"

class PathsConfig(BaseModel):
    raw: Path = Path("data/raw")
    processed: Path = Path("data/processed")
    static: Path = Path("data/static")
    holidays: Path = Path("data/static/turkish_holidays.parquet")

class DataLoaderConfig(BaseModel):
    excel: ExcelColumnsConfig
    validation: ValidationConfig
    paths: PathsConfig
```

#### F) OpenMeteo Config
```python
class OpenMeteoApiConfig(BaseModel):
    base_url_historical: str
    base_url_forecast: str
    timeout: int = Field(default=30, ge=1)
    retry_attempts: int = Field(default=3, ge=1)

class WeatherCacheConfig(BaseModel):
    backend: Literal["sqlite"] = "sqlite"
    path: str = "data/external/weather_cache.db"
    ttl_hours: int = Field(default=6, ge=1)

class OpenMeteoConfig(BaseModel):
    api: OpenMeteoApiConfig
    variables: list[str]
    cache: WeatherCacheConfig
```

#### G) Feature Configs (5 modül)
```python
class ConsumptionLagConfig(BaseModel):
    min_lag: int = Field(default=48, ge=48)  # LEAKAGE GUARD
    values: list[int]

    @field_validator("values")
    @classmethod
    def all_lags_ge_min(cls, v: list[int], info: ValidationInfo) -> list[int]: ...

class ConsumptionRollingConfig(BaseModel):
    windows: list[int]
    functions: list[str]

class ConsumptionConfig(BaseModel):
    lags: ConsumptionLagConfig
    rolling: ConsumptionRollingConfig
    ewma: EwmaConfig
    expanding: ExpandingConfig    # min_periods >= 48 validator
    momentum: MomentumConfig
    quantile: QuantileConfig
```

Benzer pattern: `CalendarConfig`, `WeatherFeaturesConfig`, `SolarConfig`, `EpiasConfig`

#### H) Model Configs
```python
class CatBoostTrainingConfig(BaseModel):
    task_type: Literal["CPU", "GPU"] = "CPU"
    iterations: int = Field(default=2000, ge=100)
    learning_rate: float = Field(default=0.05, gt=0.0, lt=1.0)
    depth: int = Field(default=6, ge=1, le=16)
    loss_function: str = "RMSE"
    eval_metric: str = "MAPE"
    early_stopping_rounds: int = Field(default=200, ge=1)
    has_time: bool = True
    random_seed: int = 42
    verbose: int = 100

class CatBoostConfig(BaseModel):
    training: CatBoostTrainingConfig
    categorical_features: list[str]
    nan_handling: dict[str, str]
```

Benzer: `ProphetConfig`, `TFTConfig`, `HyperparameterConfig`, `CrossValidationConfig`

#### I) Environment Config (secrets)
```python
class EnvConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_env: Literal["development", "production"] = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = ""
    epias_username: str = ""
    epias_password: str = ""
    mlflow_tracking_uri: str = "http://localhost:5000"
```

#### J) Root Settings & Loader
```python
class Settings(BaseModel):
    """Root config — tüm YAML config'leri birleştirir."""
    project: ProjectConfig
    logging: LoggingConfig
    region: RegionConfig
    forecast: ForecastConfig
    pipeline: PipelineConfig
    data_loader: DataLoaderConfig
    openmeteo: OpenMeteoConfig
    features: FeaturesConfig          # calendar + consumption + weather + solar + epias
    catboost: CatBoostConfig
    prophet: ProphetConfig
    tft: TFTConfig
    hyperparameters: HyperparameterConfig
    env: EnvConfig

def load_config(config_dir: Path | None = None) -> Settings:
    """Tüm YAML dosyalarını okur, merge eder, Settings döndürür."""
    ...

def get_default_config() -> Settings:
    """Test ve development için default config."""
    ...
```

### 5.2 `src/energy_forecast/config/__init__.py`

```python
from energy_forecast.config.settings import (
    Settings,
    EnvConfig,
    load_config,
    get_default_config,
)

__all__ = ["Settings", "EnvConfig", "load_config", "get_default_config"]
```

### 5.3 YAML Loader Mantığı

```
load_config(config_dir)
  ├── settings.yaml → project, logging, region, forecast
  ├── pipeline.yaml → pipeline
  ├── data_loader.yaml → data_loader
  ├── openmeteo.yaml → openmeteo
  ├── features/calendar.yaml → features.calendar
  ├── features/consumption.yaml → features.consumption
  ├── features/weather.yaml → features.weather
  ├── features/solar.yaml → features.solar
  ├── features/epias.yaml → features.epias
  ├── models/catboost.yaml → catboost
  ├── models/prophet.yaml → prophet
  ├── models/tft.yaml → tft
  ├── models/hyperparameters.yaml → hyperparameters
  └── .env → env (pydantic-settings auto)
```

Her YAML dosyası `yaml.safe_load()` ile okunur, nested dict olarak merge edilir,
sonra `Settings(**merged_dict)` ile Pydantic validation çalışır.

## 6. Leakage Guard Validators

Config seviyesinde leakage kurallarını zorlamak:

```python
# ConsumptionLagConfig
@field_validator("values")
def all_lags_ge_min(cls, v, info):
    min_lag = info.data.get("min_lag", 48)
    for lag in v:
        if lag < min_lag:
            raise ValueError(f"Lag {lag} < min_lag {min_lag} — leakage riski!")
    return v

# ExpandingConfig
@field_validator("min_periods")
def min_periods_ge_48(cls, v):
    if v < 48:
        raise ValueError(f"min_periods {v} < 48 — leakage riski!")
    return v

# EpiasConfig
@field_validator("values")  # lags altında
def epias_lags_ge_min(cls, v, info):
    # Aynı min_lag=48 kontrolü
    ...
```

## 7. Test Stratejisi

### 7.1 `tests/unit/test_config.py`

| # | Test Senaryosu | Tip |
|---|----------------|-----|
| 1 | `test_load_config_from_project_yamls` | Happy path — gerçek YAML'lardan yükleme |
| 2 | `test_settings_has_all_sections` | Root config'te tüm alanlar mevcut |
| 3 | `test_region_weights_sum_to_one` | Validator — ağırlık toplamı 1.0 |
| 4 | `test_region_weights_invalid_sum` | Validator — hatalı toplam → ValidationError |
| 5 | `test_consumption_lag_min_48` | Leakage guard — min_lag enforced |
| 6 | `test_consumption_lag_below_min_raises` | Leakage guard — lag < 48 → hata |
| 7 | `test_expanding_min_periods_ge_48` | Leakage guard — min_periods enforced |
| 8 | `test_expanding_min_periods_below_48_raises` | Leakage guard — min_periods < 48 → hata |
| 9 | `test_epias_lag_min_48` | Leakage guard — EPIAS lag'lar da ≥ 48 |
| 10 | `test_pipeline_drop_raw_epias_default_true` | Default değer kontrolü |
| 11 | `test_env_config_defaults` | EnvConfig default'lar doğru |
| 12 | `test_env_config_from_dotenv` | .env dosyasından yükleme (tmp_path) |
| 13 | `test_invalid_yaml_raises` | Bozuk YAML → ConfigError |
| 14 | `test_missing_yaml_raises` | Eksik YAML → FileNotFoundError |
| 15 | `test_get_default_config` | Default config oluşturulabilir |
| 16 | `test_catboost_has_time_true` | has_time default'u zorunlu true |
| 17 | `test_openmeteo_variables_list` | 11 hava durumu değişkeni mevcut |
| 18 | `test_forecast_horizon_48` | Horizon default 48 |
| 19 | `test_data_loader_paths` | Path'ler doğru |
| 20 | `test_config_immutable` | Frozen model — mutation denemesi hata verir |

### 7.2 Fixtures (`tests/conftest.py`)

```python
@pytest.fixture()
def settings(configs_dir: Path) -> Settings:
    """Load Settings from project configs."""
    return load_config(configs_dir)

@pytest.fixture()
def default_settings() -> Settings:
    """Get default Settings (no YAML files needed)."""
    return get_default_config()
```

### 7.3 Mock Gereksinimi

**Yok** — Config sistemi tamamen lokal dosya okur, external API çağrısı yok.
Sadece `.env` testi için `tmp_path` ve `monkeypatch` kullanılır.

## 8. Çıkış Kriterleri

- [ ] `make test` → 20 config testi geçer
- [ ] `make lint` → ruff + mypy strict temiz
- [ ] `load_config()` gerçek YAML dosyalarıyla çalışır
- [ ] `get_default_config()` YAML olmadan çalışır
- [ ] Leakage guard validator'lar config seviyesinde koruma sağlar
- [ ] Commit: `feat(config): implement Pydantic V2 config system with YAML loader`

## 9. Kapsam Dışı (Sonraki Milestone'lar)

- YAML dosyalarının içerik değişikliği (mevcut değerler yeterli)
- Config'i kullanan modüllerin güncellenmesi (M2+ sırasında)
- Runtime config override (env var ile YAML override — gerek yok şimdilik)
