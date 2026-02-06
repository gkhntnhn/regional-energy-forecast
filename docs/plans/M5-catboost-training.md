# M5: CatBoost Training — Implementasyon Plani

> Tarih: 2026-02-07
> Bagimlilik: M0-M4 tamamlanmis olmali (tumu tamam)
> Onceligi: P0 — ilk model, tum training altyapisi burada kurulur

---

## 1. Kapsam

M5, CatBoost modelinin egitim pipeline'ini kurar. Bu milestone ayni zamanda
tum modellerin (Prophet, TFT) kullanacagi **ortak training altyapisini** da icerir:

- Custom TSCV splitter — takvim ayi bazli (ORTAK — tum modeller)
- Metrik fonksiyonlari (ORTAK)
- Dinamik Optuna search space parser (ORTAK)
- MLflow experiment tracker (ORTAK)
- CatBoost trainer (model-spesifik)
- CLI entry point (`make train-catboost`)

---

## 2. Dosya Plani

### 2.1 Yeni Olusturulacak Dosyalar

| # | Dosya | Satir (hedef) | Aciklama |
|---|-------|---------------|----------|
| 1 | `src/energy_forecast/training/splitter.py` | ~130 | Custom TSCV — takvim ayi bazli |
| 2 | `src/energy_forecast/training/search.py` | ~40 | Dinamik Optuna search space parser |
| 3 | `src/energy_forecast/training/experiment.py` | ~130 | MLflow experiment tracker |
| 4 | `src/energy_forecast/training/catboost_trainer.py` | ~250 | CatBoost egitim + Optuna |
| 5 | `src/energy_forecast/training/run.py` | ~120 | CLI entry point (`python -m`) |
| 6 | `tests/unit/test_training/__init__.py` | 1 | Package init |
| 7 | `tests/unit/test_training/test_splitter.py` | ~220 | Splitter unit testleri |
| 8 | `tests/unit/test_training/test_metrics.py` | ~150 | Metrik unit testleri |
| 9 | `tests/unit/test_training/test_search.py` | ~80 | Search space parser testleri |
| 10 | `tests/unit/test_training/test_catboost_trainer.py` | ~250 | CatBoost trainer testleri |
| 11 | `tests/unit/test_training/test_experiment.py` | ~120 | MLflow tracker testleri |

### 2.2 Degistirilecek Dosyalar

| # | Dosya | Degisiklik |
|---|-------|-----------|
| 1 | `src/energy_forecast/training/__init__.py` | Public API export'lari ekle |
| 2 | `src/energy_forecast/training/metrics.py` | Stub → gercek implementasyon |
| 3 | `src/energy_forecast/models/catboost.py` | Stub → gercek implementasyon |
| 4 | `src/energy_forecast/models/base.py` | Kucuk iyilestirmeler |
| 5 | `src/energy_forecast/config/settings.py` | Dinamik search space + ay bazli CV config |
| 6 | `configs/models/hyperparameters.yaml` | Dinamik format + ay bazli |
| 7 | `pyproject.toml` | mypy overrides (mlflow, catboost, optuna) |

### 2.3 Silinecek Dosyalar

| Dosya | Neden |
|-------|-------|
| `src/energy_forecast/training/cross_validation.py` | `splitter.py` ile degistiriliyor |
| `src/energy_forecast/training/tuning.py` | `search.py` + `catboost_trainer.py` ile degistiriliyor |

---

## 3. Adim Adim Implementasyon

### Adim 1: Config — Dinamik Search Space + Ay Bazli CV

**Dosya:** `src/energy_forecast/config/settings.py`

#### Silinecek modeller:
- `SearchRangeConfig` (satir 667-673)
- `CatBoostSearchSpace` (satir 676-701)

#### Yeni modeller:

```python
class SearchParamConfig(BaseModel, frozen=True):
    """Tek bir Optuna search parametresinin tanimi.

    YAML'dan dinamik olarak yuklenir. Yeni parametre eklemek icin
    sadece YAML'a satir eklemek yeterli — kod degismez.

    type=int:         trial.suggest_int(name, low, high, step?, log?)
    type=float:       trial.suggest_float(name, low, high, step?, log?)
    type=categorical: trial.suggest_categorical(name, choices)
    """

    type: Literal["int", "float", "categorical"]
    low: float | None = None
    high: float | None = None
    step: float | None = None
    log: bool = False
    choices: list[Any] | None = None

    @model_validator(mode="after")
    def _validate_range_or_choices(self) -> SearchParamConfig:
        if self.type in ("int", "float"):
            if self.low is None or self.high is None:
                raise ValueError(f"type={self.type} requires low and high")
            if self.low > self.high:
                raise ValueError(f"low ({self.low}) > high ({self.high})")
            if self.log and self.step is not None:
                raise ValueError("log=true and step are mutually exclusive")
        elif self.type == "categorical":
            if not self.choices or len(self.choices) < 1:
                raise ValueError("type=categorical requires non-empty choices")
        return self


class ModelSearchConfig(BaseModel, frozen=True):
    """Bir model icin Optuna search space + trial sayisi."""

    n_trials: int = Field(default=50, ge=1)
    search_space: dict[str, SearchParamConfig] = Field(default_factory=dict)
```

#### Guncellenen modeller:

```python
class CrossValidationConfig(BaseModel, frozen=True):
    """Takvim ayi bazli TSCV ayarlari.

    val_months ve test_months takvim ayi olarak sayilir, sabit gun
    sayisi DEGIL. Ornek: val_months=1, test_months=1 iken
    Ekim → train bitis, Kasim → val, Aralik → test.
    Her split tam takvim ayina hizalanir.
    """

    n_splits: int = Field(default=12, ge=2)
    val_months: int = Field(default=1, ge=1)       # takvim ayi
    test_months: int = Field(default=1, ge=1)       # takvim ayi
    gap_hours: int = Field(default=0, ge=0)
    shuffle: bool = False


class HyperparameterConfig(BaseModel, frozen=True):
    catboost: ModelSearchConfig = Field(default_factory=ModelSearchConfig)
    prophet: ModelSearchConfig = Field(default_factory=ModelSearchConfig)
    tft: ModelSearchConfig = Field(default_factory=ModelSearchConfig)
    cross_validation: CrossValidationConfig = Field(
        default_factory=CrossValidationConfig,
    )
    target_col: str = "consumption"
```

#### `_build_settings_dict` guncelleme:

```python
"hyperparameters": {
    "catboost": {
        "n_trials": hyperparams_data.get("catboost", {}).get("n_trials", 50),
        "search_space": hyperparams_data.get("catboost", {}).get("search_space", {}),
    },
    "prophet": {
        "n_trials": hyperparams_data.get("prophet", {}).get("n_trials", 30),
        "search_space": hyperparams_data.get("prophet", {}).get("search_space", {}),
    },
    "tft": {
        "n_trials": hyperparams_data.get("tft", {}).get("n_trials", 20),
        "search_space": hyperparams_data.get("tft", {}).get("search_space", {}),
    },
    "cross_validation": hyperparams_data.get("cross_validation", {}),
    "target_col": hyperparams_data.get("target_col", "consumption"),
},
```

#### YAML (`configs/models/hyperparameters.yaml`):

```yaml
# Optuna hyperparameter search spaces
# Her parametre: type (int/float/categorical), low, high, step?, log?, choices?
# YAML'a yeni parametre eklemek yeterli — kod degismez

target_col: consumption

catboost:
  n_trials: 50
  search_space:
    iterations:
      type: int
      low: 1000
      high: 3000
    learning_rate:
      type: float
      low: 0.01
      high: 0.1
      log: true
    depth:
      type: int
      low: 4
      high: 7
    l2_leaf_reg:
      type: float
      low: 1.0
      high: 10.0
    min_child_samples:
      type: int
      low: 5
      high: 100
    subsample:
      type: float
      low: 0.6
      high: 1.0
    loss_function:
      type: categorical
      choices: ["RMSE", "MAE"]

prophet:
  n_trials: 30
  search_space: {}

tft:
  n_trials: 20
  search_space: {}

cross_validation:
  n_splits: 12
  val_months: 1
  test_months: 1
  gap_hours: 0
  shuffle: false
```

---

### Adim 2: Custom TSCV Splitter — Takvim Ayi Bazli (~130 satir)

**Dosya:** `src/energy_forecast/training/splitter.py`

**Eski projeden referans:** `src/training/cross_validation.py`

#### Tasarim:

```python
@dataclass(frozen=True)
class SplitInfo:
    """Tek bir CV split'inin sinir bilgileri."""
    split_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp      # Bu ayin son saati (dahil)
    val_start: pd.Timestamp      # Takvim ayi baslangici
    val_end: pd.Timestamp        # Takvim ayi sonu (dahil)
    test_start: pd.Timestamp     # Takvim ayi baslangici
    test_end: pd.Timestamp       # Takvim ayi sonu (dahil)


class TimeSeriesSplitter:
    """Takvim ayi bazli expanding-window TSCV splitter.

    Her split tam takvim aylarina hizalanir:
    - val ve test bloklari tam takvim aylari (28-31 gun)
    - train baslangictan val_start'a kadar genisler (expanding)

    n_splits=12, val_months=1, test_months=1 ile:
      Split  1: train=[...Oca] val=Sub       test=Mar
      Split  2: train=[...Sub] val=Mar       test=Nis
      ...
      Split 12: train=[...Ara] val=Oca(+1y)  test=Sub(+1y)

    Bu sayede her takvim ayinin MAPE'si ayri raporlanir.
    """

    def __init__(
        self,
        n_splits: int = 12,
        val_months: int = 1,
        test_months: int = 1,
        gap_hours: int = 0,
    ) -> None: ...

    @classmethod
    def from_config(cls, cv_config: CrossValidationConfig) -> TimeSeriesSplitter: ...

    def split(self, df: pd.DataFrame) -> list[SplitInfo]: ...

    def iter_splits(
        self, df: pd.DataFrame,
    ) -> Iterator[tuple[SplitInfo, pd.DataFrame, pd.DataFrame, pd.DataFrame]]: ...
```

#### Algoritma (ay bazli, backward-anchored, expanding):

```python
def split(self, df: pd.DataFrame) -> list[SplitInfo]:
    idx = df.index  # DatetimeIndex
    data_start = idx.min()

    # Verinin kapsadigi son tam ayin sonunu bul
    last_month_end = idx.max().normalize().replace(day=1) + MonthEnd(0)
    # Eger veri ayin son saatine kadar gelmemisse, bir onceki ay sonuna git
    if idx.max() < last_month_end:
        last_month_end = last_month_end - MonthBegin(1) + MonthEnd(0)
    # VEYA daha basit: pd.Timestamp(year, month, 1) + MonthEnd(0) pattern'i

    block_months = self.val_months + self.test_months  # 1+1=2 ay/block
    splits: list[SplitInfo] = []

    for i in range(self.n_splits):
        # Sondan geriye dogru: her block 2 ay (val+test)
        # i=0 → en son block, i=n_splits-1 → en eski block
        offset = i * block_months  # ay cinsinden geri kayma

        # Test ayi sinirlarini hesapla
        test_end_month = last_month_end - DateOffset(months=offset)
        test_start_month = test_end_month - DateOffset(months=self.test_months - 1)
        test_start = test_start_month.replace(day=1)
        test_end = test_end_month  # ayin son gunu

        # Val ayi sinirlarini hesapla
        val_end_month = test_start - DateOffset(months=0)  # test_start'in onceki ay sonu
        # Daha dogru:
        val_end = test_start - pd.Timedelta(hours=1 + self.gap_hours)
        val_start_month = val_end.normalize().replace(day=1)
        # val_months=1 ise val tam 1 takvim ayi
        val_start = (test_start - DateOffset(months=self.val_months)).replace(day=1)
        val_end = test_start - pd.Timedelta(hours=1)  # gap_hours=0 → test_start - 1h
        if self.gap_hours > 0:
            val_end = val_end - pd.Timedelta(hours=self.gap_hours)

        # Train: baslangictan val_start'a kadar
        train_start = data_start
        train_end = val_start - pd.Timedelta(hours=1)
        if self.gap_hours > 0:
            train_end = train_end - pd.Timedelta(hours=self.gap_hours)

        # Guard: yeterli train verisi var mi?
        if train_end <= train_start:
            break

        splits.append(SplitInfo(
            split_idx=len(splits),
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            test_start=test_start,
            test_end=test_end,
        ))

    # Kronolojik siraya cevir (en eski split = 0)
    splits.reverse()
    for i, s in enumerate(splits):
        splits[i] = SplitInfo(split_idx=i, ...)  # idx'leri yeniden numaralandir
    return splits
```

#### Somut Ornek (3 yillik veri: 2022-01 → 2024-12, n_splits=12):

```
Split  0: train=[2022-01 ... 2023-01] val=2023-02       test=2023-03
Split  1: train=[2022-01 ... 2023-02] val=2023-03       test=2023-04
Split  2: train=[2022-01 ... 2023-03] val=2023-04       test=2023-05
Split  3: train=[2022-01 ... 2023-04] val=2023-05       test=2023-06
Split  4: train=[2022-01 ... 2023-05] val=2023-06       test=2023-07
Split  5: train=[2022-01 ... 2023-06] val=2023-07       test=2023-08
Split  6: train=[2022-01 ... 2023-07] val=2023-08       test=2023-09
Split  7: train=[2022-01 ... 2023-08] val=2023-09       test=2023-10
Split  8: train=[2022-01 ... 2023-09] val=2023-10       test=2023-11
Split  9: train=[2022-01 ... 2023-10] val=2023-11       test=2023-12
Split 10: train=[2022-01 ... 2023-11] val=2023-12       test=2024-01
Split 11: train=[2022-01 ... 2023-12] val=2024-01       test=2024-02
                                       ↑ expanding      ↑ tam takvim ayi
```

**Ay siniri:** Her ay 1. gun 00:00 baslar, son gun 23:00 biter.
- Subat: 28 gun (veya 29), Mart: 31 gun, Nisan: 30 gun → dogal farklilik.
- Bu sayede "Subat MAPE: 5.2%, Temmuz MAPE: 8.1%" gibi ay bazli karsilastirma yapilir.

**Saatlik sinirlar (detay):**
```
train_end  = val_start ayi 1. gun 00:00 - 1h  = onceki ayin son gunu 23:00
val_start  = val ayi 1. gun 00:00
val_end    = test ayi 1. gun 00:00 - 1h        = val ayinin son gunu 23:00
test_start = test ayi 1. gun 00:00
test_end   = test ayinin son gunu 23:00
```

**gap_hours > 0 ise:**
```
train_end  = val_start - 1h - gap_hours
val_end    = test_start - 1h - gap_hours
```

**Onemli:**
- ASLA shuffle yok — zaman sirasi korunur
- gap_hours=0 default — min_lag=48 zaten koruma sagliyor
- Tum modeller (CatBoost, Prophet, TFT) AYNI splitter'i kullanir
- Ay bazli split → "hangi ayda model kotu?" sorusunu yanitlar

**Silinecek:** `cross_validation.py`

---

### Adim 3: Metrik Fonksiyonlari (~80 satir)

**Dosya:** `src/energy_forecast/training/metrics.py` (mevcut stub'u guncelle)

Mevcut: 7 fonksiyon, hepsi `NotImplementedError`. Gercek implementasyona gecis.

```python
def mape(y_true, y_pred) -> float:
    """MAPE (%). y_true=0 olan noktalar cikarilir."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def r_squared(y_true, y_pred) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

def smape(y_true, y_pred) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    return float(np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)

def wmape(y_true, y_pred) -> float:
    total = np.sum(np.abs(y_true))
    return float(np.sum(np.abs(y_true - y_pred)) / total * 100) if total != 0 else 0.0

def mbe(y_true, y_pred) -> float:
    return float(np.mean(y_pred - y_true))
```

**Ek — toplu hesaplama:**

```python
@dataclass(frozen=True)
class MetricsResult:
    mape: float
    mae: float
    rmse: float
    r2: float
    smape: float
    wmape: float
    mbe: float

def compute_all(y_true, y_pred) -> MetricsResult:
    """Tum metrikleri tek seferde hesapla."""
    ...
```

---

### Adim 4: Dinamik Optuna Search Space Parser (~40 satir)

**Dosya:** `src/energy_forecast/training/search.py`

```python
"""Dinamik Optuna search space → trial parameter cevirici.

YAML'daki search_space dict'ini okur, her parametre icin
uygun trial.suggest_* cagrisini otomatik yapar.
YAML'a yeni parametre eklendiginde bu dosya DEGISMEZ.

Tum modeller (CatBoost, Prophet, TFT) ayni fonksiyonu kullanir.
"""

def suggest_params(
    trial: Trial,
    search_space: dict[str, SearchParamConfig],
) -> dict[str, Any]:
    """Search space config'den Optuna trial parametreleri uret.

    Args:
        trial: Optuna trial nesnesi.
        search_space: {param_name: SearchParamConfig} dict'i.

    Returns:
        {param_name: suggested_value} dict'i.
    """
    params: dict[str, Any] = {}
    for name, cfg in search_space.items():
        if cfg.type == "int":
            params[name] = trial.suggest_int(
                name, int(cfg.low), int(cfg.high),
                step=int(cfg.step) if cfg.step is not None else 1,
                log=cfg.log,
            )
        elif cfg.type == "float":
            kwargs: dict[str, Any] = {
                "name": name, "low": cfg.low, "high": cfg.high, "log": cfg.log,
            }
            if cfg.step is not None:
                kwargs["step"] = cfg.step
            params[name] = trial.suggest_float(**kwargs)
        elif cfg.type == "categorical":
            params[name] = trial.suggest_categorical(name, cfg.choices)
    return params
```

**Akis:**

```
YAML'a eklenen yeni parametre        suggest_params()              Optuna
────────────────────────────         ────────────────              ──────
bootstrap_type:                      trial.suggest_categorical(    "Bernoulli"
  type: categorical         ──→        "bootstrap_type",    ──→
  choices: [Bayesian, Bernoulli]       [...])

random_strength:                     trial.suggest_float(          3.7
  type: float               ──→        "random_strength",   ──→
  low: 0.1                             0.1, 10.0)
  high: 10.0

→ KOD DEGISMEZ. Sadece YAML guncellenir.
```

---

### Adim 5: MLflow Experiment Tracker (~130 satir)

**Dosya:** `src/energy_forecast/training/experiment.py`

```python
class ExperimentTracker:
    """MLflow experiment tracker. Tum modeller kullanir.

    Context manager olarak calisir:
        with tracker.start_run("trial_001") as run_id:
            tracker.log_params(params)
            tracker.log_metrics(metrics_dict)
    """

    def __init__(
        self,
        experiment_name: str = "energy-forecast",
        tracking_uri: str = "http://localhost:5000",
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled
        if enabled:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)

    @contextmanager
    def start_run(self, run_name: str) -> Iterator[str | None]:
        """MLflow run baslat. enabled=False ise noop."""
        ...

    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...
    def log_model(self, model: Any, artifact_path: str = "model") -> None: ...
    def log_feature_importance(
        self, importance: dict[str, float], top_n: int = 20,
    ) -> None: ...
    def log_split_metrics(
        self, split_idx: int, train_metrics: MetricsResult,
        val_metrics: MetricsResult, test_metrics: MetricsResult,
    ) -> None: ...
```

**Onemli:**
- `enabled=False` → tum log cagrilari noop (test/development icin)
- `mlflow.*` mypy override'a eklenecek

---

### Adim 6: CatBoost Trainer + Dinamik Optuna (~250 satir)

**Dosya:** `src/energy_forecast/training/catboost_trainer.py`

#### Veri yapilari:

```python
@dataclass(frozen=True)
class SplitResult:
    """Tek bir CV split'inin egitim sonuclari."""
    split_idx: int
    train_metrics: MetricsResult
    val_metrics: MetricsResult
    test_metrics: MetricsResult
    best_iteration: int
    val_month: str               # "2024-03" — hangi ay (raporlama icin)
    test_month: str              # "2024-04"


@dataclass(frozen=True)
class TrainingResult:
    """Tum CV split'lerinin toplam sonuclari."""
    split_results: list[SplitResult]
    avg_val_mape: float
    avg_test_mape: float
    std_val_mape: float
    avg_best_iteration: int
    feature_names: list[str]


@dataclass(frozen=True)
class PipelineResult:
    """Tam training pipeline sonucu."""
    study: Study
    best_params: dict[str, Any]
    training_result: TrainingResult
    final_model: CatBoostRegressor
    training_time_seconds: float
```

`val_month` ve `test_month` alanlari sayesinde:
```
Split  0: val=2023-02 (MAPE: 5.1%)  test=2023-03 (MAPE: 4.8%)
Split  5: val=2023-07 (MAPE: 8.3%)  test=2023-08 (MAPE: 9.1%)  ← yaz aylarinda kotu
Split  9: val=2023-11 (MAPE: 4.2%)  test=2023-12 (MAPE: 5.5%)
```
→ Hangi ayda model kotu oldugu hemen gorulur.

#### CatBoostTrainer sinifi:

```python
class CatBoostTrainer:
    """CatBoost egitim pipeline'i: TSCV + Optuna + MLflow."""

    def __init__(
        self,
        settings: Settings,
        tracker: ExperimentTracker | None = None,
    ) -> None:
        self._settings = settings
        self._cb_config = settings.catboost
        self._hp_config = settings.hyperparameters
        self._search_config = settings.hyperparameters.catboost  # ModelSearchConfig
        self._tracker = tracker or ExperimentTracker(enabled=False)
        self._splitter = TimeSeriesSplitter.from_config(
            settings.hyperparameters.cross_validation
        )
        self._target_col = settings.hyperparameters.target_col

    # ── X/y ayirimi (M4 leakage audit uyarisini cozer) ──

    def _split_xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        y = df[self._target_col]
        X = df.drop(columns=[self._target_col])
        return X, y

    # ── Kategorik feature hazirligi ──

    def _prepare_categoricals(self, X: pd.DataFrame) -> list[int]:
        cat_cols = [c for c in self._cb_config.categorical_features if c in X.columns]
        for col in cat_cols:
            X[col] = X[col].fillna(self._cb_config.nan_handling.categorical).astype(str)
        return [X.columns.get_loc(c) for c in cat_cols]

    # ── Tek split egitimi ──

    def _train_split(
        self,
        split_info: SplitInfo,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        params: dict[str, Any],
    ) -> SplitResult:
        X_train, y_train = self._split_xy(train_df)
        X_val, y_val = self._split_xy(val_df)
        X_test, y_test = self._split_xy(test_df)

        cat_indices = self._prepare_categoricals(X_train)
        self._prepare_categoricals(X_val)
        self._prepare_categoricals(X_test)

        train_pool = Pool(X_train, label=y_train, cat_features=cat_indices)
        val_pool = Pool(X_val, label=y_val, cat_features=cat_indices)

        model = CatBoostRegressor(**params)
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=self._cb_config.training.early_stopping_rounds,
            verbose=self._cb_config.training.verbose,
        )

        return SplitResult(
            split_idx=split_info.split_idx,
            train_metrics=compute_all(y_train.values, model.predict(X_train)),
            val_metrics=compute_all(y_val.values, model.predict(X_val)),
            test_metrics=compute_all(y_test.values, model.predict(X_test)),
            best_iteration=model.best_iteration_,
            val_month=str(split_info.val_start.strftime("%Y-%m")),
            test_month=str(split_info.test_start.strftime("%Y-%m")),
        )

    # ── Tum split'lerde egitim ──

    def _train_all_splits(
        self, df: pd.DataFrame, params: dict[str, Any],
    ) -> TrainingResult:
        X_sample, _ = self._split_xy(df.iloc[:1])
        results: list[SplitResult] = []

        for split_info, train_df, val_df, test_df in self._splitter.iter_splits(df):
            result = self._train_split(split_info, train_df, val_df, test_df, params)
            results.append(result)

        val_mapes = [r.val_metrics.mape for r in results]
        test_mapes = [r.test_metrics.mape for r in results]
        best_iters = [r.best_iteration for r in results]

        return TrainingResult(
            split_results=results,
            avg_val_mape=float(np.mean(val_mapes)),
            avg_test_mape=float(np.mean(test_mapes)),
            std_val_mape=float(np.std(val_mapes)),
            avg_best_iteration=int(np.mean(best_iters)),
            feature_names=list(X_sample.columns),
        )

    # ── Optuna objective (DINAMIK) ──

    def _create_objective(self, df: pd.DataFrame) -> Callable[[Trial], float]:
        search_space = self._search_config.search_space
        fixed_params = {
            "task_type": self._cb_config.training.task_type,
            "eval_metric": self._cb_config.training.eval_metric,
            "random_seed": self._cb_config.training.random_seed,
            "has_time": self._cb_config.training.has_time,
            "use_best_model": True,
        }

        def objective(trial: Trial) -> float:
            suggested = suggest_params(trial, search_space)
            params = {**fixed_params, **suggested}
            result = self._train_all_splits(df, params)
            return result.avg_val_mape

        return objective

    # ── Optimize ──

    def optimize(self, df: pd.DataFrame) -> tuple[Study, TrainingResult]: ...

    # ── Final model ──

    def train_final(
        self, df: pd.DataFrame, params: dict[str, Any], n_iterations: int,
    ) -> CatBoostRegressor: ...

    # ── Tum pipeline ──

    def run(self, df: pd.DataFrame) -> PipelineResult: ...
```

---

### Adim 7: CatBoostForecaster Guncelleme (~80 satir)

**Dosya:** `src/energy_forecast/models/catboost.py`

```python
class CatBoostForecaster(BaseForecaster):
    """CatBoost forecaster — save/load/predict icin ince wrapper.

    Egitim CatBoostTrainer tarafindan yapilir.
    Bu sinif egitilmis modeli yukleme ve tahmin icin kullanilir.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._model: CatBoostRegressor | None = None

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> None:
        """Basit egitim (Trainer kullanmadan). Test kolayligi icin."""
        ...

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Model not loaded")
        predictions = self._model.predict(X)
        return pd.DataFrame({"consumption_mwh": predictions}, index=X.index)

    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("No model to save")
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path / "model.cbm"))

    def load(self, path: Path) -> None:
        self._model = CatBoostRegressor()
        self._model.load_model(str(path / "model.cbm"))
```

---

### Adim 8: CLI Entry Point (~120 satir)

**Dosya:** `src/energy_forecast/training/run.py`

```python
"""CLI entry point for model training.

Usage:
    python -m energy_forecast.training.run --model catboost [--n-trials 5] [--data PATH]
"""

def parse_args() -> argparse.Namespace: ...

def load_data(data_path: Path) -> pd.DataFrame: ...

def run_catboost(settings: Settings, data: pd.DataFrame) -> None:
    tracker = ExperimentTracker(
        experiment_name="energy-forecast-catboost",
        tracking_uri=settings.env.mlflow_tracking_uri,
    )
    trainer = CatBoostTrainer(settings, tracker)
    result = trainer.run(data)
    logger.info("Best MAPE: {:.2f}%", result.training_result.avg_val_mape)
    logger.info("Best params: {}", result.best_params)

def main() -> None:
    args = parse_args()
    settings = load_config()
    data = load_data(args.data)
    if args.model == "catboost":
        run_catboost(settings, data)
    ...
```

**CLI argumanlari:**
- `--model`: catboost (zorunlu)
- `--n-trials`: Optuna trial sayisi override (opsiyonel)
- `--data`: Feature-engineered parquet yolu (default: `data/processed/features.parquet`)
- `--no-mlflow`: MLflow devre disi birak

---

### Adim 9: `__init__.py` + Temizlik

**`src/energy_forecast/training/__init__.py`:**

```python
"""Training utilities: cross-validation, metrics, search, experiment tracking."""

from energy_forecast.training.metrics import MetricsResult, compute_all
from energy_forecast.training.search import suggest_params
from energy_forecast.training.splitter import SplitInfo, TimeSeriesSplitter

__all__ = [
    "MetricsResult",
    "SplitInfo",
    "TimeSeriesSplitter",
    "compute_all",
    "suggest_params",
]
```

**Silinecek:** `cross_validation.py`, `tuning.py`

---

### Adim 10: mypy Overrides

**`pyproject.toml`:** Asagidakileri kontrol et, eksikleri ekle:
```toml
[[tool.mypy.overrides]]
module = ["mlflow.*", "catboost.*", "optuna.*"]
ignore_missing_imports = true
```

---

## 4. Test Stratejisi

### 4.1 test_splitter.py (~220 satir)

| # | Test | Aciklama |
|---|------|----------|
| 1 | `test_basic_split_count` | n_splits=12 → 12 split uretilir |
| 2 | `test_split_temporal_order` | train_end < val_start < val_end < test_start < test_end |
| 3 | `test_expanding_window` | Split 0 train < Split 11 train (expanding) |
| 4 | `test_no_overlap` | Train, val, test arasinda kesisim yok |
| 5 | `test_val_is_calendar_month` | val_start ayin 1. gunu, val_end ayin son gunu |
| 6 | `test_test_is_calendar_month` | test_start ayin 1. gunu, test_end ayin son gunu |
| 7 | `test_february_28_days` | Subat 28 gun (non-leap) → val/test dogru |
| 8 | `test_february_29_days` | Subat 29 gun (leap year) → val/test dogru |
| 9 | `test_months_vary_in_length` | Nisan(30), Mayis(31), Subat(28) → hepsi dogru |
| 10 | `test_gap_hours` | gap_hours>0 → train_end + gap < val_start |
| 11 | `test_insufficient_data` | Yetersiz veri icin ValueError |
| 12 | `test_from_config` | CrossValidationConfig'den olusturma |
| 13 | `test_iter_splits_dataframes` | iter_splits DataFrame dilimlerini dogru doner |
| 14 | `test_val_month_label` | SplitInfo.val_start → "2023-07" formati |
| 15 | `test_n_splits_3` | n_splits=3 → 3 split, hepsi tam takvim ayi |

**Fixture:** `sample_df` — 3 yillik saatlik veri (2022-01-01 → 2024-12-31), DatetimeIndex

### 4.2 test_metrics.py (~150 satir)

| # | Test | Aciklama |
|---|------|----------|
| 1 | `test_mape_perfect` | Esit degerler → MAPE = 0 |
| 2 | `test_mape_known` | Bilinen input → bilinen MAPE |
| 3 | `test_mape_zero_actual` | y_true=0 noktalar skip edilir |
| 4 | `test_mae_known` | Bilinen MAE |
| 5 | `test_rmse_known` | Bilinen RMSE |
| 6 | `test_r_squared_perfect` | R2 = 1.0 |
| 7 | `test_smape_symmetric` | SMAPE symmetry |
| 8 | `test_wmape_known` | Bilinen WMAPE |
| 9 | `test_mbe_overpredict` | MBE > 0 |
| 10 | `test_mbe_underpredict` | MBE < 0 |
| 11 | `test_compute_all` | 7 metrik doner |

### 4.3 test_search.py (~80 satir)

| # | Test | Aciklama |
|---|------|----------|
| 1 | `test_suggest_int` | type=int → trial.suggest_int |
| 2 | `test_suggest_float` | type=float → trial.suggest_float |
| 3 | `test_suggest_float_log` | log=true → suggest_float(log=True) |
| 4 | `test_suggest_float_step` | step=0.01 → suggest_float(step=0.01) |
| 5 | `test_suggest_categorical` | type=categorical → suggest_categorical |
| 6 | `test_empty_search_space` | Bos dict → bos params |
| 7 | `test_mixed_types` | int + float + categorical karma |

**Mock:** `optuna.Trial` mock (MagicMock)

### 4.4 test_catboost_trainer.py (~250 satir)

| # | Test | Aciklama |
|---|------|----------|
| 1 | `test_split_xy` | consumption ayrılır, X'te yok |
| 2 | `test_prepare_categoricals` | NaN→missing, str donusumu |
| 3 | `test_prepare_categoricals_missing_cols` | Olmayan kolon atlanir |
| 4 | `test_train_split_smoke` | Tek split (kucuk veri) |
| 5 | `test_train_all_splits` | Tum split'ler, metrikler toplu |
| 6 | `test_optimize_smoke` | n_trials=2 |
| 7 | `test_train_final` | Final model tum veri |
| 8 | `test_run_pipeline` | optimize + final |
| 9 | `test_target_col_not_in_features` | M4 audit |
| 10 | `test_has_time_true` | has_time=True |
| 11 | `test_dynamic_search_space` | YAML'dan farkli params |
| 12 | `test_split_result_month_labels` | val_month, test_month dogru |

**Mock:** CatBoost gercekten egitilir (kucuk sentetik veri ~500 satir),
MLflow `enabled=False`

### 4.5 test_experiment.py (~120 satir)

| # | Test | Aciklama |
|---|------|----------|
| 1 | `test_disabled_noop` | enabled=False → noop |
| 2 | `test_start_run` | Context manager |
| 3 | `test_log_params` | Parametreler loglanir |
| 4 | `test_log_metrics` | Metrikler loglanir |
| 5 | `test_log_model` | Model artifact |
| 6 | `test_log_feature_importance` | Feature importance |

**Mock:** `mlflow` module mock (`unittest.mock.patch`)

### 4.6 Config test eklemeleri (test_config.py'ye ~8 test)

| # | Test | Aciklama |
|---|------|----------|
| 1 | `test_search_param_int_valid` | type=int, low, high → OK |
| 2 | `test_search_param_float_log` | type=float, log=True → OK |
| 3 | `test_search_param_categorical` | type=categorical, choices → OK |
| 4 | `test_search_param_missing_low` | type=int, low=None → Error |
| 5 | `test_search_param_low_gt_high` | low > high → Error |
| 6 | `test_search_param_log_with_step` | log + step → Error |
| 7 | `test_search_param_categorical_no_choices` | choices=None → Error |
| 8 | `test_model_search_config` | n_trials + dict → OK |

---

## 5. Eski Proje Referanslari

| Eski Dosya | Yeni Dosya | Alinacak | Tasinmayacak |
|-----------|-----------|----------|--------------|
| `cross_validation.py` (~180 sat) | `splitter.py` (~130 sat) | Backward-anchored expanding mantigi | sliding, blocked stratejileri |
| `trainer.py` (~300 sat) | `catboost_trainer.py` (~250 sat) | Split egitimi, X/y, Pool | Report generation |
| `training_pipeline.py` (~200 sat) | `catboost_trainer.py` | study.optimize, final model | Hardcoded param generator |
| `metrics.py` (~150 sat) | `metrics.py` (~80 sat) | MAPE/MAE/RMSE/R2 formulleri | adjusted_r2, directional_accuracy |
| `experiment_tracker.py` (~200 sat) | `experiment.py` (~130 sat) | MLflow API pattern | Predictions logging |
| `scripts/train.py` (~150 sat) | `run.py` (~120 sat) | CLI pattern | Report generation |

---

## 6. Implementasyon Sirasi

```
Adim  1: Config (settings.py + hyperparameters.yaml + config testleri)
   ↓
Adim  2: splitter.py + test_splitter.py
   ↓
Adim  3: metrics.py (stub→impl) + test_metrics.py
   ↓
Adim  4: search.py + test_search.py
   ↓
Adim  5: experiment.py + test_experiment.py
   ↓
Adim  6: catboost_trainer.py + test_catboost_trainer.py
   ↓
Adim  7: catboost.py (models/) guncelle
   ↓
Adim  8: run.py (CLI)
   ↓
Adim  9: __init__.py + cross_validation.py sil + tuning.py sil
   ↓
Adim 10: mypy overrides + make lint
   ↓
Son: make test && make lint → temiz
```

---

## 7. Cikis Kriterleri

| Kriter | Dogrulama |
|--------|-----------|
| Tum testler gecer | `make test` → ~1000 test PASS |
| Lint temiz | `make lint` → no errors |
| TSCV 12 split, tam takvim ayi | test_splitter: val/test ay sinirlarinda |
| Subat 28/29 gun dogru | test_splitter: february testleri |
| X/y ayrimi | consumption X'te yok (M4 audit) |
| Kategorik NaN→missing | test_catboost_trainer |
| Dinamik search space | test_search: YAML→suggest otomatik |
| YAML'a param ekle → kod degismez | test_dynamic_search_space |
| Optuna calisir | test_optimize_smoke (n_trials=2) |
| MLflow calisir | test_experiment (mock) |
| has_time=True | Config'de sabit |
| Ay bazli MAPE raporlanir | SplitResult.val_month/test_month |

---

## 8. Commit Plani

`feat/m5-catboost-training` branch'inde:

```
feat(config): add dynamic search space and calendar-month CV config
feat(training): add calendar-month TSCV splitter
feat(training): implement evaluation metrics
feat(training): add dynamic Optuna search space parser
feat(training): add MLflow experiment tracker
feat(training): implement CatBoost trainer with Optuna
feat(models): implement CatBoostForecaster save/load/predict
feat(training): add CLI entry point for model training
refactor(training): remove old stubs, update exports
```

---

## 9. Risk ve Dikkat Noktalari

| Risk | Onlem |
|------|-------|
| Ay siniri hesaplama hatalari | pandas MonthBegin/MonthEnd + kapsamli testler (Sub, Nis, Ara) |
| Subat/artik yil edge case | Ayri testler: 28 ve 29 gun |
| Veri ayin ortasinda basliyorsa | Ilk eksik ayi atla, tam aylardan basla |
| Veri ayin ortasinda bitiyorsa | Son eksik ayi test'e dahil etme |
| CatBoost yavas (buyuk veri) | Test'lerde kucuk sentetik veri (~500 satir) |
| MLflow baglanti | enabled=False ile test |
| Optuna uzun surer | Test'lerde n_trials=2 |
| consumption target leakage | _split_xy her yerde cikarir |
| Dinamik YAML parsing | SearchParamConfig model_validator |
| log + step birlikte | Pydantic validator engeller |
