# TODO — Regional Energy Forecast
> Son guncelleme: 2026-03-08
> Kaynak: Code Review 2026-03-07 (`skills_results/code-review/2026-03-07_17-39/`)
> Genel Skor: **8.5/10** | Leakage: **TEMIZ** | Hedef Post-Fix: **8.8/10**

---

## P0: Kritik Bug Fix (7 adet — hemen duzelt, ~1-2 saat)
> Bunlar data inconsistency, runtime crash veya CI gap'leri. Tek basina blocker.

- [x] **CDD_BASE temperature mismatch** — `weather_repo.py:31` ✅
  - `_CDD_BASE = 22.0` → `24.0` (YAML, settings.py, custom.py ile sync)
  - Test guncellendi: `test_weather_repo.py` (2 assertion)

- [x] **Drift detector division by zero** — `drift_detector.py:248-251` ✅
  - `func.nullif(actual_mwh, 0)` ile actual_mwh=0 durumu handle edildi

- [x] **EPIAS timestamp timezone mismatch** — `epias_client.py:565-580` ✅
  - `UTC` import kaldirildi, `ZoneInfo("Europe/Istanbul")` ile degistirildi
  - 3 saatlik kayma riski ortadan kaldirildi

- [x] **backup_db.py timezone-naive + timeout yok** — `backup_db.py:40,63` ✅
  - `datetime.now(tz=_TZ_ISTANBUL)` + `subprocess.run(timeout=300)`

- [x] **backup_db.py silinen dosya path'i donduruyor** — `backup_db.py:91,98` ✅
  - GDrive upload basariliysa `None` dondur, return type `Path | None`

- [x] **CI coverage threshold 75% vs SPEC target 90%** — `ci.yml:35` ✅
  - `--cov-fail-under=85` (kademe: 85 → 88 → 90)

- [x] **CI ruff check tests/ kapsamiyor** — `ci.yml:18` ✅
  - `ruff check src/ tests/`

---

## P1: Guvenlik + Tutarlilik (en onemli 10, ~1 gun)
> Config drift, guvenlik aciklari, performans sorunlari. Sessiz bug kaynaklari.

- [x] **Pydantic defaults YAML'dan 7 noktada sapmis** — `settings.py` ✅
  - EpiasConfig.variables: 5→3 (FDPP + Bilateral cikarildi)
  - CatBoost categoricals: 36→28 (R2 pruned features sync)
  - CatBoost loss_function: "MAE"→"RMSE", iterations: 2000→5000, early_stopping: 200→100, verbose: 100→500
  - Prophet regressors: -wind_speed_10m -is_peak, +apparent_temperature +is_sunday (R2 sync)
  - TFT time_varying_known: 19→24 (+wth_cdd, wth_hdd, hdd_x_hour, temp_x_hour, cdd_x_hour)
  - TFT time_varying_unknown: 19→16 (moved hdd_x_hour to known, removed 3 extras not in YAML)

- [x] **conftest.py FDPP deprecated degisken iceriyor** — `tests/conftest.py:200` ✅
  - FDPP cikarildi: conftest, test_epias, test_pipeline, test_epias_client, test_schemas, test_pipeline_e2e

- [x] **`/status/active` endpoint auth yok** — `app.py:464-498` ✅
  - Response strip edildi: `{"busy": bool, "queue_size": int}` — job metadata kaldirildi

- [x] **PREDICTION_COL sabiti kullanilmiyor** — `catboost.py:95`, `prophet.py:100` ✅
  - `PREDICTION_COL` import edildi, hardcoded string kaldirildi

- [x] **job_repo.get_stats() full table scan** — `job_repo.py:123-129` ✅
  - SQL `select(status, func.count()).group_by(status)` ile degistirildi

- [x] **run.py bos metrics DB'ye yaziyor** — `run.py:484-486` ✅
  - Runner fonksiyonlari dict(metrics, model_path, best_params) dondurur
  - _complete_model_run metrics/model_path/hyperparams'i DB'ye yazar

- [x] **datetime.now() timezone-naive (5+ yer)** — scripts/ + tests/ ✅
  - `backfill_epias.py`, `fetch_profile_coefficients.py` (2 yer), `test_schemas.py` (4 yer), `test_job_manager.py`

- [x] **GDrive query injection** — `gdrive.py:217` ✅
  - `name.replace("'", "\\'")` ile escape eklendi

- [x] **GDrive retry logic yok** — `gdrive.py` (tum class) ✅
  - GDriveTransientError + @retry(stop=3, wait_exponential(2,30))
  - _find_folder, _create_folder, _upload_file — HttpError 429/500/502/503 retry

- [x] **weather_actuals.py `object` type hints** — `weather_actuals.py:25-26,106-107` ✅
  - `TYPE_CHECKING` import ile `async_sessionmaker | object` ve `Settings | object` tipi

---

## P2: Kod Organizasyonu — Decompose (planlanan, sprint)
> Buyuk dosyalari parcala. Fonksiyonel degisiklik yok, sadece yeniden yapi.

- [x] **settings.py 1508 satir → config/ paket** ✅
  - general.py, features.py, models.py, api.py, _loader.py
  - `__init__.py` re-export tum config class'lari, 37 dosya import guncellendi

- [x] **ensemble_trainer.py 991 satir → 4 dosya** ✅
  - ensemble_trainer.py (~350): orchestrator + dataclasses + delegate methods
  - ensemble_stacking.py (~200): OOF builder, meta-learner, stacking test eval
  - ensemble_weights.py (~250): SLSQP optimizer, split metrics, weighted eval
  - ensemble_report.py (~120): comparison_df, print_summary

- [ ] **job_manager.py process_job_db 382 satir → pipeline steps**
  - _run_prediction_step, _store_predictions_step, _store_weather_step
  - _create_output_step, _send_email_step, _archive_step

- [x] **SplitResult + _optuna_storage DRY refactor** ✅
  - 3 trainer'da neredeyse ayni dataclass → training/results.py (tek SplitResult, best_iteration=0 default)
  - _optuna_storage() → training/utils.py (tek `optuna_storage()` fonksiyonu, 3 trainer delegate)

- [x] **Context feature builder 3x duplicate → tek fonksiyon** ✅
  - `build_context_features()` in training/ensemble_utils.py
  - 3 yerdeki duplicate (ensemble.py, ensemble_stacking.py x2) bu fonksiyona delegate

- [ ] **analytics_repo.py SQL aggregation**
  - 561 satir, her query full table scan → Python aggregation
  - Fix: PG dialect icin SQL aggregation, SQLite icin mevcut Python fallback

---

## P3: Test Iyilestirmeleri

- [ ] **test_schemas.py negatif testler** — ConsumptionSchema, WeatherSchema, EpiasSchema
  - Sadece happy-path var, wrong dtype/NaN threshold/missing column testleri yok

- [ ] **test_audit_repo.py genislet** — `get_recent_by_action` test edilmemis
  - Drift cooldown mekanizmasi bu metoda bagimli

- [ ] **Prophet/ensemble testlerine @pytest.mark.slow** — 3+ test
  - Gercek model instance olusturuluyor ama slow marker yok

- [ ] **Pre-commit mypy stub'lari** — `.pre-commit-config.yaml:22-24`
  - pandas-stubs, types-SQLAlchemy eksik → CI ile farkli sonuc

---

## P4: Kucuk Iyilestirmeler (opsiyonel, toplu fix)

- [ ] cleanup_jobs.py `print()` → `loguru` (2 yer)
- [ ] backup_db.py hardcoded DB credentials → env/config'den oku
- [ ] prepare_dataset.py 7x `except Exception` → spesifik exception
- [ ] openmeteo_client.py broad `except Exception` geocoding
- [ ] Prophet load() metadata iki kez okuyor → tek okuma
- [ ] TFT predict() context_df=None yorumu yaniltici → guncelle
- [ ] Version "0.1.0" ve feature_count 153 hardcoded → config'den oku
- [ ] Email template "24 saatlik" → 48 saat (SPEC uyumu)
- [ ] DriftConfig defaults YAML ile duplicate → tek kaynak
- [ ] Scheduler ilk calisma 1 saat gecikebilir → startup check
- [ ] _debug_utils.py TeeLogger context manager yok
- [ ] promote_model.py relative Path → project-root-relative
- [ ] TFT assert → proper exception (`tft.py:318`)
- [ ] ensemble_trainer verbose=50 hardcoded → config
- [ ] monitoring __init__.py re-export eksik
- [ ] admin_router circular import riski (fragile import chain)

---

## Backlog (Proje Seviyesi)

| # | Gorev | Oncelik | Durum |
|---|-------|---------|-------|
| B1 | Production HPO (3-model, 12-fold, TFT NeuralForecast) | Yuksek | Bekliyor |
| B2 | Ensemble trainer refactor — ayrı egitim + hafif weight opt | Orta | P2 ile birlikte |
| B3 | AWS deploy (Docker + ECS/App Runner) — M11 | Yuksek | Planlanmadi |
| B4 | Frontend web UI (React/Next.js) | Orta | Planlanmadi |

---

## Tamamlanan Calisma Ozeti

### Code Review Fix'leri (2026-02-28, Sprint 1-4)
> 22 fix tamamlandi. Detay: git history, eski todo sprint 1-4.
> Skor: 7.7/10 → 8.5/10 (Sprint 1-2 review'lari ile dogrulandi)

### Database Integration (2026-03-07, Faz 1-4)
> PostgreSQL + Alembic + 6 repository + dual-mode serving + analytics dashboard
> 766 test, %88 coverage. Detay: git history (feat/faz1-4 branch'leri)

### TFT NeuralForecast Migration (2026-03-05)
> pytorch-forecasting → NeuralForecast. GPU util %1-5 → %96-100.

### Code Review 2026-03-07 — Full Audit
> 5 paralel agent, 15,094 satir incelendi. 7 kritik, 33 onemli, 36 kucuk bulgu.
> Leakage: TEMIZ (4 katman savunma dogrulandi)
> Rapor: `skills_results/code-review/2026-03-07_17-39/13_final_scorecard.md`

### P0 + P1 Bug Fixes (2026-03-07)
> 7 P0 kritik + 10 P1 guvenlik/tutarlilik fix tamamlandi.
> Skor: 7.7/10 → 8.5/10

### P2-A: settings.py Decomposition (2026-03-08)
> 1508 satir → 5-modul config/ paket. 37 dosya import guncellendi. 757 test pass.
> Docker + PostgreSQL + API boot dogrulandi.

---

## Referanslar
- Code Review 2026-03-07: `skills_results/code-review/2026-03-07_17-39/`
- Code Review 2026-03-05: `skills_results/code-review/2026-03-05_review/`
- Code Review 2026-02-28: git history (sprint 1-4 fix'leri)
- DB Integration Plans: `skills_results/plan-skills/2026-03-06_22-*`
