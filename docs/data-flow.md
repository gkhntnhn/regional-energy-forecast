# Data Flow — Musteri Hareket Dongusu

> Son guncelleme: 2026-03-07 | Faz 1 + Faz 2 + Faz 3 + Faz 4 tamamlandi
>
> Bu dokuman musterinin web UI'ya girip Excel yuklemesinden itibaren
> tum verinin nerede saklandigini ve nasil aktigi gosterir.
> Her yeni faz/ozellik eklendiginde guncellenmelidir.

---

## 1. Excel Upload (POST /predict)

Musteri web UI'dan Excel yukler ve email girer.

| Veri | Nerede | Detay |
|------|--------|-------|
| Ham Excel dosyasi | Dosya sistemi | `data/uploads/{job_id}_{file_stem}.xlsx` |
| Job kaydi | PostgreSQL `jobs` | id, email, excel_path, file_stem, status="pending" |
| Audit log | PostgreSQL `audit_logs` | action="predict_request", user_email, ip_address, {job_id, file_name} |

Frontend'e `job_id` doner, polling baslar (`GET /status/{job_id}`).

---

## 2. Prediction Matching (Retroaktif Dogrulama)

Yeni Excel'deki gercek tuketim (T-1 23:00'a kadar) ile
**onceki** job'larin tahminleri karsilastirilir.

| Veri | Nerede | Detay |
|------|--------|-------|
| Gerceklesen tuketim | PostgreSQL `predictions` (ONCEKI job'lar guncellenir) | `actual_mwh` = Excel'den eslesen saat |
| Hata yuzdesi | PostgreSQL `predictions` | `error_pct` = \|gercek - tahmin\| / gercek * 100 |
| Eslestirme zamani | PostgreSQL `predictions` | `matched_at` = islem ani |

> Her yeni upload, eski tahminlerin dogrulugunu otomatik olcer.
> Musteri bunun farkinda degildir — arka planda sessizce calisir.

---

## 3. Tahmin Pipeline'i

`prediction_service.run_prediction()` calisir:

```
Excel parse -> EPIAS cek -> OpenMeteo cek -> Feature pipeline ->
CatBoost + Prophet + TFT -> Ensemble -> 48 saatlik tahmin
```

| Veri | Nerede | Detay |
|------|--------|-------|
| Feature-engineered DataFrame | RAM (gecici) | ~153 feature, 48 satir forecast |
| Weather DataFrame | RAM (gecici) | `predictions.attrs["weather_data"]` |
| EPIAS metadata | RAM (gecici) | `predictions.attrs["epias_snapshot"]` |

Bu adimda DB'ye yazilmaz, RAM'de tutulur. Sonraki adimda persist edilir.

---

## 4. Tahminler ve Metadata DB'ye Yazilir

| Veri | Nerede | Detay |
|------|--------|-------|
| Ensemble tahminleri | PostgreSQL `predictions` | 24 satir (T+1): model_source="ensemble" |
| Per-model tahminleri | PostgreSQL `predictions` | 24 satir x N model: model_source="catboost"/"prophet"/"tft" |
| Hava durumu tahmini | PostgreSQL `weather_snapshots` | ~48 satir: is_actual=false, temperature, humidity, wind, hdd, cdd |
| EPIAS snapshot | PostgreSQL `jobs.metadata_` (JSONB) | data_range, last_values, row_count, nan_summary |
| Feature importance | PostgreSQL `jobs.metadata_` (JSONB) | `feature_importance_top15`: top-15 CatBoost feature + importance |

> `predictions.period`: "day_ahead" (GOP, T+1). Per-model tahminler ayni forecast_dt'ler
> icin ayri satirlarda saklanir — bu sayede model karsilastirma analizi yapilabilir.
>
> `weather_snapshots`: Tahmin anindaki hava durumu tahmini saklanir.
> Gercek hava verisi geldiginde (Adim 6) forecast vs actual karsilastirma yapilabilir.
>
> `feature_importance_top15`: CatBoost `get_feature_importance()` sonucu,
> admin dashboard'da feature trend analizi icin kullanilir.

---

## 5. Cikti Dosyasi ve Email

| Veri | Nerede | Detay |
|------|--------|-------|
| Tahmin Excel ciktisi | Dosya sistemi | `data/outputs/{file_stem}_forecast.xlsx` |
| Email eki | Gmail SMTP | Ayni Excel musteri email'ine gonderilir |
| Job durumu | PostgreSQL `jobs` | status="completed", result_path, email_status="sent" |
| Audit log | PostgreSQL `audit_logs` | action="job_complete" (veya "job_failed" + error) |

> Hata durumunda: `email_status="failed"`, `email_error` alani doldurulur,
> `audit_logs`'a action="job_failed" yazilir.

---

## 5b. Artifact Arsivleme ve GDrive Upload

Email sonrasi, feature dataset'leri ve cikti dosyasi arsivlenir.

| Veri | Nerede | Detay |
|------|--------|-------|
| Historical features | Dosya sistemi (gecici) | `data/archive/jobs/{job_id}/features_historical.parquet` |
| Forecast features | Dosya sistemi (gecici) | `data/archive/jobs/{job_id}/features_forecast.parquet` |
| Metadata JSON | Dosya sistemi (gecici) | `data/archive/jobs/{job_id}/metadata.json` |
| Tum artifact'lar | Google Drive | `forecasts/YYYY/MM/DD/HH-MM_job_id/` altinda yuklenir |
| Artifact path'leri | PostgreSQL `jobs` | historical_path, forecast_path, archive_path |

> GDrive klasor yapisi:
> ```
> GDrive Root/
> +-- backups/
> |   +-- 2026/
> |       +-- 03/
> |           +-- 07/
> |               +-- 14-49/
> |               |   +-- energy_forecast_2026-03-07_14-49.sql.gz
> |               +-- 16-30/
> |                   +-- energy_forecast_2026-03-07_16-30.sql.gz
> +-- forecasts/
>     +-- 2026/
>         +-- 03/
>             +-- 07/
>                 +-- 14-34_5ec264086885/
>                 |   +-- features_historical.parquet
>                 |   +-- features_forecast.parquet
>                 |   +-- metadata.json
>                 |   +-- test_forecast.xlsx
>                 +-- 14-45_4aa14df09e7c/
>                     +-- ...
> ```
>
> GDRIVE_CREDENTIALS_PATH ve GDRIVE_BACKUP_FOLDER_ID .env'de tanimli degilse
> sadece lokal arsiv olusturulur, upload atlanir. Non-fatal: upload hatasi
> pipeline'i kirmaz.

---

## 5c. Drift Detection (Her job sonrasi, otomatik)

Tahmin ve metadata DB'ye yazildiktan sonra, drift detector calisir.

| Kontrol | Detay |
|---------|-------|
| MAPE threshold | Son 7 gundeki ortalama MAPE > esik (varsayilan %5) |
| MAPE trend | Son 4 haftanin MAPE trendi artis gosteriyor mu? |
| Bias shift | Son 7 gundeki ortalama bias (tahmin - gercek) sifirdan sapiyor mu? |

| Veri | Nerede | Detay |
|------|--------|-------|
| Drift alert | PostgreSQL `audit_logs` | action="drift_mape_threshold" / "drift_mape_trend" / "drift_bias_shift" |
| Email bildirim | Gmail SMTP | Drift tespit edilirse admin email'e uyari gonderilir |
| Cooldown | PostgreSQL `audit_logs` | 24 saat icinde ayni tipte tekrar email gonderilmez |

> Drift detection `configs/monitoring.yaml` ile konfigure edilir.
> `email_on_warning: false` ile sadece kritik drift'lerde email gonderilir.
> Minimum orneklem sayisi (min_samples) guard'i ile veri azken false alarm engellenir.

---

## 6. Arka Plan Gorevleri (Musteri tetiklemez)

### 6a. Weather Actuals Scheduler (Otomatik, her gun 04:00)

| Veri | Nerede | Detay |
|------|--------|-------|
| Gercek hava durumu | PostgreSQL `weather_snapshots` | is_actual=true, job_id=NULL, T-2 gunu, 24 satir/gun |

OpenMeteo Archive API'den cekilir. Idempotent — ayni gun iki kez cekmez.
Forecast vs actual karsilastirma icin kullanilir.

### 6b. Database Backup (Manuel: `make db-backup`)

| Veri | Nerede | Detay |
|------|--------|-------|
| DB dump | Google Drive | `energy_forecast_YYYY-MM-DD_HH-MM.sql.gz` |

`pg_dump` -> gzip -> OAuth2 ile GDrive'a upload. Yerel kopya silinir.

### 6c. Data Retention Cleanup (Manuel: `make cleanup-old-data`)

90 gun retention policy. Eski verileri temizler, metadata'yi korur.

| Islem | Detay |
|-------|-------|
| Predictions sil | 90 gunden eski tahmin satirlari silinir |
| Weather snapshots sil | 90 gunden eski forecast snapshot'lari silinir (is_actual=true KORUNUR) |
| Jobs arsivle | 90 gunden eski job'lar status="archived" yapilir (metadata korunur) |

> `make cleanup-dry-run` ile onceden kac satir etkilenecegi gorulur (yazma yapilmaz).
> Weather actuals asla silinmez — uzun vadeli hava durumu dogruluk analizi icin gereklidir.

### 6d. Model Training Tracking (Egitim sirasinda, otomatik)

| Veri | Nerede | Detay |
|------|--------|-------|
| Egitim kaydi | PostgreSQL `model_runs` | model_type, status, val_mape, test_mape, n_trials, n_splits |
| Promoted model | PostgreSQL `model_runs` | is_promoted=true, promoted_at, model_path |

`run.py` egitim basinda `_start_model_run()`, bitiste `_complete_model_run()` cagirir.
`make promote-model MODEL=catboost` ile en iyi model `final_models/`'a kopyalanir ve
DB'de `is_promoted=true` isaretlenir.

---

## 7. Admin Analytics Dashboard (GET /admin/)

Admin dashboard tum analitik verileri **sadece okur** (read-only).
Yeni veri yazmaz, mevcut DB kayitlarini sorgular ve gorsellestirir.

| Sekme | Veri Kaynagi | Endpoint |
|-------|-------------|----------|
| Uretim MAPE | `predictions` (error_pct) | `/admin/analytics/mape/daily`, `/weekly`, `/hourly` |
| Model Karsilastirma | `predictions` (per-model) | `/admin/analytics/models/mape`, `/comparison`, `/hourly` |
| Hava Durumu | `weather_snapshots` (forecast vs actual) | `/admin/analytics/weather/horizon`, `/variables` |
| Feature Onemi | `jobs.metadata_` (JSONB) | `/admin/analytics/features/trend` |
| Egitim Gecmisi | `model_runs` | `/admin/models/runs`, `/promoted` |
| Sistem | DB health + `audit_logs` (drift) | `/admin/system/health`, `/admin/models/drift/status` |
| Job Gecmisi (footer) | `jobs` + `predictions` | `/admin/jobs/history`, `/admin/jobs/{id}/details` |

> Tum admin endpoint'leri Bearer token authentication gerektirir.
> `use_db=False` modunda bos veri doner (hata vermez).
> Dashboard: `src/energy_forecast/serving/static/admin.html` (Chart.js + Bootstrap 5)

---

## Ozet: Ne Nerede Saklanir?

| Veri | PostgreSQL | Dosya Sistemi | Google Drive |
|------|:----------:|:-------------:|:------------:|
| Job kaydi (durum, email, tarih) | `jobs` | — | — |
| Ensemble tahminleri (T+1 MWh) | `predictions` | — | — |
| Per-model tahminleri (catboost/prophet/tft) | `predictions` | — | — |
| Tahmin dogruluk (MAPE) | `predictions.error_pct` | — | — |
| Feature importance (top-15) | `jobs.metadata_` (JSONB) | — | — |
| EPIAS piyasa snapshot | `jobs.metadata_` (JSONB) | — | — |
| Tahmin Excel ciktisi | — | `data/outputs/` | `{ay}/{job_id}/` |
| Upload edilen Excel | — | `data/uploads/` | — |
| Feature dataset (historical) | `jobs.historical_path` | `data/archive/` (gecici) | `{ay}/{job_id}/` |
| Feature dataset (forecast) | `jobs.forecast_path` | `data/archive/` (gecici) | `{ay}/{job_id}/` |
| Job metadata JSON | — | `data/archive/` (gecici) | `{ay}/{job_id}/` |
| Hava durumu tahmini (job bazli) | `weather_snapshots` | — | — |
| Gercek hava durumu (gunluk) | `weather_snapshots` | — | — |
| Drift alert kayitlari | `audit_logs` | — | — |
| Audit log (kim ne yapti) | `audit_logs` | — | — |
| Egitim gecmisi (MAPE, trial sayisi) | `model_runs` | — | — |
| Model dosyalari (.cbm, .pkl, .ckpt) | — | `final_models/` | — |
| DB backup | — | gecici | `backups/` |

---

## PostgreSQL Tablo Yapisi

```
jobs (24 kolon)
  ├── id (PK, 12-char UUID)
  ├── email, status, progress, error
  ├── excel_path, file_stem, result_path
  ├── created_at, completed_at
  ├── metadata (JSONB — epias_snapshot, config vb.)
  ├── config_snapshot, model_versions, epias_snapshot (JSONB)
  ├── excel_hash (SHA256)
  ├── historical_path, forecast_path, archive_path
  ├── email_status, email_sent_at, email_error, email_attempts
  ├── -> predictions (1:N, CASCADE)
  └── -> weather_snapshots (1:N, SET NULL)

predictions (11 kolon)
  ├── id (PK, auto)
  ├── job_id (FK -> jobs)
  ├── forecast_dt, consumption_mwh, period, model_source
  ├── created_at
  └── actual_mwh, error_pct, matched_at (Faz 2 — retroaktif)

weather_snapshots (15 kolon)
  ├── id (PK, auto)
  ├── job_id (FK -> jobs, nullable)
  ├── forecast_dt, fetched_at, is_actual
  ├── temperature_2m, apparent_temperature, relative_humidity_2m
  ├── dew_point_2m, precipitation, snow_depth, surface_pressure
  ├── wind_speed_10m, wind_direction_10m, shortwave_radiation
  ├── weather_code (SmallInteger)
  └── wth_hdd, wth_cdd (turetilmis)

audit_logs (6 kolon)
  ├── id (PK, auto)
  ├── action, user_email, ip_address
  ├── details (JSONB)
  └── created_at

model_runs (13 kolon)
  ├── id (PK, auto)
  ├── model_type (catboost/prophet/tft/ensemble)
  ├── status (running/completed/failed)
  ├── val_mape, test_mape
  ├── n_trials, n_splits, best_params (JSONB)
  ├── model_path, is_promoted, promoted_at
  └── started_at, completed_at
```

---

## Akis Diyagrami (Basitlestirilmis)

```
Musteri
  │
  ├─ Excel upload ──► [Dosya Sistemi] data/uploads/
  │                 ──► [PostgreSQL]   jobs + audit_logs
  │
  ▼
Background Job
  │
  ├─ Onceki tahminleri dogrula ──► [PostgreSQL] predictions (GUNCELLE)
  ├─ Pipeline calistir ──────────► [RAM] gecici
  ├─ Ensemble tahminleri kaydet ─► [PostgreSQL] predictions (model_source=ensemble)
  ├─ Per-model tahminleri kaydet ► [PostgreSQL] predictions (catboost/prophet/tft)
  ├─ Hava verisini kaydet ───────► [PostgreSQL] weather_snapshots
  ├─ EPIAS + Feature imp kaydet ─► [PostgreSQL] jobs.metadata_ (JSONB)
  ├─ Drift detection calistir ──► [PostgreSQL] audit_logs (drift alert)
  │                              ► [Gmail SMTP] drift uyari email (24h cooldown)
  ├─ Excel ciktisi olustur ──────► [Dosya Sistemi] data/outputs/
  ├─ Email gonder ───────────────► [Gmail SMTP]
  ├─ Artifact arsivle ──────────► [Google Drive] forecasts/YYYY/MM/DD/
  └─ Audit log ──────────────────► [PostgreSQL] audit_logs

Scheduler (04:00)
  └─ Gercek hava cek ───────────► [PostgreSQL] weather_snapshots (is_actual=true)

Training Pipeline (run.py)
  └─ Egitim kaydi ──────────────► [PostgreSQL] model_runs (start/complete/fail)

Admin (GET /admin/)
  └─ Analitik sorgular ─────────► [PostgreSQL] OKUMA (predictions, jobs, model_runs, audit_logs)
     Dashboard: MAPE trend, model karsilastirma, hava dogruluk, feature onemi

Manuel (make db-backup)
  └─ pg_dump + upload ──────────► [Google Drive] backups/YYYY/MM/DD/

Manuel (make cleanup-old-data)
  └─ 90 gun retention ──────────► [PostgreSQL] predictions SIL, jobs ARSIVLE
     (weather actuals KORUNUR)
```
