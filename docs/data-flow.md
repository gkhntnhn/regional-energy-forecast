# Data Flow — Musteri Hareket Dongusu

> Son guncelleme: 2026-03-07 | Faz 1 + Faz 2 tamamlandi
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
| Tahmin degerleri | PostgreSQL `predictions` | 48 satir: job_id, forecast_dt, consumption_mwh, period, model_source |
| Hava durumu tahmini | PostgreSQL `weather_snapshots` | ~48 satir: is_actual=false, temperature, humidity, wind, hdd, cdd |
| EPIAS snapshot | PostgreSQL `jobs.metadata` (JSONB) | data_range, last_values, row_count, nan_summary |

> `predictions.period`: ilk 24 saat = "intraday" (GIP), son 24 saat = "day_ahead" (GOP)
>
> `weather_snapshots`: Tahmin anindaki hava durumu tahmini saklanir.
> Gercek hava verisi geldiginde (Adim 6) forecast vs actual karsilastirma yapilabilir.

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

---

## Ozet: Ne Nerede Saklanir?

| Veri | PostgreSQL | Dosya Sistemi | Google Drive |
|------|:----------:|:-------------:|:------------:|
| Job kaydi (durum, email, tarih) | `jobs` | — | — |
| Tahmin degerleri (48 saatlik MWh) | `predictions` | — | — |
| Tahmin dogruluk (MAPE) | `predictions.error_pct` | — | — |
| Tahmin Excel ciktisi | — | `data/outputs/` | `{ay}/{job_id}/` |
| Upload edilen Excel | — | `data/uploads/` | — |
| Feature dataset (historical) | `jobs.historical_path` | `data/archive/` (gecici) | `{ay}/{job_id}/` |
| Feature dataset (forecast) | `jobs.forecast_path` | `data/archive/` (gecici) | `{ay}/{job_id}/` |
| Job metadata JSON | — | `data/archive/` (gecici) | `{ay}/{job_id}/` |
| Hava durumu tahmini (job bazli) | `weather_snapshots` | — | — |
| Gercek hava durumu (gunluk) | `weather_snapshots` | — | — |
| EPIAS piyasa snapshot | `jobs.metadata` (JSONB) | — | — |
| Audit log (kim ne yapti) | `audit_logs` | — | — |
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
  ├─ Tahminleri kaydet ─────────► [PostgreSQL] predictions (YENI)
  ├─ Hava verisini kaydet ──────► [PostgreSQL] weather_snapshots
  ├─ EPIAS meta kaydet ─────────► [PostgreSQL] jobs.metadata
  ├─ Excel ciktisi olustur ─────► [Dosya Sistemi] data/outputs/
  ├─ Email gonder ──────────────► [Gmail SMTP]
  └─ Audit log ─────────────────► [PostgreSQL] audit_logs

Scheduler (04:00)
  └─ Gercek hava cek ──────────► [PostgreSQL] weather_snapshots (is_actual=true)

Manuel (make db-backup)
  └─ pg_dump + upload ─────────► [Google Drive] .sql.gz
```
