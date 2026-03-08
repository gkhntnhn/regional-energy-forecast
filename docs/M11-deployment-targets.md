# M11 Deployment & Data Architecture Targets

> Tarih: 2026-03-08
> Durum: Planlanıyor
> Kapsam: Veritabanı mimarisi, compute platformu, deploy altyapısı

---

## 1. Hedef Mimari (Genel Bakış)

```
                    ┌──────────────────────────────┐
                    │      Google Cloud Run         │
                    │  FastAPI container (min=1)    │
                    │  CatBoost + Prophet + TFT     │
                    └──────────┬───────────────────┘
                               │
               ┌───────────────┼───────────────┐
               │               │               │
      ┌────────▼──────┐ ┌─────▼─────┐ ┌───────▼───────┐
      │   NeonDB       │ │  Google   │ │ Cloud         │
      │  (PostgreSQL)  │ │  Drive    │ │ Scheduler     │
      │  Tum veriler   │ │  Yedek    │ │ Cron jobs     │
      └────────────────┘ └───────────┘ └───────────────┘
```

---

## 2. Veri Mimarisi: Dosyadan PostgreSQL'e Gecis

### 2.1 Mevcut Durum (Degisecek)

| Veri | Mevcut Format | Sorun |
|------|---------------|-------|
| EPIAS market | `data/external/epias/epias_market_{year}.parquet` | Yillik dosya, incremental update yok, container'da kaybolur |
| EPIAS generation | `data/external/epias/epias_generation_{year}.parquet` | Ayni sorun |
| Weather cache | `data/external/weather_cache.sqlite` | Opak (requests-cache binary), audit edilemez |
| Holidays | `data/static/turkish_holidays.parquet` | Dosya bazli, versiyon kontrolu yok |
| Profile coef | `data/external/profile/profile_coef_{year}.parquet` | Kullanilmiyor henuz |

### 2.2 Hedef Durum

Tum external veriler NeonDB (PostgreSQL) icinde tek kaynak (single source of truth):

```sql
-- EPIAS Market: Saatlik, incremental upsert (tum 5 degisken — hangisinin feature olacagi YAML'dan belirlenir)
CREATE TABLE epias_market (
    datetime        TIMESTAMPTZ PRIMARY KEY,
    fdpp            DOUBLE PRECISION,  -- KGUP toplam (region=TR1, quarterly fetch)
    rtc             DOUBLE PRECISION,
    dam_purchase    DOUBLE PRECISION,
    bilateral       DOUBLE PRECISION,
    load_forecast   DOUBLE PRECISION,
    fetched_at      TIMESTAMPTZ DEFAULT now()
);

-- EPIAS Generation: Saatlik (tum yakit tipleri — hangisinin feature olacagi YAML'dan belirlenir)
CREATE TABLE epias_generation (
    datetime            TIMESTAMPTZ PRIMARY KEY,
    gen_asphaltite_coal DOUBLE PRECISION,
    gen_biomass         DOUBLE PRECISION,
    gen_black_coal      DOUBLE PRECISION,
    gen_dammed_hydro    DOUBLE PRECISION,
    gen_fueloil         DOUBLE PRECISION,
    gen_geothermal      DOUBLE PRECISION,
    gen_import_coal     DOUBLE PRECISION,
    gen_import_export   DOUBLE PRECISION,
    gen_lignite         DOUBLE PRECISION,
    gen_lng             DOUBLE PRECISION,
    gen_naphta          DOUBLE PRECISION,
    gen_natural_gas     DOUBLE PRECISION,
    gen_river           DOUBLE PRECISION,
    gen_sun             DOUBLE PRECISION,
    gen_total           DOUBLE PRECISION,
    gen_wasteheat       DOUBLE PRECISION,
    gen_wind            DOUBLE PRECISION,
    fetched_at          TIMESTAMPTZ DEFAULT now()
);

-- Weather Cache: Sehir x saat x kaynak
CREATE TABLE weather_cache (
    datetime              TIMESTAMPTZ NOT NULL,
    city                  TEXT NOT NULL,
    source                TEXT NOT NULL,  -- 'historical' | 'forecast'
    temperature_2m        DOUBLE PRECISION,
    relative_humidity_2m  DOUBLE PRECISION,
    apparent_temperature  DOUBLE PRECISION,
    dew_point_2m          DOUBLE PRECISION,
    precipitation         DOUBLE PRECISION,
    snow_depth            DOUBLE PRECISION,
    weather_code          SMALLINT,
    surface_pressure      DOUBLE PRECISION,
    wind_speed_10m        DOUBLE PRECISION,
    wind_direction_10m    DOUBLE PRECISION,
    shortwave_radiation   DOUBLE PRECISION,
    fetched_at            TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (datetime, city, source)
);

-- Holidays: Statik referans tablo
CREATE TABLE turkish_holidays (
    date              DATE PRIMARY KEY,
    holiday_name      TEXT,
    tatil_tipi        SMALLINT,
    bayram_gun_no     SMALLINT,
    is_ramadan        BOOLEAN,
    bayrama_kalan_gun SMALLINT
);

-- Profile Coefficients: Yillik referans
CREATE TABLE profile_coefficients (
    datetime    TIMESTAMPTZ NOT NULL,
    year        SMALLINT NOT NULL,
    coefficient DOUBLE PRECISION,
    PRIMARY KEY (datetime, year)
);
```

### 2.3 Kazanimlar

| Mevcut | Hedef |
|--------|-------|
| Parquet dosya kaybedilebilir | ACID garanti, NeonDB yonetimli backup |
| Yillik toplu cekme (29 Aralik problemi) | Gunluk incremental upsert |
| Weather cache opak, audit edilemez | Sorgulanabilir, sehir bazli denetim |
| Container restart = veri kaybi | DB disarida, container stateless |
| Race condition (dosya kilidi yok) | Transaction isolation |
| Checksum yok, sessiz bozulma riski | PostgreSQL veri butunlugu |

### 2.4 Parquet Dosyalarinin Yeni Rolu

Parquet dosyalari silinmiyor, rolu degisiyor:

```
Eski: Parquet = production veri kaynagi (pipeline dogrudan okur)
Yeni: Parquet = seed/backup/import formati

  API ──────→ PostgreSQL (production source of truth)
  Parquet ──→ PostgreSQL (seed/import, lokal dev)
  PostgreSQL → Parquet    (backup/export, opsiyonel)
```

---

## 3. Compute: Google Cloud Run

### 3.1 Neden Cloud Run

| Kriter | Karar Nedeni |
|--------|-------------|
| **Maliyet** | min=1 instance, idle CPU ucretsiz. ~$3-5/ay (dusuk trafik) |
| **Cold start** | min_instances=1 → modeller bellekte, cold start yok |
| **ML model serving** | 2 GB RAM yeterli, 60 dk timeout yeterli |
| **Sunucu yonetimi** | Sifir — container deploy et, gerisini Google yonetir |
| **GDrive** | Ayni ekosistem, native entegrasyon |
| **Cron** | Cloud Scheduler ile gunluk EPIAS/weather sync |
| **Docker** | Mevcut Dockerfile dogrudan kullanilir |

### 3.2 Elenenen Alternatifler

| Platform | Elenme Nedeni |
|----------|---------------|
| **AWS Lambda** | Cold start oldurucu (3 ML model yukleme 30-60 sn), 15 dk timeout |
| **AWS EC2** | 7/24 acik = dusuk trafik icin gereksiz maliyet (~$15-30/ay) |
| **AWS ECS Fargate** | ALB sabit maliyeti (~$16/ay), bu olcek icin over-engineered |

### 3.3 Cloud Run Konfigurasyonu

```yaml
# Hedef Cloud Run ayarlari
service: energy-forecast-api
region: europe-west1          # NeonDB ile ayni bolge
min_instances: 1              # Cold start yok
max_instances: 3              # Burst trafik icin
memory: 2Gi                   # 3 model + FastAPI
cpu: 2                        # Prediction CPU-bound
timeout: 300s                 # 5 dk (tahmin icin yeterli)
concurrency: 1                # ML model thread-safe degilse
port: 8000                    # Mevcut FastAPI portu
```

### 3.4 Cloud Scheduler (Cron Jobs)

| Cron | Saat | Endpoint | Aciklama |
|------|------|----------|----------|
| EPIAS market sync | 06:00 UTC+3 | `POST /internal/sync-epias` | Dunku market verisi |
| EPIAS generation sync | 06:15 UTC+3 | `POST /internal/sync-epias-gen` | Dunku uretim verisi |
| Weather sync | 06:30 UTC+3 | `POST /internal/sync-weather` | Guncellenmis forecast |
| DB backup (GDrive) | 02:00 UTC+3 | `POST /internal/backup` | Haftalik yedek |

---

## 4. NeonDB (PostgreSQL)

### 4.1 Neden NeonDB

| Ozellik | Deger |
|---------|-------|
| **Serverless** | Otomatik scale-up/down, idle'da suspend |
| **Free tier** | 0.5 GB storage, 1 compute endpoint |
| **Branching** | Dev/staging icin DB branch (git-like) |
| **Connection pooler** | Dahili PgBouncer (serverless baglanti yonetimi) |
| **Backup** | Otomatik, point-in-time recovery |

### 4.2 Baglanti Stratejisi

```
Cloud Run  →  NeonDB pooler endpoint (connection pooling)
              postgresql+asyncpg://user:pass@ep-xxx.region.neon.tech/dbname?sslmode=require

Lokal dev  →  NeonDB direkt veya lokal Docker PostgreSQL
              make seed-db ile veri yukle
```

### 4.3 Tablolar (Toplam: 10)

**Mevcut (5 tablo):**
- `jobs` — Tahmin job'lari
- `predictions` — Saatlik tahmin sonuclari
- `weather_snapshots` — Tahmin anindaki hava durumu kaydi
- `audit_logs` — API islem gecmisi
- `model_runs` — Egitim gecmisi

**Yeni (5 tablo):**
- `epias_market` — EPIAS piyasa verileri (saatlik)
- `epias_generation` — EPIAS uretim verileri (saatlik)
- `weather_cache` — OpenMeteo cache (sehir x saat)
- `turkish_holidays` — Tatil referans tablosu
- `profile_coefficients` — Profil katsayilari

---

## 5. Google Drive Yedekleme

### 5.1 Yedek Icerigi

| Icerik | Frekans | Format | GDrive Klasor |
|--------|---------|--------|---------------|
| Tahmin sonuclari | Her job | Excel (.xlsx) | `forecasts/YYYY/MM/DD/` |
| DB backup | Haftalik | SQL dump (.sql.gz) | `backups/db/` |
| Model artifact | Promote sonrasi | .cbm, .pkl, .ckpt | `backups/models/` |

### 5.2 Klasor Yapisi

```
Google Drive (Service Account)
└── energy-forecast/
    ├── forecasts/
    │   └── 2026/03/08/
    │       └── 10-00_abc123/
    │           ├── forecast.xlsx
    │           └── metadata.json
    ├── backups/
    │   ├── db/
    │   │   └── 2026-03-08_weekly.sql.gz
    │   └── models/
    │       ├── catboost_v1.cbm
    │       ├── prophet_v1.pkl
    │       └── tft_v1/
    └── reports/
        └── weekly_accuracy.xlsx
```

---

## 6. Lokal Gelistirme Deneyimi

### 6.1 Secenekler

| Yontem | Komut | Veri Kaynagi |
|--------|-------|--------------|
| **Lokal PG + seed** | `docker compose up -d && make seed-db` | Parquet → lokal DB |
| **NeonDB branch** | `neonctl branches create --name dev` | Prod verisinin kopyasi |
| **Sadece test** | `make test` | SQLite (mevcut, degismez) |

### 6.2 Seed Mekanizmasi

```bash
# Ilk kurulum (bir kere, ~30 saniye)
docker compose up -d postgres
alembic upgrade head
make seed-db-full              # Mevcut parquet'lar → lokal DB

# Gunluk gelistirme
docker compose up -d postgres  # Volume'da veri kalici
make prepare-data              # Pipeline artik DB'den okur
make train-catboost

# Sifirdan baslama (nadir)
docker compose down -v         # Volume sil
docker compose up -d postgres
alembic upgrade head
make seed-db-full              # 30 saniye, API bekleme yok
```

### 6.3 Docker Compose (Lokal)

```yaml
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: energy_forecast
      POSTGRES_USER: forecast_user
      POSTGRES_PASSWORD: dev_password
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data  # Kalici volume

volumes:
  pgdata:  # docker compose down ile SILINMEZ
           # sadece docker compose down -v ile silinir
```

---

## 7. Uygulama Sirasi

Onerilen uygulama fazlari:

### Faz 1: Veri Mimarisi (DB tablolari)
- [ ] Alembic migration: 5 yeni tablo (epias_market, epias_generation, weather_cache, turkish_holidays, profile_coefficients)
- [ ] Repository siniflar: EpiasRepository, WeatherCacheRepository, HolidayRepository, ProfileRepository
- [ ] seed_db.py script: Parquet → PostgreSQL import
- [ ] Docker Compose volume duzeni

### Faz 2: Client Refactor
- [ ] epias_client.py: save_cache/load_cache → DB upsert/select
- [ ] openmeteo_client.py: requests-cache → weather_cache tablosu
- [ ] calendar.py: Parquet okuma → DB okuma
- [ ] Pipeline refactor: pd.read_parquet() → SELECT ... INTO DataFrame

### Faz 3: Cloud Run Deploy
- [ ] Dockerfile optimize (multi-stage, model bake)
- [ ] Cloud Run service olustur (min=1, 2GB RAM)
- [ ] NeonDB baglanti (connection pooler)
- [ ] Environment variables (Secret Manager)
- [ ] Health check endpoint dogrulama

### Faz 4: Cron & Yedekleme
- [ ] Cloud Scheduler: Gunluk EPIAS/weather sync
- [ ] Internal sync endpoint'leri (auth korumasinda)
- [ ] GDrive yedekleme entegrasyonu
- [ ] Haftalik DB backup cron

### Faz 5: CI/CD
- [ ] GitHub Actions: push → build → Cloud Run deploy
- [ ] CI'da PostgreSQL service (SQLite dual-mode kaldirilabilir)
- [ ] Staging environment (NeonDB branch)

---

## 8. Maliyet Tahmini (Aylik)

| Servis | Free Tier | Tahmini Maliyet |
|--------|-----------|-----------------|
| Cloud Run (min=1, 2GB RAM) | 2M request | ~$3-5 |
| NeonDB | 0.5 GB, 1 endpoint | $0 (free) veya ~$19 (Pro) |
| Cloud Scheduler | 3 job | $0 (free) |
| Google Drive | 15 GB | $0 (free) |
| **Toplam** | | **$3-24/ay** |

---

## 9. Referanslar

| Dosya | Icerik |
|-------|--------|
| `CLAUDE.md` | Proje kurallari ve mimari |
| `SPEC.md` | Proje spesifikasyonu |
| `docs/decisions/ADR-001-data-leakage-audit.md` | Leakage audit karari |
| `docs/data-flow.md` | Veri akis haritasi |
