# RunPod GPU ile Production HPO Egitim Rehberi

RunPod bulut GPU uzerinde production-seviye hiperparametre optimizasyonu.

## Neden RunPod?

- TFT (PyTorch) icin GPU gerekli — H100 Tensor Core ile en hizli DL egitimi
- CatBoost `task_type: "GPU"` ile 2-5x hizlanma
- 50 CatBoost + 30 Prophet + 20 TFT trial, 12-fold TSCV local'de 10+ saat surer

## GPU: H100 PCIe ($1.99/saat)

| Ozellik | Deger |
|---------|-------|
| GPU | NVIDIA H100 PCIe, 80GB VRAM |
| RAM | 188 GB |
| vCPU | 16 |
| Tahmini sure | 2-3.5 saat |
| Tahmini maliyet | $4-7 |

## Hizli Bakis

```
LOCAL (senin PC'n)                     POD (RunPod)
------------------                     ------------
1. pack_data.sh       --SCP-->   2. setup_pod.sh
                                 3. apply_prod_config.sh
                                 4. run_training.sh
                                 5. pack_results.sh
6. download_results.sh  <--SCP--
7. Local'de dogrula
8. POD'U KAPAT!
```

## Adim Adim Rehber

### 0. On Gereksinimler (tek seferlik)

```bash
# SSH key olustur (varsa atla)
ssh-keygen -t ed25519 -C "runpod"
cat ~/.ssh/id_ed25519.pub
# --> Kopyala ve RunPod'a yapistir:
# https://www.runpod.io/console/user/settings --> SSH Keys
```

### 1. Veriyi Paketle (local)

```bash
cd C:\Users\pc\Desktop\Python\Projects\regional-energy-forecast

# Weather cache'i yenile (guncel hava tahmini icin)
rm -f data/external/weather_cache.sqlite
uv run python scripts/prepare_dataset.py --skip-epias -v
# Bu islem OpenMeteo'dan guncel hava verisi ceker (~2-5 dk)

# Data dosyalarini paketle (~16MB arsiv)
bash scripts/runpod/pack_data.sh
```

> **Neden prepare_dataset?** Pod'a gondermeden once weather cache'in guncel olmasi lazim.
> OpenMeteo forecast verisi zamanla degisir — ne kadar tazeyse tahmin o kadar iyi.

### 2. RunPod Pod Olustur

1. https://www.runpod.io/console/pods --> **Deploy**
2. GPU sec: **H100 PCIe** (1x)
3. Template: **RunPod PyTorch 2.x** (CUDA + Python hazir)
4. Container Disk: **50 GB** (kod + ortam + model ciktilari)
5. Volume Disk: **20 GB** (opsiyonel, pod silinse bile kalir)
6. **Customize Deployment** --> Expose TCP Port: **22** (SSH icin)
7. **Deploy** tikla

Pod baslayana kadar bekle (20sn - 5dk). Status "Running" olunca **Connect** --> SSH komutunu kopyala.

### 3. Veriyi Pod'a Transfer Et

```bash
# <POD_IP> ve <PORT> degerleri RunPod dashboard'dan gelir
scp -P <PORT> -i ~/.ssh/id_ed25519 runpod_data.tar.gz root@<POD_IP>:/workspace/

# Opsiyonel: EPIAS credential'lar gerekiyorsa .env gonder
scp -P <PORT> -i ~/.ssh/id_ed25519 .env root@<POD_IP>:/workspace/.env
```

### 4. Pod'a Baglan ve Ortami Kur

```bash
ssh root@<POD_IP> -p <PORT> -i ~/.ssh/id_ed25519

# Pod icinde:
cd /workspace
bash regional-energy-forecast/scripts/runpod/setup_pod.sh https://github.com/KULLANICI/regional-energy-forecast.git
```

> **setup_pod.sh ne yapar?**
> 1. Repoyu clone eder
> 2. Data arsivini acar
> 3. .env dosyasini kopyalar
> 4. `uv sync` ile dependency'leri kurar
> 5. CUDA ve CatBoost GPU'yu dogrular
> 6. `prepare_dataset.py` ile feature'lari olusturur

Eger setup_pod.sh henuz mevcut degilse (ilk clone oncesi), manuel yap:
```bash
git clone <REPO_URL>
cd regional-energy-forecast
tar -xzf /workspace/runpod_data.tar.gz
pip install uv && uv sync --all-extras
uv run python scripts/prepare_dataset.py --skip-epias -v
```

### 5. Production Config Uygula

```bash
cd /workspace/regional-energy-forecast
bash scripts/runpod/apply_prod_config.sh
```

Uygulanan degisiklikler:

| Dosya | Degisiklik |
|-------|------------|
| hyperparameters.yaml | CatBoost 50 trial, Prophet 30 trial, TFT 20 trial |
| hyperparameters.yaml | CatBoost iterations 500-3000, 3 loss fonksiyonu (RMSE/MAE/MAPE) |
| hyperparameters.yaml | TFT: hidden 32-128, attention [1,2,4], lstm [1,2], batch [64,128,256] |
| hyperparameters.yaml | Cross-validation: 12 split (dev'deki 2'den) |
| catboost.yaml | task_type: GPU, iterations: 3000, early_stopping: 50 |
| tft.yaml | max_epochs: 50, patience: 5, batch: 128, workers: 4, fast_epochs: 10 |

> **Not:** Script mevcut dev config'leri `.dev_backup/` klasorune yedekler.
> Geri almak icin: `bash scripts/runpod/restore_dev_config.sh`

### 6. Egitimi Baslat

```bash
bash scripts/runpod/run_training.sh
```

Bu komut **tmux** oturumunda 4 egitimi sirayla baslatir (SSH kopsa bile devam eder):

| # | Model | Cihaz | Tahmini Sure |
|---|-------|-------|-------------|
| 1 | CatBoost | GPU | ~30-60 dk |
| 2 | Prophet | CPU | ~45-90 dk |
| 3 | TFT | GPU | ~60-120 dk |
| 4 | Ensemble | CPU | ~10-15 dk |

```bash
# Canli izle
tmux attach -t hpo-training

# Geri cik (egitim arka planda devam eder)
# Tusla: Ctrl+B, sonra D

# Hala calisiyor mu kontrol et
tmux ls
```

> **Onemli:** SSH baglantisi koparsa egitim durmaz. Yeniden baglanip
> `tmux attach -t hpo-training` ile izlemeye devam edebilirsin.

### 7. Sonuclari Paketle (pod'da)

Egitim tamamlandiktan sonra:
```bash
cd /workspace/regional-energy-forecast
bash scripts/runpod/pack_results.sh
```

Paketlenen dosyalar:
- `trained_models.tar.gz` — Egitilmis modeller (models/ + final_models/)
- `training_logs.tar.gz` — Egitim loglari
- `prod_configs.tar.gz` — Production config dosyalari
- `optuna_studies.tar.gz` — Optuna SQLite veritabanlari (varsa)

### 8. Sonuclari Indir (local)

```bash
cd C:\Users\pc\Desktop\Python\Projects\regional-energy-forecast
bash scripts/runpod/download_results.sh <POD_IP> <PORT>
```

Script indirilen arsivleri projeye cikarip cikarmamak istediginizi sorar.

### 9. POD'U KAPAT!

RunPod dashboard --> pod'u **Stop** veya **Delete** et.

| Islem | Ne Olur |
|-------|---------|
| **Stop** | Pod durur, volume kalir, disk ucreti devam eder |
| **Delete** | Her sey silinir, ucret durur |

> **UYARI:** Bos duran pod bile $1.99/saat ucret keser. Egitim bitince HEMEN kapat!

### 10. Local'de Dogrula

```bash
# Testleri calistir
make test

# API'yi baslat ve tahmin dene
make serve
# Baska terminalde:
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer <API_KEY>" \
  -F "file=@data/raw/Consumption_Input_Format.xlsx"
```

## Script Referansi

| Script | Nerede | Ne Yapar |
|--------|--------|----------|
| `pack_data.sh` | Local | Data dosyalarini dogrular ve paketler |
| `setup_pod.sh` | Pod | Repo clone, data acma, dependency kurulum, CUDA dogrulama |
| `apply_prod_config.sh` | Pod | Config'leri production HPO degerlerine cevirir |
| `restore_dev_config.sh` | Her ikisi | Config'leri dev degerlerine geri alir |
| `run_training.sh` | Pod | 4 model egitimini tmux'ta sirayla baslatir |
| `pack_results.sh` | Pod | Egitilmis modelleri ve loglari paketler |
| `download_results.sh` | Local | SCP ile indirir ve projeye cikarir |

## Transfer Edilen Dosyalar

### Local --> Pod (~16MB sikistirilmis)

| Dosya | Boyut | Zorunlu |
|-------|-------|---------|
| data/raw/Consumption_Input_Format.xlsx | 1.6 MB | Evet |
| data/external/epias/*.parquet | 6.6 MB | Evet |
| data/external/profile/*.parquet | 2.5 MB | Evet |
| data/external/weather_cache.sqlite | ~17 MB | Evet |
| data/static/turkish_holidays.parquet | 15 KB | Evet |

### Pod --> Local (~50-200MB tahmini)

| Dosya | Aciklama |
|-------|----------|
| models/ | Zaman damgali egitim ciktilari |
| final_models/ | En iyi modeller (serving icin) |
| *_training.log | Egitim loglari |
| configs/models/*.yaml | Production HPO degerleri |

## Sorun Giderme

**CatBoost GPU calismiyorsa:**
Bazi RunPod template'leri CatBoost GPU desteklemiyor.
`catboost.yaml` --> `task_type: "CPU"` yap ve devam et. Daha yavas ama calisir.

**SSH baglantisi koptu:**
Egitim tmux'ta devam ediyor. Yeniden baglan: `tmux attach -t hpo-training`

**Disk doldu:**
RunPod dashboard'da container disk boyutunu artir (pod restart gerektirir).

**CUDA bulunamadi:**
GPU template (PyTorch 2.x) sectiginizden emin olun, CPU-only degil.

**Weather API timeout:**
Weather cache local'den transfer ediliyor. `prepare_dataset.py` taze veri
isterse OpenMeteo ucretsiz ve credential gerektirmez.

**uv bulunamadi:**
`pip install uv` calistir, sonra `uv sync --all-extras` tekrar dene.
