# Smoke Test Dogrulama Plani (Gecici)

> Amac: M11'e gecmeden once her modelin uctan uca calistigini dogrulamak.
> Her adim VS Code Run & Debug ile calistirilir, terminal ciktisi incelenir.
> Tamamlanan adimlar [x] ile isaretlenir.

## On Kosullar
- [x] .env dosyasi mevcut ve EPIAS credentials dogru
- [x] data/raw/Consumption_Input_Format.xlsx mevcut
- [x] data/static/turkish_holidays.parquet mevcut
- [x] data/external/profile/*.parquet mevcut (2020-2026)
- [x] uv sync calistirildi, tum dependency'ler kurulu
- [x] VS Code'da Python interpreter .venv secili

## Adim 0: Dataset Hazirlama
Launch config: "Prepare Dataset"

### Kontrol Listesi:
- [x] Script hatasiz calisti
- [x] Excel yuklendi (satir sayisi logda gorunuyor)
- [x] 48 bos forecast satiri eklendi
- [x] EPIAS verileri cekildi (cache veya API)
- [x] OpenMeteo verileri cekildi
- [x] Feature pipeline calisti
- [x] data/processed/features_historical.parquet olustu
- [x] data/processed/features_forecast.parquet olustu (tam 48 satir)
- [x] Feature sayisi logda gorunuyor (~150+)
- [x] NaN orani kabul edilebilir seviyede

### Beklenen Cikti:
```
features_historical.parquet: ~48000 rows, ~150 columns
features_forecast.parquet: 48 rows, ~150 columns
```

### Sorun varsa:
- EPIAS 401/403 -> .env credentials kontrol et
- OpenMeteo timeout -> --skip-weather ile tekrar dene
- NaN fazla -> /feature-check calistir

---

## Adim 1: CatBoost Smoke Training
Launch config: "Train CatBoost (Smoke)"
Parametreler: iterations=10, n_trials=2, n_splits=2

### Kontrol Listesi:
- [x] Script hatasiz basladi
- [x] Config override uygulandi (smoke_test.yaml)
- [x] features_historical.parquet okundu
- [x] TimeSeriesSplitter fold'lari olustu (2 fold)
- [x] Optuna study basladi (2 trial)
- [x] Trial 1 tamamlandi -- MAPE loglandi
- [x] Trial 2 tamamlandi -- MAPE loglandi
- [x] Best params secildi
- [x] Final model train edildi
- [x] Test seti uzerinde evaluate edildi
- [x] Model kaydedildi (models/catboost/)
- [x] Toplam sure < 2 dakika

### Incelenecek Metrikler:
- [x] Val MAPE: ____% (not al)
- [x] Test MAPE: ____% (not al)
- [x] Train-Val gap: ____% (>%10 ise overfitting suphesi)
- [x] Feature importance top-5 mantikli mi?

### Beklenen Araliklar (smoke, 10 iteration):
- MAPE: %5-%30 arasi normal (az iterasyon, dusuk performans beklenir)
- Sure: 30sn - 2dk

### Sorun varsa:
- "features.parquet not found" -> Adim 0'i tekrar calistir
- Cok yuksek MAPE (>%50) -> Feature'larda sorun, /feature-check calistir
- Memory error -> CatBoost'ta depth dusur

### Post-training (opsiyonel):
- [x] /debug-model catboost calistir
- [x] Notlar: CatBoost Notlar:
- Val MAPE: 5.51%, Test MAPE: 3.92%, Overfit gap: 1.60%
- 38/150 feature aktif (smoke'da normal)
- Ramazan bayramı en büyük zayıflık → tuning'de feature eklenecek
- Sabah 08:00 ramp-up en zor saat

---

## Adim 2: Prophet Smoke Training
Launch config: "Train Prophet (Smoke)"
Parametreler: n_trials=2, n_splits=2

### Kontrol Listesi:
- [x] Script hatasiz basladi
- [x] Config override uygulandi
- [x] Prophet ds+y format donusumu yapildi
- [x] Holiday dataframe yuklendi
- [x] Regressor'lar eklendi
- [x] Optuna study basladi (2 trial)
- [x] Trial 1 tamamlandi -- MAPE loglandi
- [x] Trial 2 tamamlandi -- MAPE loglandi
- [x] Best params secildi
- [x] Final model train edildi
- [x] Test seti uzerinde evaluate edildi
- [x] Model kaydedildi (models/prophet/)
- [x] Toplam sure < 5 dakika

### Incelenecek Metrikler:
- [x] Val MAPE: ____% (not al)
- [x] Test MAPE: ____% (not al)
- [x] Trend mantikli mi? (artis/azalis)
- [x] Seasonality: gunluk ve haftalik pattern var mi?

### Bilinen Sorunlar (Onceki Session'dan):
- Prophet smoke test'te hata vermisti -- hata mesajini not et
- Muhtemel sorunlar: ds/y format, holiday format, regressor NaN

### Sorun varsa:
- "No holidays found" -> data/static/turkish_holidays.parquet kontrol
- NaN error -> Regressor'larda NaN var, feature pipeline kontrol
- Convergence warning -> Kabul edilebilir, ignore et

### Post-training (opsiyonel):
- [x] /debug-model prophet calistir
- [x] Notlar: Prophet Notlar:
- Val MAPE: 7.11%, Test MAPE: 6.17%, Overfit gap: 0.69% (çok iyi)
- R²: 0.8867, Bias: 0.0 MWh
- Regressor fix yapıldı (temperature_2m, relative_humidity_2m)
- BUG: Fourier order config'den okunmuyor, auto-detect kullanılıyor → TODO
- Pazar MAPE %10.1 — weekly fourier order düşük
- Bayram günleri MAPE %23-26 — CatBoost ile aynı pattern
- Holiday katsayıları ~0 — etkisiz
- Model kaydedildi: models/prophet/model.pkl

---

## Adim 3: TFT Smoke Training
Launch config: "Train TFT (Smoke)"
Parametreler: max_epochs=2, n_trials=2, n_splits=2, hidden_size=16

### Kontrol Listesi:
- [x] Script hatasiz basladi
- [x] Config override uygulandi
- [x] TimeSeriesDataSet olusturuldu
- [x] Known/unknown covariate ayrimi dogru
- [x] DataLoader'lar olustu (train + val)
- [x] Trainer basladi (max_epochs=2)
- [x] Epoch 1 tamamlandi -- train loss loglandi
- [x] Epoch 2 tamamlandi -- val loss loglandi
- [x] Optuna trial'lar tamamlandi
- [x] Best model secildi
- [x] Test seti uzerinde evaluate edildi
- [x] Model kaydedildi (models/tft/)
- [x] GPU/CPU durumu loglandi
- [x] Toplam sure < 10 dakika

### Incelenecek Metrikler:
- [x] Val MAPE: ____% (not al)
- [x] Test MAPE: ____% (not al)
- [x] Train loss dusuyor mu? (epoch 1 > epoch 2)
- [x] Attention weights hesaplandi mi?

### Beklenen Araliklar (smoke, 2 epoch):
- MAPE: %10-%50 arasi normal (cok az epoch, dusuk performans beklenir)
- Sure: 2dk - 10dk (CPU'da yavas)

### Bilinen Sorunlar (Onceki Session'dan):
- TFT smoke test'te hata vermisti -- hata mesajini not et
- Muhtemel sorunlar: TimeSeriesDataSet format, covariate mismatch, memory

### Sorun varsa:
- "ValueError: unknown column" -> tft.yaml covariates listesi ile features uyumsuz
- CUDA error -> CPU'ya dus (accelerator: cpu)
- Memory error -> batch_size kucult veya hidden_size=8 yap
- "No valid time series" -> encoder_length + prediction_length > veri uzunlugu

### Post-training (opsiyonel):
- [x] /debug-model tft calistir
- [x] Notlar: TFT Notlar:
- Val MAPE: 7.80%, Test MAPE: 2.93%
- GPU: RTX 4060 Ti aktif, 6.2 it/s
- 3 epoch, 34695 parametre, 27dk sürdü
- Versiyon fix yapıldı (LightningModule, LearningRateMonitor, GPU fallback)
- Model kaydedildi: models/tft/
- Production'da epoch artırılacak (3 → 50)_______________

---

## Adim 4: Ensemble Smoke Training
Launch config: "Train Ensemble (Smoke)"
Parametreler: 3 model, smoke config

### On Kosul: Adim 1, 2, 3 basariyla tamamlanmis olmali!

### Kontrol Listesi:
- [ ] Script hatasiz basladi
- [ ] 3 model sirayla egitildi (CatBoost -> Prophet -> TFT)
- [ ] Tum modeller basarili (veya graceful degradation aktif)
- [ ] Weight optimization calisti (SLSQP)
- [ ] Optimized weights loglandi
- [ ] Weights sum = 1.0
- [ ] Ensemble MAPE hesaplandi
- [ ] Comparison table yazdirildi
- [ ] ensemble_weights.json kaydedildi
- [ ] Modeller kaydedildi

### Incelenecek Metrikler:
- [ ] CatBoost MAPE: ____%
- [ ] Prophet MAPE: ____%
- [ ] TFT MAPE: ____%
- [ ] Ensemble MAPE: ____% (en iyi tek modelden dusuk olmali)
- [ ] Weights: CatBoost=___, Prophet=___, TFT=___

### Beklenen:
- Ensemble MAPE < en iyi tek model MAPE (degilse weight optimization'da sorun var)

### Sorun varsa:
- Bir model fail -> Graceful degradation devreye girmeli, kalan modellerle devam
- Ensemble MAPE > en iyi tek model -> Weight bounds kontrol et
- "No models available" -> Adim 1-3'te model dosyalari olusmamis

---

## Adim 5: Feature Validation
Command: /feature-check

### Kontrol Listesi:
- [ ] 152+/- feature sayisi dogru
- [ ] NaN orani >%50 olan feature yok
- [ ] Constant feature yok (varyans=0)
- [ ] Target ile korelasyon >0.99 olan feature yok (leakage!)
- [ ] Redundant feature ciftleri not edildi
- [ ] Lag min >= 48 dogrulandi

### Notlar: _______________

---

## Adim 6: Prediction Validation
Command: /predict-check

### On Kosul: Ensemble model egitilmis olmali

### Kontrol Listesi:
- [ ] Forecast parquet 48 satir
- [ ] NaN yok
- [ ] Negatif deger yok
- [ ] Gunduz > Gece pattern'i var
- [ ] Saatlik degisimler smooth
- [ ] Deger araligi mantikli (historical ile uyumlu)

### Notlar: _______________

---

## Adim 7 (Opsiyonel): API Smoke Test
Launch config: "Serve API"

### Kontrol Listesi:
- [ ] uvicorn basladi (port 8000)
- [ ] GET /health -> 200 OK
- [ ] POST /predict (Excel + email) -> job_id dondu
- [ ] GET /status/{job_id} -> status takip edildi
- [ ] Email gonderildi (veya SMTP hata loglandi)

---

## Sonuc Ozeti

| Model | Val MAPE | Test MAPE | Sure | Durum |
|-------|----------|-----------|------|-------|
| CatBoost | ___% | ___% | ___s | _ |
| Prophet | ___% | ___% | ___s | _ |
| TFT | ___% | ___% | ___s | _ |
| Ensemble | ___% | ___% | ___s | _ |

### Karar:
- [ ] Tum modeller calisiyor -> M11'e gec
- [ ] Sorunlar var -> Duzelt ve tekrar test et

### Duzeltilmesi Gereken Sorunlar:
1. _______________
2. _______________
3. _______________

---
> Bu dosya gecicidir. Tum smoke testler basariyla tamamlandiktan sonra silinecektir.
> Tarih: 2026-02-14
---
