# M6: Prophet Training — Implementasyon Planı

> Tarih: 2026-02-07
> Bağımlılık: M0-M5 tamamlanmış olmalı
> Öncelik: P0 — ikinci model, ensemble için gerekli

---

## Kapsam

M6, Prophet modelinin eğitim pipeline'ını kurar. M5'te oluşturulan ortak altyapıyı
(splitter, search, metrics, experiment) kullanır.

### M5'ten Miras Alınan (Değişiklik Yok)
- `TimeSeriesSplitter` — takvim ayı bazlı TSCV
- `suggest_params` — dinamik Optuna search space parser
- `compute_all` / `MetricsResult` — metrik hesaplama

### Prophet-Spesifik Eklentiler
- `ProphetTrainer` — Prophet eğitim pipeline'ı
- `ProphetForecaster` güncelleme — stub → gerçek implementasyon
- `hyperparameters.yaml` — Prophet search space tanımı
- CLI güncellemesi — `--model prophet` desteği

---

## Dosya Değişiklikleri

### Yeni Dosyalar
| Dosya | Açıklama |
|-------|----------|
| `src/energy_forecast/training/prophet_trainer.py` | Prophet eğitim + Optuna |
| `tests/unit/test_training/test_prophet_trainer.py` | Trainer testleri |
| `tests/unit/test_models/test_prophet_forecaster.py` | Forecaster testleri |

### Güncellenen Dosyalar
| Dosya | Değişiklik |
|-------|-----------|
| `configs/models/hyperparameters.yaml` | Prophet search space eklendi |
| `src/energy_forecast/models/prophet.py` | Stub → gerçek implementasyon |
| `src/energy_forecast/training/run.py` | `--model prophet` CLI desteği |
| `src/energy_forecast/training/experiment.py` | `log_prophet_model` metodu |

---

## Prophet Format Dönüşümü

Prophet standart DataFrame yerine özel format bekler:
- `ds`: datetime kolonu (zorunlu)
- `y`: target kolonu (eğitim için zorunlu)
- Regressor kolonları: config'den okunur

Dönüşüm trainer'da yapılır (`_to_prophet_format`), forecaster basit kalır.

---

## Search Space

```yaml
prophet:
  n_trials: 30
  search_space:
    changepoint_prior_scale:
      type: float
      low: 0.001
      high: 0.5
      log: true
    seasonality_prior_scale:
      type: float
      low: 0.01
      high: 50.0
      log: true
    holidays_prior_scale:
      type: float
      low: 0.01
      high: 50.0
      log: true
    seasonality_mode:
      type: categorical
      choices: ["additive", "multiplicative"]
```

---

## Çıkış Kriterleri

- [x] Tüm testler geçer
- [x] Lint temiz
- [x] Aynı splitter kullanılır (TSCV 12 split)
- [x] Dinamik search space (YAML'dan)
- [x] Prophet ds+y formatı (trainer'da)
- [x] Regressor'lar config'den okunur
- [x] Holidays parquet'ten yüklenir
- [x] CLI `--model prophet` çalışır
- [x] Ay bazlı MAPE raporlanır
