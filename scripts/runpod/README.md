# RunPod GPU HPO Training Guide

Production-level hyperparameter optimization on RunPod cloud GPU.

## Why RunPod?

- TFT (PyTorch) needs GPU — H100 Tensor Cores for fastest training
- CatBoost `task_type: "GPU"` gives 2-5x speedup
- 50 CatBoost + 30 Prophet + 20 TFT trials with 12-fold TSCV would take 10+ hours locally

## GPU: H100 PCIe ($1.99/hr)

| Spec | Value |
|------|-------|
| GPU | NVIDIA H100 PCIe, 80GB VRAM |
| RAM | 188 GB |
| vCPU | 16 |
| Estimated time | 2-3.5 hours |
| Estimated cost | $4-7 |

## Quick Start

```
LOCAL                              POD
─────                              ───
1. pack_data.sh ──SCP──►  2. setup_pod.sh
                           3. apply_prod_config.sh
                           4. run_training.sh
                           5. pack_results.sh
6. download_results.sh ◄──SCP──
7. Verify locally
8. STOP THE POD!
```

## Step-by-Step

### 0. Prerequisites (one-time)

```bash
# SSH key (skip if you already have one)
ssh-keygen -t ed25519 -C "runpod"
cat ~/.ssh/id_ed25519.pub
# → Paste into RunPod: https://www.runpod.io/console/user/settings → SSH Keys
```

### 1. Package Data (local)

```bash
cd C:\Users\pc\Desktop\Python\Projects\regional-energy-forecast

# Refresh weather cache (optional — if stale)
rm -f data/external/weather_cache.sqlite
uv run python scripts/prepare_dataset.py --skip-epias -v

# Package data files (~28MB)
bash scripts/runpod/pack_data.sh
```

### 2. Create RunPod Pod

1. https://www.runpod.io/console/pods → **Deploy**
2. GPU: **H100 PCIe** (1x)
3. Template: **RunPod PyTorch 2.x**
4. Container Disk: **50 GB**
5. Volume Disk: **20 GB** (optional, for persistence)
6. Expose TCP Port: **22** (SSH)
7. **Deploy**

Wait for status "Running" → click **Connect** → copy SSH command.

### 3. Transfer Data to Pod

```bash
# Replace <POD_IP> and <PORT> from RunPod dashboard
scp -P <PORT> -i ~/.ssh/id_ed25519 runpod_data.tar.gz root@<POD_IP>:/workspace/

# Optional: transfer .env if EPIAS credentials needed
scp -P <PORT> -i ~/.ssh/id_ed25519 .env root@<POD_IP>:/workspace/.env
```

### 4. SSH into Pod & Setup

```bash
ssh root@<POD_IP> -p <PORT> -i ~/.ssh/id_ed25519

# Inside pod:
cd /workspace
# Edit the REPO_URL in the command or pass as argument
bash regional-energy-forecast/scripts/runpod/setup_pod.sh https://github.com/YOUR_USER/regional-energy-forecast.git
# If setup_pod.sh isn't available yet (first clone), do manually:
#   git clone <REPO_URL>
#   cd regional-energy-forecast
#   tar -xzf /workspace/runpod_data.tar.gz
#   pip install uv && uv sync --all-extras
#   uv run python scripts/prepare_dataset.py --skip-epias -v
```

### 5. Apply Production Config

```bash
cd /workspace/regional-energy-forecast
bash scripts/runpod/apply_prod_config.sh
```

Changes applied:
- CatBoost: 50 trials, GPU, iterations 500-3000, 3 loss functions
- Prophet: 30 trials
- TFT: 20 trials, hidden 32-128, epochs 50, 4 workers
- Cross-validation: 12 splits

### 6. Start Training

```bash
bash scripts/runpod/run_training.sh
```

This starts all 4 trainings in a **tmux** session (SSH-safe):
1. CatBoost (GPU) — ~30-60 min
2. Prophet (CPU) — ~45-90 min
3. TFT (GPU) — ~60-120 min
4. Ensemble — ~10-15 min

```bash
# Watch live
tmux attach -t hpo-training

# Detach (training continues)
# Press: Ctrl+B, then D

# Check if still running
tmux ls
```

### 7. Package Results (on pod)

After training completes:
```bash
cd /workspace/regional-energy-forecast
bash scripts/runpod/pack_results.sh
```

### 8. Download Results (local)

```bash
cd C:\Users\pc\Desktop\Python\Projects\regional-energy-forecast
bash scripts/runpod/download_results.sh <POD_IP> <PORT>
```

### 9. STOP THE POD!

Go to RunPod dashboard → **Stop** or **Delete** the pod.
Idle pods still bill at $1.99/hr!

### 10. Verify Locally

```bash
# Run tests
make test

# Start API and test prediction
make serve
# In another terminal:
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer <API_KEY>" \
  -F "file=@data/raw/Consumption_Input_Format.xlsx"
```

## Script Reference

| Script | Where | Purpose |
|--------|-------|---------|
| `pack_data.sh` | Local | Package data files for transfer |
| `setup_pod.sh` | Pod | Clone repo, extract data, install deps |
| `apply_prod_config.sh` | Pod | Switch to production HPO config |
| `restore_dev_config.sh` | Either | Revert to dev config values |
| `run_training.sh` | Pod | Run all 4 trainings in tmux |
| `pack_results.sh` | Pod | Package models/logs for download |
| `download_results.sh` | Local | SCP download + extract results |

## Troubleshooting

**CatBoost GPU fails:** Some RunPod templates don't have CatBoost GPU support.
Edit `catboost.yaml` → `task_type: "CPU"` and continue. Training will be slower but works.

**SSH disconnects:** Training continues in tmux. Reconnect and `tmux attach -t hpo-training`.

**Out of disk:** Increase container disk in RunPod dashboard (requires pod restart).

**CUDA not found:** Make sure you selected a GPU template (PyTorch 2.x), not a CPU-only one.

**Weather API timeout:** The weather cache is transferred from local. If `prepare_dataset.py`
needs fresh data, OpenMeteo is free and doesn't need credentials.
