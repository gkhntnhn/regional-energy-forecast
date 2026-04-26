# Scheduler Configuration — Daily Ingestion

Production deployments must keep the EPIAS market/generation and weather
tables fresh. The application exposes two token-protected endpoints under
`/internal/*` that any HTTP scheduler can invoke daily.

This document covers cron / Windows Task Scheduler / GCP Cloud Scheduler /
GitHub Actions setup for the recommended 06:00 + 06:30 schedule.

---

## Endpoint reference

| Method | Path                                  | Effect                                      | Duration  | Idempotent |
|--------|---------------------------------------|---------------------------------------------|-----------|------------|
| POST   | `/internal/sync-epias?days_back=N`    | Reconcile EPIAS parquet cache -> DB tables  | 20-30s    | yes        |
| POST   | `/internal/sync-weather?days_back=N`  | Reconcile weather parquet cache -> DB       | 4-6 min   | yes        |

Both endpoints require the header `X-Internal-Token: <INTERNAL_TOKEN>`.
Both return JSON of shape:

```json
{
  "sync_type": "epias" | "weather",
  "days_back": 7,
  "rows_fetched": 0,
  "rows_upserted": 110766,
  "last_date": "2026-04-26",
  "duration_ms": 25922,
  "errors": []
}
```

`errors[]` is non-empty for partial-failure cases — treat HTTP 200 with
non-empty `errors` as a soft warning rather than success.

### Status codes

| Code | Meaning                                  | Operator action                          |
|------|------------------------------------------|------------------------------------------|
| 200  | Success (check `errors[]`)               | Inspect audit_logs for partial failures  |
| 401  | Missing or invalid `X-Internal-Token`    | Rotate token, update scheduler secret    |
| 503  | `INTERNAL_TOKEN` not configured on host  | Set env var, restart server              |

---

## Token management

The internal token gates operator-only endpoints, separately from the
user-facing `API_KEY`. It is read from the `INTERNAL_TOKEN` environment
variable. Empty value disables the endpoints (503).

Generate a fresh token:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Rotation cadence: **quarterly** (or after staff change). Update both the
server env var and every scheduler entry in lockstep — endpoints rejected
401 immediately after rotation if the scheduler is stale.

---

## Recommended schedule

```
06:00  POST /internal/sync-epias?days_back=7
06:30  POST /internal/sync-weather?days_back=2
```

Rationale:

- **EPIAS** publishes T-1 market data overnight (~03:00-05:00 TR). 06:00 is
  the earliest reliable window.
- **OpenMeteo** T-2 actuals stabilize before 06:00 TR; T-1 is partial.
- The 30-minute gap prevents the two heavy DB write windows from overlapping.

`days_back` is a soft window — values above the deepest gap have no extra
cost since the upsert is idempotent.

---

## Platform setup

### A) Linux / macOS cron

`/etc/cron.d/energy-forecast-sync`:

```cron
# Daily sync — keep DB fresh against parquet cache + upstream APIs
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
TOKEN=replace-me

0 6 * * * www-data curl -fsS -X POST -H "X-Internal-Token: $TOKEN" "https://your-host/internal/sync-epias?days_back=7"   >> /var/log/sync-epias.log   2>&1
30 6 * * * www-data curl -fsS -X POST -H "X-Internal-Token: $TOKEN" "https://your-host/internal/sync-weather?days_back=2" >> /var/log/sync-weather.log 2>&1
```

Use `--max-time 540` (9 min) for `sync-weather` to absorb the slower path.

### B) Windows Task Scheduler

Two tasks via `schtasks /Create`:

```powershell
$Token = "replace-me"
$Url1  = "http://localhost:8000/internal/sync-epias?days_back=7"
$Url2  = "http://localhost:8000/internal/sync-weather?days_back=2"

$Action1 = "powershell -Command `"Invoke-RestMethod -Method Post -Uri '$Url1' -Headers @{ 'X-Internal-Token' = '$Token' }`""
$Action2 = "powershell -Command `"Invoke-RestMethod -Method Post -Uri '$Url2' -Headers @{ 'X-Internal-Token' = '$Token' } -TimeoutSec 540`""

schtasks /Create /TN "EnergyForecast-SyncEpias"   /TR $Action1 /SC DAILY /ST 06:00 /RU SYSTEM /F
schtasks /Create /TN "EnergyForecast-SyncWeather" /TR $Action2 /SC DAILY /ST 06:30 /RU SYSTEM /F
```

The `/RU SYSTEM` flag avoids per-user logon dependency. Replace the URLs
when the API is exposed on a non-localhost host.

### C) GCP Cloud Scheduler

```bash
TOKEN=$(gcloud secrets versions access latest --secret=energy-forecast-internal-token)
URL_BASE="https://energy-forecast-xxxxx-ew.a.run.app"

gcloud scheduler jobs create http sync-epias \
  --schedule "0 6 * * *" \
  --time-zone "Europe/Istanbul" \
  --uri "${URL_BASE}/internal/sync-epias?days_back=7" \
  --http-method POST \
  --headers "X-Internal-Token=${TOKEN}" \
  --attempt-deadline 60s

gcloud scheduler jobs create http sync-weather \
  --schedule "30 6 * * *" \
  --time-zone "Europe/Istanbul" \
  --uri "${URL_BASE}/internal/sync-weather?days_back=2" \
  --http-method POST \
  --headers "X-Internal-Token=${TOKEN}" \
  --attempt-deadline 540s
```

Store the token in Secret Manager (`gcloud secrets create
energy-forecast-internal-token`) and reference it from Cloud Run env vars
to keep one source of truth.

### D) GitHub Actions

`.github/workflows/daily-sync.yml`:

```yaml
name: Daily Sync
on:
  schedule:
    - cron: "0 3 * * *"   # 06:00 TR (UTC+3)
    - cron: "30 3 * * *"  # 06:30 TR
  workflow_dispatch:

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - name: Pick endpoint by schedule
        id: route
        run: |
          if [[ "${{ github.event.schedule }}" == "0 3 * * *" ]]; then
            echo "path=sync-epias?days_back=7" >> $GITHUB_OUTPUT
            echo "timeout=60" >> $GITHUB_OUTPUT
          else
            echo "path=sync-weather?days_back=2" >> $GITHUB_OUTPUT
            echo "timeout=540" >> $GITHUB_OUTPUT
          fi
      - name: Call sync endpoint
        run: |
          curl -fsS -X POST \
            --max-time ${{ steps.route.outputs.timeout }} \
            -H "X-Internal-Token: ${{ secrets.INTERNAL_TOKEN }}" \
            "${{ secrets.API_URL }}/internal/${{ steps.route.outputs.path }}"
```

GitHub cron triggers run on UTC; convert to your timezone explicitly.

---

## Monitoring

Every sync invocation writes an audit row:

```sql
SELECT created_at, action, jsonb_pretty(details::jsonb)
  FROM audit_logs
 WHERE action IN ('sync_epias', 'sync_weather')
 ORDER BY created_at DESC
 LIMIT 10;
```

Suggested alerts:

- **Missing daily run**: no `sync_epias` row in the last 26h -> page operator
- **Stale data**: `details->>'last_date'` lag > 2 days -> investigate parquet
  cache staleness (run `make backfill-epias` / `make seed-weather`)
- **Repeated `errors[]`**: warn after 2 consecutive non-empty error lists

---

## Troubleshooting

| Symptom                                  | Cause                                    | Fix                                      |
|------------------------------------------|------------------------------------------|------------------------------------------|
| HTTP 200 with `rows_upserted: 0`         | Parquet cache empty                      | `make backfill-epias` then retry          |
| HTTP 200 with non-empty `errors[]`       | Partial seed failure                     | Read `details.log_tail` in audit_logs    |
| HTTP 401                                 | Stale token in scheduler                 | Update scheduler secret to match env     |
| HTTP 503                                 | `INTERNAL_TOKEN` empty on server         | Set env var, restart                     |
| HTTP 504 / connection timeout            | `sync-weather` exceeded scheduler budget | Raise scheduler timeout to 540s          |
