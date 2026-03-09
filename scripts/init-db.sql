-- Initialize additional databases for MLflow and Optuna.
-- Main energy_forecast database is created by POSTGRES_DB env var.

-- MLflow experiment tracking backend store
CREATE DATABASE mlflow;
GRANT ALL PRIVILEGES ON DATABASE mlflow TO forecast_user;
