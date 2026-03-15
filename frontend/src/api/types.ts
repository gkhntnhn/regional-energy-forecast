export type JobStatus = "pending" | "processing" | "completed" | "failed" | "cancelled" | "running";

export interface Job {
  job_id: string;
  status: JobStatus;
  created_at: string;
  completed_at?: string;
  progress?: number | string;
  error?: string;
}

export interface Prediction {
  datetime: string;
  consumption_mwh: number;
  period: "intraday" | "day_ahead";
}

export interface JobResult {
  success: boolean;
  predictions: Prediction[];
  metadata: {
    model: string;
    weights: Record<string, number>;
    last_data_point: string;
    weather_updated_at: string;
    latency_ms: number;
  };
  statistics: {
    count: number;
    mean: number;
    min: number;
    max: number;
  };
  download_url: string;
}

export interface ModelInfo {
  name: string;
  version: string;
  loaded: boolean;
  path: string;
}

export interface MapeDataPoint {
  date: string;
  mape: number;
}

export interface HourlyMape {
  hour: number;
  mape: number;
}

export interface ModelMape {
  model: string;
  mape: number;
  mae: number;
  rmse: number;
}

// Matches actual backend response
export interface JobHistoryItem {
  id: string;
  email?: string;
  status: JobStatus;
  progress?: string;
  created_at: string;
  completed_at?: string;
}

export interface JobHistoryResponse {
  total: number;
  page: number;
  size: number;
  pages: number;
  jobs: JobHistoryItem[];
}

// Matches actual backend response
export interface TrainingRun {
  id: number;
  model_type: string;
  status: string;
  val_mape: number | null;
  test_mape: number | null;
  n_trials: number;
  n_splits: number;
  feature_count: number;
  is_promoted: boolean;
  started_at: string;
  completed_at: string | null;
  duration_seconds: number | null;
}

export interface DriftInfo {
  feature: string;
  drift_score: number;
  is_drifted: boolean;
  timestamp: string;
}

// Matches actual backend response
export interface SystemHealthInfo {
  database: string;
  models_loaded: boolean;
  model_info: {
    loaded: boolean;
    active_models: string[];
    weights: Record<string, number>;
    forecast_horizon: number;
  };
}

export type ForecastType = "day_ahead" | "full";
