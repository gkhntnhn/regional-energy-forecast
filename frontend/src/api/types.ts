export type JobStatus = "pending" | "processing" | "completed" | "failed" | "cancelled";

export interface Job {
  job_id: string;
  status: JobStatus;
  created_at: string;
  completed_at?: string;
  progress?: number;
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

export interface JobHistoryItem {
  job_id: string;
  status: JobStatus;
  created_at: string;
  completed_at?: string;
  mape?: number;
  excel_file?: string;
}

export interface TrainingRun {
  run_id: string;
  model: string;
  started_at: string;
  mape: number;
  params: Record<string, unknown>;
}

export interface DriftInfo {
  feature: string;
  drift_score: number;
  is_drifted: boolean;
  timestamp: string;
}

export interface SystemHealthInfo {
  status: string;
  uptime_seconds: number;
  queue_size: number;
  db_connected: boolean;
  models_loaded: string[];
}

export type ForecastType = "day_ahead" | "full";
