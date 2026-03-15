import { api } from "./client";
import type { Job, Prediction } from "./types";

export async function createPrediction(
  file: File,
  email: string,
): Promise<Job> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("email", email);
  return api.post<Job>("/predict", formData);
}

export async function getJobStatus(jobId: string): Promise<Job> {
  return api.get<Job>(`/status/${jobId}`);
}

interface JobDetailsResponse {
  id: string;
  status: string;
  predictions: Array<{
    forecast_dt: string;
    consumption_mwh: number;
    period: string;
    model_source: string;
  }>;
  metadata: Record<string, unknown> | null;
}

export async function getJobPredictions(jobId: string): Promise<Prediction[]> {
  const details = await api.get<JobDetailsResponse>(
    `/admin/jobs/${jobId}/details`,
  );

  // Filter to ensemble predictions only (main output)
  return details.predictions
    .filter((p) => p.model_source === "ensemble")
    .map((p) => ({
      datetime: p.forecast_dt,
      consumption_mwh: p.consumption_mwh,
      period: p.period as "intraday" | "day_ahead",
    }));
}

export async function cancelJob(jobId: string): Promise<void> {
  await api.delete(`/status/${jobId}`);
}

export async function getActiveJob(): Promise<{ active: boolean; job_id?: string }> {
  return api.get("/status/active");
}

export async function getJobs(): Promise<Job[]> {
  return api.get<Job[]>("/jobs");
}
