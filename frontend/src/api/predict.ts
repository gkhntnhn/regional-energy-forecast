import { api } from "./client";
import type { Job, JobResult } from "./types";

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

export async function getJobResult(jobId: string): Promise<JobResult> {
  return api.get<JobResult>(`/status/${jobId}`);
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
