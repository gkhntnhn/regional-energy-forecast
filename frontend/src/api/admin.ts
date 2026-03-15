import { api } from "./client";
import type {
  MapeDataPoint,
  HourlyMape,
  ModelMape,
  JobHistoryResponse,
  TrainingRun,
  DriftInfo,
  SystemHealthInfo,
  ModelInfo,
} from "./types";

export async function getDailyMape(): Promise<MapeDataPoint[]> {
  return api.get("/admin/analytics/mape/daily");
}

export async function getWeeklyMape(): Promise<MapeDataPoint[]> {
  return api.get("/admin/analytics/mape/weekly");
}

export async function getHourlyMape(): Promise<HourlyMape[]> {
  return api.get("/admin/analytics/mape/hourly");
}

export async function getModelMape(): Promise<ModelMape[]> {
  return api.get("/admin/analytics/models/mape");
}

export async function getModelComparison(): Promise<Record<string, unknown>[]> {
  return api.get("/admin/analytics/models/comparison");
}

export async function getJobHistory(
  page = 1,
  perPage = 20,
): Promise<JobHistoryResponse> {
  return api.get(`/admin/jobs/history?page=${page}&per_page=${perPage}`);
}

export async function getTrainingRuns(): Promise<TrainingRun[]> {
  return api.get("/admin/models/runs");
}

export async function getPromotedModels(): Promise<Record<string, unknown>[]> {
  return api.get("/admin/models/promoted");
}

export async function getDriftStatus(): Promise<DriftInfo[]> {
  return api.get("/admin/models/drift/status");
}

export async function getSystemHealth(): Promise<SystemHealthInfo> {
  return api.get("/admin/system/health");
}

export async function getModels(): Promise<ModelInfo[]> {
  return api.get("/models");
}
