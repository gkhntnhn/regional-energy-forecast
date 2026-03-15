import { create } from "zustand";
import type { Job } from "@/api/types";

const channel = new BroadcastChannel("energy-forecast-sync");

interface JobStore {
  activeJobs: Job[];
  lastResult: Job | null;
  addJob: (job: Job) => void;
  updateJob: (jobId: string, updates: Partial<Job>) => void;
  removeJob: (jobId: string) => void;
  setLastResult: (job: Job) => void;
}

export const useJobStore = create<JobStore>((set, get) => {
  channel.onmessage = (event: MessageEvent) => {
    const data = event.data as { type: string; jobs?: Job[]; lastResult?: Job };
    if (data.type === "JOB_SYNC") {
      set({
        activeJobs: data.jobs ?? get().activeJobs,
        lastResult: data.lastResult ?? get().lastResult,
      });
    }
  };

  function broadcast() {
    const state = get();
    channel.postMessage({
      type: "JOB_SYNC",
      jobs: state.activeJobs,
      lastResult: state.lastResult,
    });
  }

  return {
    activeJobs: [],
    lastResult: null,

    addJob: (job) => {
      set((s) => ({ activeJobs: [...s.activeJobs, job] }));
      broadcast();
    },

    updateJob: (jobId, updates) => {
      set((s) => ({
        activeJobs: s.activeJobs.map((j) =>
          j.job_id === jobId ? { ...j, ...updates } : j,
        ),
      }));
      broadcast();
    },

    removeJob: (jobId) => {
      set((s) => ({
        activeJobs: s.activeJobs.filter((j) => j.job_id !== jobId),
      }));
      broadcast();
    },

    setLastResult: (job) => {
      set({ lastResult: job });
      broadcast();
    },
  };
});
