import { useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { getJobs } from "@/api/predict";
import { useJobStore } from "@/stores/jobs";

/**
 * On mount, fetch active jobs from backend and hydrate Zustand store.
 * This restores the queue after page refresh. Runs only once.
 */
export function useActiveJobs() {
  const addJob = useJobStore((s) => s.addJob);
  const hasHydrated = useRef(false);

  const { data } = useQuery({
    queryKey: ["jobs-list"],
    queryFn: getJobs,
    staleTime: 10_000,
  });

  useEffect(() => {
    if (!data || hasHydrated.current) return;
    hasHydrated.current = true;

    // Backend /jobs returns { jobs: [...] }
    const jobs = (data as unknown as { jobs: Array<{ id: string; status: string; created_at: string }> }).jobs;
    if (!jobs) return;

    const active = jobs.filter(
      (j) => j.status === "queued" || j.status === "running" || j.status === "pending",
    );

    for (const j of active) {
      addJob({
        job_id: j.id,
        status: j.status as "queued" | "running",
        created_at: j.created_at,
      });
    }
  }, [data, addJob]);
}
