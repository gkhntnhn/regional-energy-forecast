import { useQuery } from "@tanstack/react-query";
import { getJobStatus } from "@/api/predict";
import { useJobStore } from "@/stores/jobs";
import { POLL_INTERVAL_MS } from "@/lib/constants";
import { useEffect } from "react";

export function useJobPolling(jobId: string | null) {
  const updateJob = useJobStore((s) => s.updateJob);
  const removeJob = useJobStore((s) => s.removeJob);
  const setLastResult = useJobStore((s) => s.setLastResult);

  const query = useQuery({
    queryKey: ["job-status", jobId],
    queryFn: () => getJobStatus(jobId!),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === "completed" || status === "failed" || status === "cancelled") {
        return false;
      }
      return POLL_INTERVAL_MS;
    },
  });

  useEffect(() => {
    if (!query.data || !jobId) return;

    const { status } = query.data;
    updateJob(jobId, query.data);

    if (status === "completed" || status === "failed" || status === "cancelled") {
      removeJob(jobId);
      if (status === "completed") {
        setLastResult(query.data);
      }
    }
  }, [query.data, jobId, updateJob, removeJob, setLastResult]);

  return query;
}
