import { useQuery, keepPreviousData } from "@tanstack/react-query";
import { getJobHistory } from "@/api/admin";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Skeleton } from "@/components/ui/Skeleton";
import { formatDateTime, formatMape } from "@/lib/utils";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { useState } from "react";

const PER_PAGE = 15;

export function JobHistory() {
  const [page, setPage] = useState(1);

  const { data, isLoading, error } = useQuery({
    queryKey: ["job-history", page],
    queryFn: () => getJobHistory(page, PER_PAGE),
    placeholderData: keepPreviousData,
  });

  const totalPages = data ? Math.ceil(data.total / PER_PAGE) : 0;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-900">Job Gecmisi</h3>
          {data && (
            <span className="text-xs text-slate-400">
              Toplam: {data.total}
            </span>
          )}
        </div>
      </CardHeader>
      <CardContent className="p-0">
        {isLoading && !data && (
          <div className="p-5 space-y-3">
            {Array.from({ length: 5 }).map((_, i) => (
              <Skeleton key={i} className="h-8 w-full" />
            ))}
          </div>
        )}

        {error && <p className="p-5 text-sm text-rose-500">Veri yuklenemedi</p>}

        {data && data.items.length > 0 && (
          <>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-200 bg-slate-50/50">
                    <th className="text-left px-5 py-2 text-xs font-medium text-slate-500">Job ID</th>
                    <th className="text-left px-5 py-2 text-xs font-medium text-slate-500">Tarih</th>
                    <th className="text-center px-5 py-2 text-xs font-medium text-slate-500">Durum</th>
                    <th className="text-right px-5 py-2 text-xs font-medium text-slate-500">MAPE</th>
                  </tr>
                </thead>
                <tbody>
                  {data.items.map((job) => (
                    <tr
                      key={job.job_id}
                      className="border-b border-slate-100 hover:bg-slate-50 transition-colors duration-150"
                    >
                      <td className="px-5 py-2 font-mono text-xs text-slate-600">
                        {job.job_id.slice(0, 8)}
                      </td>
                      <td className="px-5 py-2 text-xs text-slate-600">
                        {formatDateTime(job.created_at)}
                      </td>
                      <td className="px-5 py-2 text-center">
                        <StatusBadge status={job.status} />
                      </td>
                      <td className="px-5 py-2 text-right font-mono text-xs">
                        {job.mape != null ? formatMape(job.mape) : "-"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-between px-5 py-3 border-t border-slate-200">
                <Button
                  variant="ghost"
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={page <= 1}
                  className="gap-1 text-xs"
                >
                  <ChevronLeft size={14} />
                  Onceki
                </Button>
                <span className="text-xs text-slate-500">
                  {page} / {totalPages}
                </span>
                <Button
                  variant="ghost"
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  disabled={page >= totalPages}
                  className="gap-1 text-xs"
                >
                  Sonraki
                  <ChevronRight size={14} />
                </Button>
              </div>
            )}
          </>
        )}

        {data && data.items.length === 0 && (
          <p className="p-5 text-sm text-slate-400">Henuz job gecmisi yok</p>
        )}
      </CardContent>
    </Card>
  );
}
