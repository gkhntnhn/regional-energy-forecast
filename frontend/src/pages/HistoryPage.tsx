import { useState } from "react";
import { useQuery, keepPreviousData } from "@tanstack/react-query";
import { getJobHistory } from "@/api/admin";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Skeleton } from "@/components/ui/Skeleton";
import { formatDateTime } from "@/lib/utils";
import { ChevronLeft, ChevronRight, Download } from "lucide-react";
import { API_BASE_URL } from "@/lib/constants";
import { useAuthStore } from "@/stores/auth";

const PER_PAGE = 20;

export function HistoryPage() {
  const [page, setPage] = useState(1);
  const token = useAuthStore((s) => s.token);

  const { data, isLoading, error } = useQuery({
    queryKey: ["history-jobs", page],
    queryFn: () => getJobHistory(page, PER_PAGE),
    placeholderData: keepPreviousData,
  });

  const totalPages = data?.pages ?? 0;

  function handleDownload(resultFile: string) {
    fetch(`${API_BASE_URL}/files/${resultFile}`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((r) => {
        if (!r.ok) return;
        return r.blob();
      })
      .then((blob) => {
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = resultFile;
        a.click();
        URL.revokeObjectURL(url);
      });
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-slate-900">Gecmis Tahminler</h2>
        <p className="text-sm text-slate-500">Onceki tahmin sonuclari</p>
      </div>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-900">Tahmin Gecmisi</h3>
            {data && (
              <span className="text-xs text-slate-400">Toplam: {data.total}</span>
            )}
          </div>
        </CardHeader>
        <CardContent className="p-0">
          {isLoading && !data && (
            <div className="p-5 space-y-3">
              {Array.from({ length: 8 }).map((_, i) => (
                <Skeleton key={i} className="h-10 w-full" />
              ))}
            </div>
          )}

          {error && <p className="p-5 text-sm text-rose-500">Veri yuklenemedi</p>}

          {data && data.jobs.length > 0 && (
            <>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-200 bg-slate-50/50">
                      <th className="text-left px-5 py-2.5 text-xs font-medium text-slate-500">Job ID</th>
                      <th className="text-left px-5 py-2.5 text-xs font-medium text-slate-500">Olusturma</th>
                      <th className="text-left px-5 py-2.5 text-xs font-medium text-slate-500">Tamamlanma</th>
                      <th className="text-center px-5 py-2.5 text-xs font-medium text-slate-500">Durum</th>
                      <th className="text-left px-5 py-2.5 text-xs font-medium text-slate-500">Ilerleme</th>
                      <th className="text-center px-5 py-2.5 text-xs font-medium text-slate-500">Indir</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.jobs.map((job) => {
                      const resultFile = (job as unknown as { result_file?: string }).result_file;
                      return (
                        <tr
                          key={job.id}
                          className="border-b border-slate-100 hover:bg-slate-50 transition-colors duration-150"
                        >
                          <td className="px-5 py-2.5 font-mono text-xs text-slate-600">
                            {job.id.slice(0, 12)}
                          </td>
                          <td className="px-5 py-2.5 text-xs text-slate-600">
                            {formatDateTime(job.created_at)}
                          </td>
                          <td className="px-5 py-2.5 text-xs text-slate-600">
                            {job.completed_at ? formatDateTime(job.completed_at) : "-"}
                          </td>
                          <td className="px-5 py-2.5 text-center">
                            <StatusBadge status={job.status} />
                          </td>
                          <td className="px-5 py-2.5 text-xs text-slate-500 max-w-[200px] truncate">
                            {job.progress ?? "-"}
                          </td>
                          <td className="px-5 py-2.5 text-center">
                            {job.status === "completed" && resultFile && (
                              <button
                                onClick={() => handleDownload(resultFile)}
                                className="text-emerald-600 hover:text-emerald-700 cursor-pointer"
                                title="Excel indir"
                              >
                                <Download size={14} />
                              </button>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

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

          {data && data.jobs.length === 0 && (
            <p className="p-5 text-sm text-slate-400">Henuz tahmin gecmisi yok</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
