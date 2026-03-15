import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { formatMwh, formatDateTime } from "@/lib/utils";
import { Download, Table } from "lucide-react";
import type { Prediction } from "@/api/types";
import type { ForecastType } from "@/api/types";
import { useState } from "react";
import { API_BASE_URL } from "@/lib/constants";
import { useAuthStore } from "@/stores/auth";

interface ResultTableProps {
  predictions: Prediction[];
  downloadUrl?: string;
  stats?: { count: number; mean: number; min: number; max: number };
}

export function ResultTable({ predictions, downloadUrl, stats }: ResultTableProps) {
  const [forecastType, setForecastType] = useState<ForecastType>("full");
  const token = useAuthStore((s) => s.token);

  const filtered = forecastType === "day_ahead"
    ? predictions.filter((p) => p.period === "day_ahead")
    : predictions;

  function handleDownload() {
    if (!downloadUrl) return;
    const link = document.createElement("a");
    link.href = `${API_BASE_URL}${downloadUrl}`;
    if (token) {
      // Use fetch for authenticated download
      fetch(`${API_BASE_URL}${downloadUrl}`, {
        headers: { Authorization: `Bearer ${token}` },
      })
        .then((r) => r.blob())
        .then((blob) => {
          const url = URL.createObjectURL(blob);
          link.href = url;
          link.download = downloadUrl.split("/").pop() ?? "forecast.xlsx";
          link.click();
          URL.revokeObjectURL(url);
        });
      return;
    }
    link.click();
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
            <Table size={16} className="text-slate-400" />
            Tahmin Sonuclari
          </h3>
          <div className="flex items-center gap-3">
            {/* Forecast type toggle */}
            <div className="flex bg-slate-100 rounded-sm p-0.5">
              <button
                onClick={() => setForecastType("day_ahead")}
                className={`px-3 py-1 text-xs font-medium rounded-sm transition-colors cursor-pointer ${
                  forecastType === "day_ahead"
                    ? "bg-white text-slate-900 shadow-sm"
                    : "text-slate-500 hover:text-slate-700"
                }`}
              >
                T+1 (GOP)
              </button>
              <button
                onClick={() => setForecastType("full")}
                className={`px-3 py-1 text-xs font-medium rounded-sm transition-colors cursor-pointer ${
                  forecastType === "full"
                    ? "bg-white text-slate-900 shadow-sm"
                    : "text-slate-500 hover:text-slate-700"
                }`}
              >
                T+T+1
              </button>
            </div>

            {downloadUrl && (
              <Button variant="secondary" onClick={handleDownload} className="gap-1.5 text-xs">
                <Download size={14} />
                Excel
              </Button>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        {/* Stats bar */}
        {stats && (
          <div className="flex gap-6 px-5 py-3 border-b border-slate-100 bg-slate-50/50">
            <Stat label="Ortalama" value={`${formatMwh(stats.mean)} MWh`} />
            <Stat label="Min" value={`${formatMwh(stats.min)} MWh`} />
            <Stat label="Max" value={`${formatMwh(stats.max)} MWh`} />
            <Stat label="Satir" value={String(filtered.length)} />
          </div>
        )}

        {/* Table */}
        <div className="overflow-x-auto max-h-96">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-white">
              <tr className="border-b border-slate-200">
                <th className="text-left px-5 py-2 text-xs font-medium text-slate-500">Saat</th>
                <th className="text-right px-5 py-2 text-xs font-medium text-slate-500">Tuketim (MWh)</th>
                <th className="text-center px-5 py-2 text-xs font-medium text-slate-500">Donem</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((p, i) => (
                <tr
                  key={i}
                  className="border-b border-slate-100 hover:bg-slate-50 transition-colors duration-150"
                >
                  <td className="px-5 py-2 font-mono text-xs text-slate-600">
                    {formatDateTime(p.datetime)}
                  </td>
                  <td className="px-5 py-2 text-right font-mono text-xs font-medium text-slate-900">
                    {formatMwh(p.consumption_mwh)}
                  </td>
                  <td className="px-5 py-2 text-center">
                    <span
                      className={`inline-block px-2 py-0.5 text-xs rounded-sm ${
                        p.period === "day_ahead"
                          ? "bg-emerald-50 text-emerald-700"
                          : "bg-amber-50 text-amber-700"
                      }`}
                    >
                      {p.period === "day_ahead" ? "GOP" : "GIP"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs text-slate-400">{label}</p>
      <p className="text-sm font-mono font-medium text-slate-900">{value}</p>
    </div>
  );
}
