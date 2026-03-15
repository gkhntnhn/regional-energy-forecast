import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { BarChart3 } from "lucide-react";
import type { Prediction } from "@/api/types";

interface ResultChartProps {
  predictions: Prediction[];
}

interface ChartDataPoint {
  hour: string;
  value: number;
  period: string;
}

export function ResultChart({ predictions }: ResultChartProps) {
  const data: ChartDataPoint[] = predictions.map((p) => {
    const dt = new Date(p.datetime);
    return {
      hour: `${dt.getDate().toString().padStart(2, "0")}/${(dt.getMonth() + 1).toString().padStart(2, "0")} ${dt.getHours().toString().padStart(2, "0")}:00`,
      value: p.consumption_mwh,
      period: p.period,
    };
  });

  // Find the boundary between intraday and day_ahead
  const boundaryIndex = data.findIndex((d) => d.period === "day_ahead");

  return (
    <Card>
      <CardHeader>
        <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
          <BarChart3 size={16} className="text-slate-400" />
          48 Saatlik Tahmin
        </h3>
      </CardHeader>
      <CardContent>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 4, right: 8, left: 8, bottom: 0 }}>
              <defs>
                <linearGradient id="emeraldGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#10b981" stopOpacity={0.2} />
                  <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid
                strokeDasharray="3 3"
                vertical={false}
                stroke="#f1f5f9"
              />
              <XAxis
                dataKey="hour"
                tick={{ fontSize: 11, fontFamily: "JetBrains Mono", fill: "#94a3b8" }}
                axisLine={{ stroke: "#e2e8f0" }}
                tickLine={false}
                interval="preserveStartEnd"
              />
              <YAxis
                tick={{ fontSize: 11, fontFamily: "JetBrains Mono", fill: "#94a3b8" }}
                axisLine={false}
                tickLine={false}
                width={60}
                tickFormatter={(v: number) => `${v.toLocaleString("tr-TR")}`}
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (!active || !payload?.[0]) return null;
                  const d = payload[0].payload as ChartDataPoint;
                  return (
                    <div className="bg-slate-900 text-white rounded-sm px-3 py-2 text-xs shadow-lg">
                      <p className="font-mono">{d.hour}</p>
                      <p className="font-mono font-medium mt-0.5">
                        {d.value.toLocaleString("tr-TR", { minimumFractionDigits: 1 })} MWh
                      </p>
                      <p className="text-slate-400 mt-0.5">
                        {d.period === "day_ahead" ? "GOP (T+1)" : "GIP (T)"}
                      </p>
                    </div>
                  );
                }}
              />
              {boundaryIndex > 0 && (
                <ReferenceLine
                  x={data[boundaryIndex]?.hour}
                  stroke="#94a3b8"
                  strokeDasharray="4 4"
                  label={{
                    value: "T+1",
                    position: "top",
                    fontSize: 10,
                    fill: "#94a3b8",
                    fontFamily: "JetBrains Mono",
                  }}
                />
              )}
              <Area
                type="monotone"
                dataKey="value"
                stroke="#10b981"
                strokeWidth={2}
                fill="url(#emeraldGradient)"
                dot={false}
                activeDot={{ r: 4, fill: "#10b981", stroke: "#fff", strokeWidth: 2 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
