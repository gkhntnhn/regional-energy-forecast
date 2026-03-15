import { useState } from "react";
import { MapeCharts } from "@/components/admin/MapeCharts";
import { ModelComparison } from "@/components/admin/ModelComparison";
import { JobHistory } from "@/components/admin/JobHistory";
import { ModelRuns } from "@/components/admin/ModelRuns";
import { DriftStatus } from "@/components/admin/DriftStatus";
import { SystemHealth } from "@/components/admin/SystemHealth";
import { cn } from "@/lib/utils";
import { BarChart3, Cpu, Cloud, Settings } from "lucide-react";

type Tab = "performance" | "models" | "weather" | "system";

const tabs: { id: Tab; label: string; icon: typeof BarChart3 }[] = [
  { id: "performance", label: "Performans", icon: BarChart3 },
  { id: "models", label: "Modeller", icon: Cpu },
  { id: "weather", label: "Hava Durumu", icon: Cloud },
  { id: "system", label: "Sistem", icon: Settings },
];

export function AdminPage() {
  const [activeTab, setActiveTab] = useState<Tab>("performance");

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-slate-900">Admin Panel</h2>
        <p className="text-sm text-slate-500">Model performansi ve sistem durumu</p>
      </div>

      {/* Tab navigation */}
      <div className="flex gap-1 border-b border-slate-200">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={cn(
              "flex items-center gap-2 px-4 py-2.5 text-sm font-medium border-b-2 transition-colors cursor-pointer",
              activeTab === tab.id
                ? "border-emerald-500 text-slate-900"
                : "border-transparent text-slate-500 hover:text-slate-700",
            )}
          >
            <tab.icon size={16} />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === "performance" && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="lg:col-span-2">
            <MapeCharts />
          </div>
          <ModelComparison />
          <JobHistory />
        </div>
      )}

      {activeTab === "models" && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <ModelRuns />
          <DriftStatus />
        </div>
      )}

      {activeTab === "weather" && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-sm shadow-sm p-5">
            <h3 className="text-sm font-semibold text-slate-900 mb-2">Horizon Dogrulugu</h3>
            <p className="text-sm text-slate-400">Hava durumu tahmin dogrulugu verileri henuz mevcut degil.</p>
          </div>
          <div className="bg-white rounded-sm shadow-sm p-5">
            <h3 className="text-sm font-semibold text-slate-900 mb-2">Degisken Dogrulugu</h3>
            <p className="text-sm text-slate-400">Degisken bazli dogruluk verileri henuz mevcut degil.</p>
          </div>
        </div>
      )}

      {activeTab === "system" && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <SystemHealth />
        </div>
      )}
    </div>
  );
}
