import { NavLink } from "react-router-dom";
import { cn } from "@/lib/utils";
import { BarChart3, Upload, Clock, Settings, ChevronLeft, ChevronRight, Zap } from "lucide-react";
import { useState } from "react";

const navItems = [
  { to: "/", icon: Upload, label: "Dashboard" },
  { to: "/history", icon: Clock, label: "Gecmis" },
  { to: "/admin", icon: BarChart3, label: "Admin" },
];

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className={cn(
        "flex flex-col bg-slate-900 text-white transition-[width] duration-200 h-screen sticky top-0",
        collapsed ? "w-14" : "w-60",
      )}
    >
      <div className="flex items-center gap-3 px-4 h-14 border-b border-slate-800">
        <Zap size={20} className="text-emerald-500 shrink-0" />
        {!collapsed && (
          <span className="text-sm font-semibold tracking-tight truncate">
            Energy Forecast
          </span>
        )}
      </div>

      <nav className="flex-1 py-3">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 px-4 py-2.5 text-sm transition-colors duration-150",
                isActive
                  ? "bg-slate-800 text-white border-r-2 border-emerald-500"
                  : "text-slate-400 hover:text-white hover:bg-slate-800/50",
              )
            }
          >
            <item.icon size={18} className="shrink-0" />
            {!collapsed && <span>{item.label}</span>}
          </NavLink>
        ))}
      </nav>

      <div className="border-t border-slate-800">
        <NavLink
          to="/admin"
          className="flex items-center gap-3 px-4 py-2.5 text-sm text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors"
        >
          <Settings size={18} className="shrink-0" />
          {!collapsed && <span>Ayarlar</span>}
        </NavLink>
      </div>

      <button
        onClick={() => setCollapsed(!collapsed)}
        className="flex items-center justify-center h-10 border-t border-slate-800 text-slate-500 hover:text-white transition-colors cursor-pointer"
      >
        {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
      </button>
    </aside>
  );
}
