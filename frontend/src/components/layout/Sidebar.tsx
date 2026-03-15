import { NavLink } from "react-router-dom";
import { cn } from "@/lib/utils";
import { BarChart3, Upload, Clock, ChevronLeft, ChevronRight, Zap, X } from "lucide-react";

const navItems = [
  { to: "/", icon: Upload, label: "Dashboard" },
  { to: "/history", icon: Clock, label: "Gecmis" },
  { to: "/admin", icon: BarChart3, label: "Admin" },
];

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
  mobileOpen: boolean;
  onMobileClose: () => void;
}

export function Sidebar({ collapsed, onToggle, mobileOpen, onMobileClose }: SidebarProps) {
  return (
    <>
      {/* Mobile overlay */}
      {mobileOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={onMobileClose}
        />
      )}

      <aside
        className={cn(
          "flex flex-col bg-slate-900 text-white h-screen z-50",
          // Desktop
          "hidden lg:flex lg:sticky lg:top-0 transition-[width] duration-200",
          collapsed ? "lg:w-14" : "lg:w-60",
          // Mobile
          mobileOpen && "!flex fixed inset-y-0 left-0 w-60",
        )}
      >
        <div className="flex items-center justify-between px-4 h-14 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <Zap size={20} className="text-emerald-500 shrink-0" />
            {(!collapsed || mobileOpen) && (
              <span className="text-sm font-semibold tracking-tight truncate">
                Energy Forecast
              </span>
            )}
          </div>
          {/* Mobile close */}
          {mobileOpen && (
            <button onClick={onMobileClose} className="lg:hidden text-slate-400 hover:text-white cursor-pointer">
              <X size={18} />
            </button>
          )}
        </div>

        <nav className="flex-1 py-3">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              onClick={onMobileClose}
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
              {(!collapsed || mobileOpen) && <span>{item.label}</span>}
            </NavLink>
          ))}
        </nav>

        {/* Collapse toggle — desktop only */}
        <button
          onClick={onToggle}
          className="hidden lg:flex items-center justify-center h-10 border-t border-slate-800 text-slate-500 hover:text-white transition-colors cursor-pointer"
        >
          {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </button>
      </aside>
    </>
  );
}
