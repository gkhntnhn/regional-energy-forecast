import { useAuthStore } from "@/stores/auth";
import { Button } from "@/components/ui/Button";
import { LogOut, Menu } from "lucide-react";

interface HeaderProps {
  onMenuClick: () => void;
}

export function Header({ onMenuClick }: HeaderProps) {
  const logout = useAuthStore((s) => s.logout);

  return (
    <header className="flex items-center justify-between h-14 px-4 sm:px-6 bg-white border-b border-slate-200">
      <div className="flex items-center gap-3">
        <button
          onClick={onMenuClick}
          className="lg:hidden text-slate-500 hover:text-slate-700 cursor-pointer"
        >
          <Menu size={20} />
        </button>
        <h1 className="text-base sm:text-lg font-semibold text-slate-900">
          Uludag Bolge Enerji Tahmin
        </h1>
      </div>
      <Button variant="ghost" onClick={logout} className="gap-2 text-slate-500">
        <LogOut size={16} />
        <span className="hidden sm:inline">Cikis</span>
      </Button>
    </header>
  );
}
