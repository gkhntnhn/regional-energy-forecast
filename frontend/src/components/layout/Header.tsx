import { useAuthStore } from "@/stores/auth";
import { Button } from "@/components/ui/Button";
import { LogOut } from "lucide-react";

export function Header() {
  const logout = useAuthStore((s) => s.logout);

  return (
    <header className="flex items-center justify-between h-14 px-6 bg-white border-b border-slate-200">
      <h1 className="text-lg font-semibold text-slate-900">
        Uludag Bolge Enerji Tahmin
      </h1>
      <Button variant="ghost" onClick={logout} className="gap-2 text-slate-500">
        <LogOut size={16} />
        <span className="hidden sm:inline">Cikis</span>
      </Button>
    </header>
  );
}
