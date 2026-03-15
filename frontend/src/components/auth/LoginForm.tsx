import { useState, type FormEvent } from "react";
import { useAuthStore } from "@/stores/auth";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/Button";
import { api } from "@/api/client";
import { Zap, Eye, EyeOff } from "lucide-react";
import { cn } from "@/lib/utils";

export function LoginForm() {
  const [token, setToken] = useState("");
  const [remember, setRemember] = useState(true);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [showToken, setShowToken] = useState(false);
  const [shake, setShake] = useState(false);

  const login = useAuthStore((s) => s.login);
  const navigate = useNavigate();

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!token.trim()) return;

    setLoading(true);
    setError("");

    try {
      // Temporarily set token to validate it via /models
      useAuthStore.getState().login(token.trim(), false);
      await api.get("/models");
      login(token.trim(), remember);
      navigate("/", { replace: true });
    } catch {
      useAuthStore.getState().logout();
      setError("Gecersiz API anahtari");
      setShake(true);
      setTimeout(() => setShake(false), 600);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50 p-4">
      <div
        className={cn(
          "w-full max-w-sm bg-white rounded-sm shadow-sm p-8",
          shake && "animate-shake",
        )}
      >
        <div className="flex items-center gap-3 mb-8">
          <div className="w-10 h-10 bg-emerald-600 rounded-sm flex items-center justify-center">
            <Zap size={20} className="text-white" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-slate-900">Energy Forecast</h1>
            <p className="text-xs text-slate-500">Uludag Bolge Tahmin Sistemi</p>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1.5">
              API Anahtari
            </label>
            <div className="relative">
              <input
                type={showToken ? "text" : "password"}
                value={token}
                onChange={(e) => setToken(e.target.value)}
                placeholder="Bearer token giriniz"
                className="w-full px-3 py-2 pr-10 text-sm border border-slate-200 rounded-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 font-mono"
                autoFocus
              />
              <button
                type="button"
                onClick={() => setShowToken(!showToken)}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 cursor-pointer"
              >
                {showToken ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>

          <label className="flex items-center gap-2 text-sm text-slate-600 cursor-pointer">
            <input
              type="checkbox"
              checked={remember}
              onChange={(e) => setRemember(e.target.checked)}
              className="rounded-sm border-slate-300 text-emerald-600 focus:ring-emerald-500"
            />
            Oturumu hatirla
          </label>

          {error && (
            <p className="text-sm text-rose-500 font-medium">{error}</p>
          )}

          <Button type="submit" loading={loading} className="w-full">
            Giris Yap
          </Button>
        </form>
      </div>
    </div>
  );
}
