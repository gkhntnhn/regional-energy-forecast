import { useEffect } from "react";
import { cn } from "@/lib/utils";
import { X } from "lucide-react";
import { useToastStore, type ToastItem, type ToastType } from "./toast-store";

const typeStyles: Record<ToastType, string> = {
  success: "border-l-emerald-500 bg-emerald-50 text-emerald-900",
  error: "border-l-rose-500 bg-rose-50 text-rose-900",
  info: "border-l-slate-500 bg-slate-50 text-slate-900",
};

function ToastEntry({ item, onClose }: { item: ToastItem; onClose: () => void }) {
  useEffect(() => {
    const timer = setTimeout(onClose, 4000);
    return () => clearTimeout(timer);
  }, [onClose]);

  return (
    <div
      className={cn(
        "flex items-center gap-3 px-4 py-3 border-l-4 rounded-sm shadow-sm text-sm animate-in slide-in-from-right",
        typeStyles[item.type],
      )}
    >
      <span className="flex-1">{item.message}</span>
      <button onClick={onClose} className="text-slate-400 hover:text-slate-600 cursor-pointer">
        <X size={14} />
      </button>
    </div>
  );
}

export function ToastContainer() {
  const toasts = useToastStore((s) => s.toasts);
  const remove = useToastStore((s) => s.remove);

  if (toasts.length === 0) return null;

  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-2 w-80">
      {toasts.map((t) => (
        <ToastEntry key={t.id} item={t} onClose={() => remove(t.id)} />
      ))}
    </div>
  );
}
