import { cn } from "@/lib/utils";
import type { ButtonHTMLAttributes } from "react";

type Variant = "primary" | "secondary" | "danger" | "ghost";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  loading?: boolean;
}

const variants: Record<Variant, string> = {
  primary:
    "bg-emerald-600 text-white hover:bg-emerald-700 active:scale-[0.98]",
  secondary:
    "bg-slate-100 text-slate-700 hover:bg-slate-200 active:scale-[0.98]",
  danger:
    "bg-rose-500 text-white hover:bg-rose-600 active:scale-[0.98]",
  ghost:
    "text-slate-600 hover:bg-slate-100 active:scale-[0.98]",
};

export function Button({
  variant = "primary",
  loading,
  className,
  disabled,
  children,
  ...props
}: ButtonProps) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-2 px-4 py-2 text-sm font-medium rounded-sm transition-colors duration-150 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer",
        variants[variant],
        className,
      )}
      disabled={disabled || loading}
      {...props}
    >
      {loading && (
        <span className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
      )}
      {children}
    </button>
  );
}
