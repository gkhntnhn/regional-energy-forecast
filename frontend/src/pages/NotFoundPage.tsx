import { Link } from "react-router-dom";
import { Button } from "@/components/ui/Button";

export function NotFoundPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50">
      <div className="text-center">
        <h1 className="text-6xl font-mono font-bold text-slate-300">404</h1>
        <p className="mt-2 text-sm text-slate-500">Sayfa bulunamadi</p>
        <Link to="/" className="mt-4 inline-block">
          <Button variant="secondary">Ana Sayfaya Don</Button>
        </Link>
      </div>
    </div>
  );
}
