import { Card, CardContent, CardHeader } from "@/components/ui/Card";

export function HistoryPage() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-slate-900">Gecmis Tahminler</h2>
        <p className="text-sm text-slate-500">Onceki tahmin sonuclari</p>
      </div>

      <Card>
        <CardHeader>
          <h3 className="text-sm font-semibold text-slate-900">Tahmin Gecmisi</h3>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-slate-500">
            Faz 3'te gecmis tahminler tablosu burada olacak.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
