import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { Zap } from "lucide-react";

export function DashboardPage() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-slate-900">Dashboard</h2>
        <p className="text-sm text-slate-500">Tahmin olustur ve sonuclari goruntule</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card accent className="lg:col-span-2">
          <CardHeader>
            <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
              <Zap size={16} className="text-emerald-500" />
              Tahmin Olustur
            </h3>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-slate-500">
              Faz 2'de upload formu burada olacak.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <h3 className="text-sm font-semibold text-slate-900">Aktif Isler</h3>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-slate-500">
              Faz 2'de job queue burada olacak.
            </p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <h3 className="text-sm font-semibold text-slate-900">Son Tahmin</h3>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-slate-500">
            Faz 2'de sonuc tablosu ve grafik burada olacak.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
