import { Card, CardContent, CardHeader } from "@/components/ui/Card";

export function AdminPage() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-slate-900">Admin Panel</h2>
        <p className="text-sm text-slate-500">Model performansi ve sistem durumu</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card accent>
          <CardHeader>
            <h3 className="text-sm font-semibold text-slate-900">MAPE Performansi</h3>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-slate-500">
              Faz 3'te MAPE chart'lari burada olacak.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <h3 className="text-sm font-semibold text-slate-900">Model Karsilastirma</h3>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-slate-500">
              Faz 3'te model comparison burada olacak.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <h3 className="text-sm font-semibold text-slate-900">Drift Durumu</h3>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-slate-500">
              Faz 3'te drift status paneli burada olacak.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <h3 className="text-sm font-semibold text-slate-900">Sistem Sagligi</h3>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-slate-500">
              Faz 3'te health bilgisi burada olacak.
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
