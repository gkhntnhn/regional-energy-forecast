import { useState, useRef, type FormEvent, type DragEvent } from "react";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";
import { toast } from "@/components/ui/Toast";
import { createPrediction } from "@/api/predict";
import { useJobStore } from "@/stores/jobs";
import { cn } from "@/lib/utils";
import { Upload, FileSpreadsheet, X, Zap } from "lucide-react";

export function UploadForm() {
  const [file, setFile] = useState<File | null>(null);
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const addJob = useJobStore((s) => s.addJob);

  function handleDrop(e: DragEvent) {
    e.preventDefault();
    setDragOver(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped && isExcel(dropped.name)) {
      setFile(dropped);
    } else {
      toast("Sadece .xlsx dosyasi yuklenebilir", "error");
    }
  }

  function handleFileChange(files: FileList | null) {
    const selected = files?.[0];
    if (selected && isExcel(selected.name)) {
      setFile(selected);
    } else if (selected) {
      toast("Sadece .xlsx dosyasi yuklenebilir", "error");
    }
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    try {
      const job = await createPrediction(file, email);
      addJob(job);
      toast("Tahmin islemi baslatildi", "success");
      setFile(null);
      setEmail("");
      if (fileRef.current) fileRef.current.value = "";
    } catch (err) {
      toast(
        err instanceof Error ? err.message : "Tahmin olusturulamadi",
        "error",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <Card accent>
      <CardHeader>
        <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
          <Zap size={16} className="text-emerald-500" />
          Tahmin Olustur
        </h3>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Drop zone */}
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileRef.current?.click()}
            className={cn(
              "border-2 border-dashed rounded-sm p-6 text-center cursor-pointer transition-colors duration-150",
              dragOver
                ? "border-emerald-500 bg-emerald-50"
                : file
                  ? "border-emerald-300 bg-emerald-50/50"
                  : "border-slate-200 hover:border-slate-300",
            )}
          >
            <input
              ref={fileRef}
              type="file"
              accept=".xlsx,.xls"
              onChange={(e) => handleFileChange(e.target.files)}
              className="hidden"
            />

            {file ? (
              <div className="flex items-center justify-center gap-3">
                <FileSpreadsheet size={20} className="text-emerald-600" />
                <span className="text-sm font-medium text-slate-700">
                  {file.name}
                </span>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    setFile(null);
                    if (fileRef.current) fileRef.current.value = "";
                  }}
                  className="text-slate-400 hover:text-slate-600 cursor-pointer"
                >
                  <X size={16} />
                </button>
              </div>
            ) : (
              <div className="space-y-1">
                <Upload size={24} className="mx-auto text-slate-400" />
                <p className="text-sm text-slate-500">
                  Excel dosyasini surukle veya <span className="text-emerald-600 font-medium">sec</span>
                </p>
                <p className="text-xs text-slate-400">.xlsx formati</p>
              </div>
            )}
          </div>

          {/* Email */}
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">
              E-posta <span className="text-slate-400 font-normal">(opsiyonel)</span>
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="sonuc@firma.com"
              className="w-full px-3 py-2 text-sm border border-slate-200 rounded-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500"
            />
          </div>

          {/* Submit */}
          <Button type="submit" loading={loading} disabled={!file} className="w-full">
            Tahmin Uret
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}

function isExcel(name: string): boolean {
  return /\.xlsx?$/i.test(name);
}
