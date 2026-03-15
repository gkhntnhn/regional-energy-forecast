import { Outlet } from "react-router-dom";
import { Sidebar } from "./Sidebar";
import { Header } from "./Header";
import { useAuthGuard } from "@/hooks/useAuth";
import { ToastContainer } from "@/components/ui/Toast";

export function Layout() {
  const isAuth = useAuthGuard();
  if (!isAuth) return null;

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0">
        <Header />
        <main className="flex-1 p-6">
          <Outlet />
        </main>
      </div>
      <ToastContainer />
    </div>
  );
}
