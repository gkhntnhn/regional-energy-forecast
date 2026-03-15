import { useAuthStore } from "@/stores/auth";
import { useNavigate } from "react-router-dom";
import { useEffect } from "react";

export function useAuthGuard() {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const navigate = useNavigate();

  useEffect(() => {
    if (!isAuthenticated) {
      navigate("/login", { replace: true });
    }
  }, [isAuthenticated, navigate]);

  return isAuthenticated;
}
