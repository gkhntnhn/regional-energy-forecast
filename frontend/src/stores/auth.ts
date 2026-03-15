import { create } from "zustand";
import { AUTH_STORAGE_KEY } from "@/lib/constants";

interface AuthState {
  token: string | null;
  isAuthenticated: boolean;
  login: (token: string, remember: boolean) => void;
  logout: () => void;
}

const stored = sessionStorage.getItem(AUTH_STORAGE_KEY);

export const useAuthStore = create<AuthState>((set) => ({
  token: stored,
  isAuthenticated: !!stored,

  login: (token: string, remember: boolean) => {
    if (remember) {
      sessionStorage.setItem(AUTH_STORAGE_KEY, token);
    }
    set({ token, isAuthenticated: true });
  },

  logout: () => {
    sessionStorage.removeItem(AUTH_STORAGE_KEY);
    set({ token: null, isAuthenticated: false });
  },
}));
