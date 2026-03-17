import { create } from "zustand";
import { AUTH_STORAGE_KEY } from "@/lib/constants";

interface AuthState {
  token: string | null;
  isAuthenticated: boolean;
  login: (token: string, remember: boolean) => void;
  logout: () => void;
}

const stored =
  localStorage.getItem(AUTH_STORAGE_KEY) ??
  sessionStorage.getItem(AUTH_STORAGE_KEY);

export const useAuthStore = create<AuthState>((set) => ({
  token: stored,
  isAuthenticated: !!stored,

  login: (token: string, remember: boolean) => {
    if (remember) {
      localStorage.setItem(AUTH_STORAGE_KEY, token);
    } else {
      sessionStorage.setItem(AUTH_STORAGE_KEY, token);
    }
    set({ token, isAuthenticated: true });
  },

  logout: () => {
    localStorage.removeItem(AUTH_STORAGE_KEY);
    sessionStorage.removeItem(AUTH_STORAGE_KEY);
    set({ token: null, isAuthenticated: false });
  },
}));

// Multi-tab logout sync: other tabs detect token removal
window.addEventListener("storage", (e) => {
  if (e.key === AUTH_STORAGE_KEY && e.newValue === null) {
    useAuthStore.getState().logout();
  }
});
