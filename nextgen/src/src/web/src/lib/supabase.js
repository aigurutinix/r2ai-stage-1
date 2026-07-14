import { createClient } from "@supabase/supabase-js";

// Khởi tạo Supabase client MỘT LẦN, lấy URL + anon key từ backend /config
// (không hardcode key trong source — key nằm ở biến môi trường của server).
let _client = null;
let _initPromise = null;

export function getSupabase() {
  if (_client) return Promise.resolve(_client);
  if (_initPromise) return _initPromise;

  _initPromise = (async () => {
    try {
      const res = await fetch("/config");
      const cfg = await res.json();
      if (!cfg.supabaseUrl || !cfg.supabaseAnonKey) {
        console.warn("[supabase] Thiếu SUPABASE_URL / SUPABASE_ANON_KEY — tắt tính năng đăng nhập/lịch sử.");
        return null;
      }
      _client = createClient(cfg.supabaseUrl, cfg.supabaseAnonKey, {
        auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: true },
      });
      return _client;
    } catch (e) {
      console.warn("[supabase] Không lấy được /config:", e);
      return null;
    }
  })();

  return _initPromise;
}
