# coding: utf-8
import os, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(__file__))
LOG  = os.path.join(ROOT, "system-logs", "service_watchdog.py.log")

def log(s):
    print(s, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f: f.write(s + "\n")

URL = os.environ.get("APP_HEALTH_URL", "http://127.0.0.1:8000/health")
SERVICES = [s.strip() for s in os.environ.get("APP_SERVICES", "ISK Main,ISK App").split(",") if s.strip()]

def healthy():
    try:
        with urllib.request.urlopen(URL, timeout=4) as r:
            return r.status == 200
    except Exception:
        return False

def restart_services():
    for s in SERVICES:
        try:
            os.system(f'powershell -NoProfile -Command "nssm restart \\"{s}\\" | Out-Null"')
            log(f"[WD-RESTART] {s}")
        except Exception as e:
            log(f"[WD-ERR] {s} :: {e}")

def main():
    # 3 probes over ~30s before restart
    ok = any(healthy() for _ in range(3))
    if ok:
        log("[WD] healthy")
        return 0
    else:
        log("[WD] unhealthy -> restarting services")
        restart_services()
        return 1

if __name__ == "__main__":
    sys.exit(main())

