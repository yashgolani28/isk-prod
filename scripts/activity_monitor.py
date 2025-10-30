#!/usr/bin/env python3
import os, time, subprocess, psycopg2
from pathlib import Path

# ----- PATHS (portable) -----
ROOT    = str(Path(__file__).resolve().parents[1])
STATIC  = os.path.join(ROOT, "static")
HEATMAP = os.path.join(STATIC, "heatmap_3d.png")
SCATTER = os.path.join(STATIC, "scatter_3d.png")

# ----- ENV / TUNABLES -----
DB_STALE_MIN  = float(os.getenv("DB_STALE_MIN", "10"))
VIZ_STALE_MIN = float(os.getenv("VIZ_STALE_MIN", "10"))
SERVICES      = [s.strip() for s in os.getenv("SERVICES", "ISK Main,ISK App").split(",") if s.strip()]
USE_SUDO      = os.getenv("USE_SUDO", "0").lower() in ("1", "true", "yes")
ANY           = os.getenv("ANY_STALE","0").lower() in ("1","true","yes")
FORCE         = os.getenv("FORCE_RESTART","0").lower() in ("1","true","yes")

DSN = os.getenv("DB_DSN")  # e.g. postgresql://radar_user:essi@localhost/iwr6843_db

def db_conn():
    kw = dict(application_name="iwr6843isk-activity-monitor")
    if DSN:
        return psycopg2.connect(DSN, **kw)
    return psycopg2.connect(dbname="iwr6843_db", user="radar_user", host="localhost", **kw)

def age_minutes(path: str) -> float:
    try:
        return (time.time() - os.path.getmtime(path)) / 60.0
    except Exception:
        return 1e9

def restart_service(name: str):
    if os.name == "nt":
        # Windows (NSSM)
        try: subprocess.run(["nssm", "restart", name], check=False)
        except Exception as e: print(f"[ACTIVITY] restart error {name}: {e}")
    else:
        cmd = (["sudo","-n"] if USE_SUDO else []) + ["systemctl","restart", name]
        try: subprocess.run(cmd, check=False)
        except Exception as e: print(f"[ACTIVITY] restart error {name}: {e}")

with db_conn() as conn, conn.cursor() as cur:
    cur.execute("SELECT EXTRACT(EPOCH FROM (NOW() - MAX(measured_at))) / 60.0 FROM radar_data;")
    row        = cur.fetchone()
    db_gap_min = row[0] if row and row[0] is not None else 1e9

heat_age = min(age_minutes(HEATMAP), age_minutes(SCATTER))

cond_both = (db_gap_min > DB_STALE_MIN and heat_age > VIZ_STALE_MIN)
cond_any  = (db_gap_min > DB_STALE_MIN or  heat_age > VIZ_STALE_MIN)

if FORCE or (cond_any if ANY else cond_both):
    for svc in SERVICES:
        restart_service(svc)
    print(f"[ACTIVITY] Restarted services: db_gap={db_gap_min:.1f}m heat_age={heat_age:.1f}m")
else:
    print(f"[ACTIVITY] OK db_gap_min={db_gap_min:.1f} heat_age_min={heat_age:.1f}")
