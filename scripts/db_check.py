# coding: utf-8
import os, sys, time, psycopg2
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(__file__))
LOG  = os.path.join(ROOT, "system-logs", "db_check.py.log")

def log(s):
    print(s, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f: f.write(s + "\n")

DSN = os.environ.get("DB_DSN", "postgresql://radar_user:essi@localhost/iwr6843_db")

def main():
    try:
        with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
            cur.execute("SELECT NOW()")
            ts = cur.fetchone()[0]
        log(f"[DB-OK] {ts}")
        return 0
    except Exception as e:
        log(f"[DB-ERR] {e}")
        # Windows service name for PostgreSQL 17 default:
        #   'postgresql-x64-17'
        svc = os.environ.get("PG_SVC", "postgresql-x64-17")
        # Try restart via PowerShell so this also works from Task Scheduler
        cmd = f'powershell -NoProfile -Command "Restart-Service -Name \"{svc}\" -Force"'
        try:
            rc = os.system(cmd)
            log(f"[DB-ACTION] Restarted service {svc} rc={rc}")
        except Exception as e2:
            log(f"[DB-ACTION-ERR] {e2}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
