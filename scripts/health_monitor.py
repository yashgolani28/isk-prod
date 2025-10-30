#!/usr/bin/env python3
import os, shutil, subprocess, psutil, psycopg2
from pathlib import Path

SERVICES = [s.strip() for s in os.getenv("SERVICES", "ISK Main,ISK App").split(",") if s.strip()]
DSN = os.getenv("DB_DSN")

def db_ping() -> bool:
    try:
        kw = dict(connect_timeout=2, application_name="iwr6843isk-health")
        conn = psycopg2.connect(DSN, **kw) if DSN else psycopg2.connect(dbname="iwr6843_db", user="radar_user", host="localhost", **kw)
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
        return True
    except Exception as e:
        print("[HEALTH] WARN: DB unreachable:", e); return False

def is_service_running(name: str) -> bool:
    if os.name == "nt":
        try:
            out = subprocess.check_output(["sc","query", name], text=True, stderr=subprocess.STDOUT)
            return "RUNNING" in out
        except Exception:
            return False
    else:
        return subprocess.call(["systemctl","is-active","--quiet", name]) == 0

def main():
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    disk= shutil.disk_usage(Path(ROOT := Path(__file__).resolve().parents[1]).drive if os.name=="nt" else "/")
    try:
        temp = psutil.sensors_temperatures().get("cpu-thermal",[type("T",(),{"current":None})()])[0].current
    except Exception:
        temp = None

    used_gb  = round(disk.used  / (1024**3))
    total_gb = round(disk.total / (1024**3))
    if temp is None:
        print(f"CPU {cpu:.1f}% | Temp N/A | Mem {mem.percent:.1f}% | Disk {used_gb}/{total_gb}GB")
    else:
        print(f"CPU {cpu:.1f}% | Temp {float(temp):.1f}C | Mem {mem.percent:.1f}% | Disk {used_gb}/{total_gb}GB")

    if cpu >= 70 or mem.percent >= 90 or (disk.used / disk.total) >= 0.90:
        print("[HEALTH] WARN: threshold exceeded")

    for svc in SERVICES:
        if not is_service_running(svc):
            print("[HEALTH] WARN:", svc, "inactive")

    db_ping()

if __name__ == "__main__":
    main()
