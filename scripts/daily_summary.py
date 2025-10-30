#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, math, json, datetime as dt
from pathlib import Path
import psycopg2, psycopg2.extras

# ── paths ──────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1] if HERE.parent.name.lower() == "scripts" else HERE.parent
OUT_DIR = ROOT / "backups" / "reports"
LOG_PATH = ROOT / "system-logs" / "daily_summary.log"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# make project importable
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_utils import load_config
from report import generate_pdf_report  # uses your themed ReportLab builder

# ── logging ────────────────────────────────────────────────────────────────
def log(msg: str):
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} [DAILY] {msg}"
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        # best-effort: still print
        pass
    print(line, flush=True)

# ── db helpers ─────────────────────────────────────────────────────────────
def _dsn():
    # Prefer full DSN from env; fall back to local defaults
    dsn = os.getenv("DB_DSN")
    if dsn and dsn.strip():
        return dsn.strip()
    # compatible with local installs without password env
    dsn = "dbname=iwr6843_db user=radar_user host=localhost"
    if os.getenv("PGPASSWORD"):
        dsn += f" password={os.getenv('PGPASSWORD')}"
    return dsn

def _connect():
    return psycopg2.connect(dsn=_dsn())

# ── data pull ──────────────────────────────────────────────────────────────
def fetch_violations_rows(hours=24):
    """
    Pull last `hours` of speeding rows for the report. We assume the app writes
    motion_state like 'speeding...' when a violation is recorded.
    """
    since_sql = f"NOW() - INTERVAL '{int(hours)} hours'"
    q = """
        SELECT
            measured_at,                  -- timestamptz (UTC in DB)
            datetime,                     -- legacy/nullable
            sensor,
            object_id,
            type,
            confidence,
            speed_kmh,
            velocity,
            distance,
            direction,
            motion_state,
            snapshot_path,
            snapshot_type,
            azimuth,
            elevation,
            clip_path,
            clip_status,
            clip_duration_s,
            clip_fps,
            clip_size_bytes
        FROM radar_data
        WHERE measured_at >= {since}
          AND COALESCE(motion_state,'') ILIKE 'speeding%%'
        ORDER BY measured_at ASC
    """.format(since=since_sql)

    with _connect() as conn, conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(q)
        rows = [dict(r) for r in cur.fetchall()]

    # normalize for report: use 'datetime' field (IST string expected by report’s helpers)
    for r in rows:
        r["datetime"] = r.get("measured_at") or r.get("datetime")
    return rows

# ── summary + simple charts ────────────────────────────────────────────────
def _safe_f(x, default=0.0):
    try: return float(x)
    except Exception: return float(default)

def build_summary(rows, cfg):
    total = len(rows)
    speeds = [_safe_f(r.get("speed_kmh")) for r in rows if r.get("speed_kmh") is not None]
    avg = round(sum(speeds)/len(speeds), 2) if speeds else 0.0
    top = round(max(speeds), 2) if speeds else 0.0
    low = round(min(speeds), 2) if speeds else 0.0
    # counts by coarse direction (approaching vs departing vs stationary/unknown)
    def norm_dir(v):
        v = (v or "").lower()
        if "approach" in v: return "approaching"
        if "depart"  in v: return "departing"
        if "station" in v: return "stationary"
        return "unknown"
    dirs = [norm_dir(r.get("direction")) for r in rows]
    from collections import Counter
    C = Counter(dirs)
    return {
        "total_records": total,
        "avg_speed": avg,
        "top_speed": top,
        "lowest_speed": low,
        "auto_snapshots": sum(1 for r in rows if (r.get("snapshot_type") or "").lower() == "auto"),
        "manual_snapshots": sum(1 for r in rows if (r.get("snapshot_type") or "").lower() == "manual"),
        "approaching_count": C.get("approaching", 0),
        "departing_count": C.get("departing", 0),
        "stationary_count": C.get("stationary", 0),
        "right_count": 0, "left_count": 0, "unknown_count": C.get("unknown", 0),
        # Optional limits (report has defaults if missing)
        "speed_limits": (cfg or {}).get("speed_limits", {})
    }

def build_charts(rows):
    # 1) counts by type
    from collections import Counter, defaultdict
    typ = [str((r.get("type") or "")).upper() for r in rows if r.get("type")]
    type_counts = Counter(typ)
    # 2) hourly histogram (local time label; backend will just render bars)
    by_hour = defaultdict(int)
    for r in rows:
        t = r.get("measured_at")
        try:
            hour = (t.hour if hasattr(t, "hour") else dt.datetime.fromisoformat(str(t)).hour)
        except Exception:
            continue
        by_hour[f"{hour:02}:00"] += 1
    # 3) speed buckets
    buckets = {"0–10":0, "10–20":0, "20–30":0, "30–40":0, "40–50":0, "50–60":0, "60+":0}
    for r in rows:
        s = _safe_f(r.get("speed_kmh"))
        if   s < 10: buckets["0–10"] += 1
        elif s < 20: buckets["10–20"] += 1
        elif s < 30: buckets["20–30"] += 1
        elif s < 40: buckets["30–40"] += 1
        elif s < 50: buckets["40–50"] += 1
        elif s < 60: buckets["50–60"] += 1
        else:        buckets["60+"]   += 1

    return {
        "violations_by_type": {"labels": list(type_counts.keys()), "data": list(type_counts.values())},
        "violations_by_hour": {"labels": list(by_hour.keys()),     "data": list(by_hour.values())},
        "speed_distribution": {"labels": list(buckets.keys()),     "data": list(buckets.values())},
    }

# ── main ────────────────────────────────────────────────────────────────────
def main():
    cfg = {}
    try:
        cfg = load_config()
    except Exception as e:
        log(f"config load warning: {e}")

    rows = fetch_violations_rows(hours=int(os.getenv("DAILY_SUMMARY_HOURS", "24")))
    today = dt.datetime.now().strftime("%Y%m%d")
    pdf_path = OUT_DIR / f"violations_{today}.pdf"

    try:
        charts = build_charts(rows)
        filters = {
            "Mode": "Speeding violations (last 24h)",
            "Date": dt.datetime.now().strftime("%Y-%m-%d"),
            "Records": str(len(rows)),
        }
        generate_pdf_report(
            str(pdf_path),
            title="Daily Violations Report",
            summary=build_summary(rows, cfg),
            data=rows,
            filters=filters,
            logo_path=str(ROOT / "static" / "essi_logo.jpeg"),
            charts=charts
        )
        log(f"PDF generated: {pdf_path} | count={len(rows)}")
        return 0
    except Exception as e:
        log(f"ERROR generating PDF: {e}")
        return 2

if __name__ == "__main__":
    sys.exit(main())
