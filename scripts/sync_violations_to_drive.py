#!/usr/bin/env python3
import os, sys, json, subprocess, zipfile, re
from datetime import datetime
import psycopg2, psycopg2.extras
from pathlib import Path

ROOT    = str(Path(__file__).resolve().parents[1])
SNAPS   = os.path.join(ROOT,"snapshots")
CLIPS   = os.path.join(ROOT,"clips")
EXPORTS = os.path.join(ROOT,"backups","exports")
VREPORT = os.path.join(ROOT,"backups","reports","violations")
LOGDIR  = os.path.join(ROOT,"system-logs")
for d in (EXPORTS,VREPORT,LOGDIR): os.makedirs(d, exist_ok=True)

REMOTE      = os.getenv("RCLONE_REMOTE","gdrive")
CLOUD_BASE  = os.getenv("CLOUD_BASE","ESSI/IWR6843ISK")
MARGIN_KMH  = float(os.getenv("VIOLATION_MARGIN_KMH","0"))
INCLUDE_MAN = os.getenv("INCLUDE_MANUAL","0").lower() in ("1","true","yes")
HOURS_WIN   = int(os.getenv("SYNC_WINDOW_HOURS","24"))
FORCE_DATE  = os.getenv("FORCE_DATE","").strip()
REUPLOAD    = os.getenv("REUPLOAD","0").lower() in ("1","true","yes")
MIN_MP4_BYTES = int(os.getenv("MIN_MP4_BYTES", str(100*1024)))
DSN = os.getenv("DB_DSN")

if ROOT not in sys.path: sys.path.insert(0, ROOT)
from report import generate_pdf_report

def db():
    kw = dict(application_name="iwr6843isk-syncdrive")
    dsn = os.getenv("DB_DSN")
    if dsn:
        return psycopg2.connect(dsn, **kw)
    return psycopg2.connect(dbname="iwr6843_db", user="radar_user", host="localhost", **kw)

def load_cfg():
    try: return json.load(open(os.path.join(ROOT,"config.json")))
    except Exception: return {}

def limits_from(cfg):
    raw = (cfg.get("dynamic_speed_limits") or cfg.get("speed_limits") or {})
    out = {"DEFAULT": 50.0}
    for k,v in raw.items():
        try: out[str(k).upper()] = float(v)
        except: pass
    if "default" in raw and "DEFAULT" not in out:
        try: out["DEFAULT"] = float(raw["default"])
        except: pass
    return out

def fetch_rows():
    with db() as conn, conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        if FORCE_DATE:
            cur.execute("""
                SELECT measured_at, sensor, object_id, type, confidence, speed_kmh, velocity, distance,
                       direction, motion_state, signal_level, doppler_frequency,
                       snapshot_path, snapshot_type
                FROM radar_data
                WHERE measured_at::date = to_date(%s,'YYYYMMDD')
                  AND snapshot_path IS NOT NULL AND snapshot_path <> ''
                ORDER BY measured_at ASC
            """, (FORCE_DATE,))
        else:
            cur.execute(f"""
                SELECT measured_at, sensor, object_id, type, confidence, speed_kmh, velocity, distance,
                       direction, motion_state, signal_level, doppler_frequency,
                       snapshot_path, snapshot_type
                FROM radar_data
                WHERE measured_at >= NOW() - INTERVAL '{HOURS_WIN} hours'
                  AND snapshot_path IS NOT NULL AND snapshot_path <> ''
                ORDER BY measured_at ASC
            """)
        rows = [dict(r) for r in cur.fetchall()]
        for r in rows: r["datetime"] = r["measured_at"]
        return rows

def is_violation(row, limits):
    try: spd = float(row.get("speed_kmh") or 0)
    except: spd = 0.0
    typ = str(row.get("type","UNKNOWN")).upper()
    limit = limits.get(typ, limits["DEFAULT"]) + MARGIN_KMH
    snap_type = (row.get("snapshot_type") or "").lower()
    auto_ok = (snap_type == "auto") or INCLUDE_MAN
    return auto_ok and spd > limit

_rx_ts = re.compile(r"^violation_(\d{8})_(\d{6})_")
def _parse_ts_from_name(name):
    m = _rx_ts.match(name)
    if not m: return None
    ymd, hms = m.groups()
    try: return datetime.strptime(f"{ymd}_{hms}", "%Y%m%d_%H%M%S")
    except Exception: return None

def _good_mp4(path):
    try: return os.path.getsize(path) >= MIN_MP4_BYTES
    except Exception: return False

def find_clip_paths(row):
    obj = str(row.get("object_id") or "")
    typ = str(row.get("type") or "").upper()
    dt  = row.get("datetime")
    if dt is None: return (None, None)
    dt_naive = dt.replace(tzinfo=None) if getattr(dt,"tzinfo",None) else dt
    files = []
    try: files = os.listdir(CLIPS)
    except Exception: files = []
    cand_mp4 = [f for f in files if f.endswith(".mp4") and obj and obj in f]
    cand_json= [f for f in files if f.endswith(".json") and obj and obj in f]
    cand_mp4.sort(); cand_json.sort()
    mp4 = os.path.join(CLIPS, cand_mp4[0]) if cand_mp4 else None
    js  = os.path.join(CLIPS, cand_json[0]) if cand_json else None
    if mp4 and not _good_mp4(mp4): mp4 = None
    if mp4: return (mp4, js)
    best = (None, 999999)
    for f in files:
        if not f.endswith(".mp4"): continue
        ts = _parse_ts_from_name(f)
        if not ts:
            continue
        delta = abs((ts - dt_naive).total_seconds()); 
        if delta <= 300:
            score = delta + (0 if (typ and typ in f) else 30)
            if score < best[1]: best = (f, score)
    if best[0]:
        mp4 = os.path.join(CLIPS, best[0])
        if not _good_mp4(mp4): mp4 = None
    if mp4:
        stem = os.path.splitext(os.path.basename(mp4))[0]; js_name = stem + ".json"
        if js_name in files: js = os.path.join(CLIPS, js_name)
    return (mp4, js)

def make_violation_pdf(row):
    date_folder = (FORCE_DATE if FORCE_DATE else datetime.now().strftime("%Y%m%d"))
    ddir = os.path.join(VREPORT, date_folder); os.makedirs(ddir, exist_ok=True)
    base = f"violation_{row['object_id']}_{row['datetime']:%Y%m%d_%H%M%S}.pdf"
    outp = os.path.join(ddir, base)
    generate_pdf_report(outp, title="Violation Report", data=[row], filters={"Mode":"Violations only"})
    return outp

def zip_violation(row, clip_mp4, clip_json, pdf_path):
    zdir = os.path.join(EXPORTS, f"{row['datetime']:%Y%m%d}"); os.makedirs(zdir, exist_ok=True)
    base = f"violation_{row['object_id']}_{row['datetime']:%Y%m%d_%H%M%S}.zip"
    zp = os.path.join(zdir, base)
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        sp = row.get("snapshot_path")
        if sp:
            if not os.path.isabs(sp): sp = os.path.join(SNAPS, os.path.basename(sp))
            if os.path.exists(sp): z.write(sp, os.path.basename(sp))
        if clip_mp4 and os.path.exists(clip_mp4): z.write(clip_mp4, os.path.basename(clip_mp4))
        else: z.writestr("NO_VIDEO.txt","No matching MP4 found (id/time window).")
        if clip_json and os.path.exists(clip_json): z.write(clip_json, os.path.basename(clip_json))
        if os.path.exists(pdf_path): z.write(pdf_path, os.path.basename(pdf_path))
    return zp

def uploaded_mark(zip_path): return zip_path + ".uploaded"
def already_uploaded(zip_path): return os.path.exists(uploaded_mark(zip_path))
def mark_uploaded(zip_path): open(uploaded_mark(zip_path),"w").write(datetime.now().isoformat())

def rclone_copy(local_path, cloud_path):
    return subprocess.call(["rclone","copy","--checksum","--fast-list", local_path, cloud_path])

def main():
    limits = limits_from(load_cfg())
    rows   = fetch_rows()
    viols  = [r for r in rows if is_violation(r, limits)]
    if not viols: print("[SYNC] No violations in window."); return
    for r in viols:
        mp4, js = find_clip_paths(r)
        pdf     = make_violation_pdf(r)
        zip_p   = zip_violation(r, mp4, js, pdf)
        if not REUPLOAD and already_uploaded(zip_p):
            print("[SYNC] Already uploaded:", os.path.basename(zip_p)); continue
        ymd = (f"{r['datetime']:%Y}", f"{r['datetime']:%m}", f"{r['datetime']:%d}")
        cloud_dir = f"{REMOTE}:{CLOUD_BASE}/violations/{ymd[0]}/{ymd[1]}/{ymd[2]}/"
        rc = rclone_copy(zip_p, cloud_dir)
        if rc == 0: mark_uploaded(zip_p); print("[SYNC] Uploaded:", os.path.basename(zip_p), "→", cloud_dir)
        else: print("[SYNC-ERR] rclone rc=", rc, "for", zip_p)

if __name__ == "__main__":
    main()
