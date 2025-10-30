# coding: utf-8
"""
cleanup_orphan_snapshots.py
Delete orphaned snapshot/clip files not referenced in the DB and prune empty dirs.

Defaults & ENV:
  DB_DSN                      (e.g. postgresql://user:pass@127.0.0.1:5432/iwr6843_db)
  PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE  (alternative to DB_DSN)
  ORPHAN_ROOTS                (semicolon-separated paths; overrides defaults)
  ORPHAN_RETENTION_DAYS       (int, default 14)
  ORPHAN_LOG_FILE             (full path; default <project>/system-logs/orphan_cleanup.log)

CLI:
  --roots <one or more>       (optional if ORPHAN_ROOTS or defaults exist)
  --older-than-days N         (default from env or 14)
  --dry-run                   (log only; do not delete)
"""
import argparse, os, sys, time, pathlib, shutil
import psycopg2, psycopg2.extras

SCRIPT_PATH   = pathlib.Path(__file__).resolve()
PROJECT_ROOT  = SCRIPT_PATH.parents[1]
DEFAULT_ROOTS = [str(PROJECT_ROOT / "snapshots"), str(PROJECT_ROOT / "backups")]
DEFAULT_LOG   = str(PROJECT_ROOT / "system-logs" / "orphan_cleanup.log")
LOG_PATH      = os.environ.get("ORPHAN_LOG_FILE", DEFAULT_LOG)

def _ensure_log_dir():
    lp = pathlib.Path(LOG_PATH); lp.parent.mkdir(parents=True, exist_ok=True)

def log(line: str):
    print(line, flush=True)
    if LOG_PATH:
        _ensure_log_dir()
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")

def _dsn_from_env() -> str:
    dsn = os.environ.get("DB_DSN")
    if dsn: return dsn
    host = os.environ.get("PGHOST", "127.0.0.1")
    port = os.environ.get("PGPORT", "5432")
    user = os.environ.get("PGUSER", "radar_user")
    pwd  = os.environ.get("PGPASSWORD", "essi")
    db   = os.environ.get("PGDATABASE", "iwr6843_db")
    return f"postgresql://{user}:{pwd}@{host}:{port}/{db}"

def _connect():
    return psycopg2.connect(_dsn_from_env())

def _load_db_paths() -> set[str]:
    sql = """
      SELECT snapshot_path, clip_path
      FROM radar_data
      WHERE snapshot_path IS NOT NULL OR clip_path IS NOT NULL
    """
    keep: set[str] = set()
    with _connect() as conn, conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(sql)
        for r in cur.fetchall():
            for col in ("snapshot_path", "clip_path"):
                p = r[col]
                if p:
                    keep.add(os.path.normcase(os.path.normpath(os.path.abspath(p))))
    return keep

def is_older_than(path: str, days: int | None) -> bool:
    if days is None: return True
    cutoff = time.time() - (days * 86400)
    try:
        return os.path.getmtime(path) < cutoff
    except OSError:
        return True  # treat unreadable as old

def normalize(p: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(p)))

def parse_args():
    env_roots = os.environ.get("ORPHAN_ROOTS")
    env_roots_list = [r.strip() for r in env_roots.split(";")] if env_roots else None
    ap = argparse.ArgumentParser(description="Delete orphaned snapshot/clip files.")
    ap.add_argument("--roots", nargs="+", default=env_roots_list or DEFAULT_ROOTS,
                    help="Folders to scan. Default from ORPHAN_ROOTS or snapshots+backups.")
    ap.add_argument("--older-than-days", type=int,
                    default=int(os.environ.get("ORPHAN_RETENTION_DAYS", "14")),
                    help="Only delete if older than N days (default 14 or env).")
    ap.add_argument("--dry-run", action="store_true", help="Log actions, do not delete.")
    return ap.parse_args()

def main() -> int:
    args  = parse_args()
    roots = [normalize(os.path.expandvars(os.path.expanduser(r))) for r in args.roots]
    roots = [r for r in roots if os.path.isdir(r)]
    if not roots:
        log("[ORPHAN] no valid roots to scan; nothing to do"); return 0
    try:
        keep = _load_db_paths()
    except Exception as e:
        log(f"[ERROR] database connection/lookup failed: {e}"); return 1
    log(f"[ORPHAN] loaded {len(keep)} DB paths; roots={roots}; cutoff_days={args.older_than_days}; dry_run={args.dry_run}")
    deleted = pruned = 0
    for root in roots:
        for dirpath, _, files in os.walk(root):
            for fn in files:
                fp = normalize(os.path.join(dirpath, fn))
                if fp not in keep and is_older_than(fp, args.older_than_days):
                    log(f"[DEL] {fp}")
                    if not args.dry_run:
                        try: os.remove(fp); deleted += 1
                        except Exception as e: log(f"[DEL-ERR] {fp} :: {e}")
        for dirpath, _, _ in os.walk(root, topdown=False):
            try:
                if not os.listdir(dirpath):
                    if args.dry_run: log(f"[RMDIR] {dirpath} (dry-run)")
                    else: shutil.rmtree(dirpath, ignore_errors=True); log(f"[RMDIR] {dirpath}"); pruned += 1
            except Exception as e:
                log(f"[RMDIR-ERR] {dirpath} :: {e}")
    log(f"[DONE] deleted_files={deleted} pruned_dirs={pruned} dry_run={args.dry_run}")
    return 0

if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    try: sys.exit(main())
    except KeyboardInterrupt:
        log("[ABORT] interrupted by user"); sys.exit(130)
