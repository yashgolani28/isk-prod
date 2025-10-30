#!/usr/bin/env bash
set -euo pipefail

# --- Paths ---
ROOT="/home/pi/iwr6843isk"
BACKUPS_DIR="$ROOT/backups"
LOG_DIR="$ROOT/system-logs"
LOG="$LOG_DIR/db_backup_and_sync.log"

APP_LOG1="$ROOT/system-logs/radar.log"
APP_LOG2="$ROOT/system-logs/isk-app.log"
mkdir -p "$(dirname "$LOG")"

log_both() {
  # usage: log_both "LEVEL" "message..."
  printf "%s [%s] %s\n" "$(date '+%F %T')" "$1" "$2" \
    | tee -a "$LOG" \
    | tee -a "$APP_LOG1" \
    | tee -a "$APP_LOG2" >/dev/null
}

# --- Cloud config (override via env in crontab if you like) ---
RCLONE_CONFIG="${RCLONE_CONFIG:-/home/pi/.config/rclone/rclone.conf}"
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive}"
CLOUD_BASE="${CLOUD_BASE:-ESSI/IWR6843ISK}"

# --- DB config (auth via ~/.pgpass) ---
DBNAME="iwr6843_db"
DBUSER="radar_user"

mkdir -p "$LOG_DIR"

# ========= DB dump → Drive (no local retention) =========
TS="$(date +%Y%m%d_%H%M%S)"
YEAR="$(date +%Y)"
MONTH="$(date +%m)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

DUMP="$TMP/${DBNAME}_${TS}.sql.gz"
SUM="$TMP/${DBNAME}_${TS}.sql.gz.sha256"
DEST_DB="${RCLONE_REMOTE}:${CLOUD_BASE}/db_backups/${YEAR}/${MONTH}/"

echo "$(date '+%F %T') [INFO] Starting DB dump -> (temp) $DUMP" | tee -a "$LOG"

# Create gzip-compressed dump (no ownership/privilege statements)
if ! ionice -c3 nice -n 10 pg_dump -U "$DBUSER" -h localhost --no-owner --no-privileges "$DBNAME" | gzip -c > "$DUMP"; then
  echo "$(date '+%F %T') [ERR ] pg_dump failed" | tee -a "$LOG"
  exit 1
fi

sha256sum "$DUMP" > "$SUM"
echo "$(date '+%F %T') [INFO] Dump size: $(du -h "$DUMP" | cut -f1)" | tee -a "$LOG"

echo "$(date '+%F %T') [INFO] Upload DB -> $DEST_DB" | tee -a "$LOG"
if ! env RCLONE_CONFIG="$RCLONE_CONFIG" rclone copy "$DUMP" "$DEST_DB" \
    --checksum --fast-list --transfers 2 --checkers 4 --timeout 10m --retries 5 \
    --log-level INFO --log-file "$ROOT/system-logs/drive_sync.log"; then
  echo "$(date '+%F %T') [ERR ] rclone copy DB failed" | tee -a "$LOG"
  exit 2
fi

# Upload checksum (optional)
env RCLONE_CONFIG="$RCLONE_CONFIG" rclone copy "$SUM" "$DEST_DB" \
    --transfers 1 --checkers 2 --timeout 2m --retries 3 \
    --log-level INFO --log-file "$ROOT/system-logs/drive_sync.log" || true

echo "$(date '+%F %T') [OK  ] DB backup uploaded; no local copy kept" | tee -a "$LOG"

# ========= Copy reports & exports → Drive (locals pruned by 1-day retention cron) =========
if [ -d "$BACKUPS_DIR/reports" ]; then
  echo "$(date '+%F %T') [INFO] Upload reports -> ${RCLONE_REMOTE}:${CLOUD_BASE}/reports/" | tee -a "$LOG"
  env RCLONE_CONFIG="$RCLONE_CONFIG" rclone copy "$BACKUPS_DIR/reports" "${RCLONE_REMOTE}:${CLOUD_BASE}/reports/" \
    --checksum --fast-list --transfers 2 --checkers 4 --timeout 10m --retries 5 \
    --log-level INFO --log-file "$ROOT/system-logs/drive_sync.log" || \
    echo "$(date '+%F %T') [WARN] reports copy had issues" | tee -a "$LOG"
fi

if [ -d "$BACKUPS_DIR/exports" ]; then
  echo "$(date '+%F %T') [INFO] Upload exports -> ${RCLONE_REMOTE}:${CLOUD_BASE}/exports/" | tee -a "$LOG"
  env RCLONE_CONFIG="$RCLONE_CONFIG" rclone copy "$BACKUPS_DIR/exports" "${RCLONE_REMOTE}:${CLOUD_BASE}/exports/" \
    --checksum --fast-list --transfers 2 --checkers 4 --timeout 10m --retries 5 \
    --log-level INFO --log-file "$ROOT/system-logs/drive_sync.log" || \
    echo "$(date '+%F %T') [WARN] exports copy had issues" | tee -a "$LOG"
fi

echo "$(date '+%F %T') [OK  ] Backup + Drive sync complete" | tee -a "$LOG"
