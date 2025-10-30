#!/usr/bin/env python3
import os, sys, ssl, smtplib, traceback, psycopg2, psycopg2.extras
from email.message import EmailMessage
from email.utils import make_msgid
from datetime import datetime
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
REPORT_DIR = os.path.join(ROOT, "backups", "reports")
LOGO_PATH  = os.path.join(ROOT, "static", "essi_logo.jpeg")
LOG        = os.path.join(ROOT, "system-logs", "daily_report_email.py.log")
os.makedirs(REPORT_DIR, exist_ok=True); os.makedirs(os.path.dirname(LOG), exist_ok=True)

EMAIL_FROM = os.getenv("EMAIL_FROM",""); EMAIL_TO = os.getenv("EMAIL_TO",""); EMAIL_PASS = os.getenv("EMAIL_PASS","")
SMTP_HOST  = os.getenv("SMTP_HOST","smtp.gmail.com"); SMTP_PORT = int(os.getenv("SMTP_PORT","465"))
ALLOW_RESEND = os.getenv("ALLOW_RESEND","0").lower() in ("1","true","yes")
DSN = os.getenv("DB_DSN")

def log(s): 
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {s}"
    print(line, flush=True); open(LOG,"a").write(line+"\n")

def violations_pdf_path(): return os.path.join(REPORT_DIR, f"violations_{datetime.now():%Y%m%d}.pdf")
def sent_marker_path():    return violations_pdf_path() + ".sent"

def ensure_pdf_exists():
    pdf = violations_pdf_path()
    if os.path.exists(pdf): return pdf
    if ROOT not in sys.path: sys.path.insert(0, ROOT)
    from report import generate_pdf_report

    def fetch_rows():
        kw = {}
        conn = psycopg2.connect(DSN, **kw) if DSN else psycopg2.connect(dbname="iwr6843_db", user="radar_user", host="localhost")
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute("""
                    SELECT measured_at, sensor, object_id, type, confidence, speed_kmh, velocity, distance,
                           direction, motion_state, signal_level, doppler_frequency,
                           snapshot_path, snapshot_type, reviewed, flagged
                    FROM radar_data
                    WHERE measured_at::date = CURRENT_DATE
                      AND snapshot_path IS NOT NULL AND snapshot_path <> ''
                      AND LOWER(COALESCE(motion_state,'')) LIKE 'speeding%%'
                    ORDER BY measured_at ASC
                """)
                rows = [dict(r) for r in cur.fetchall()]
                for r in rows: r["datetime"] = r["measured_at"]
                return rows

    rows = fetch_rows()
    try:
        generate_pdf_report(pdf, title="Daily Violations Report",
                            summary={"total_records": len(rows)}, data=rows,
                            filters={"Mode":"Violations (DB-labeled)","Snapshots":"Annotated images included","Date":datetime.now().strftime("%Y-%m-%d")},
                            logo_path=LOGO_PATH, charts={})
        log(f"[BUILD] Created {os.path.basename(pdf)} with {len(rows)} rows")
    except Exception as e:
        log(f"[BUILD-ERR] {e}"); raise
    return pdf

def send_email(pdf_path):
    if not (EMAIL_FROM and EMAIL_TO and EMAIL_PASS):
        raise RuntimeError("EMAIL_FROM/EMAIL_TO/EMAIL_PASS not set")
    logo_cid = make_msgid(domain="essi.local")[1:-1]
    msg = EmailMessage()
    msg["Subject"] = f"[ESSI] Daily Violations Report â€“ {datetime.now():%B %d, %Y}"
    msg["From"] = EMAIL_FROM; msg["To"] = EMAIL_TO
    msg.set_content("HTML preferred.")
    msg.add_alternative(f"""<html><body>
      <p>Hello,<br><br>Please find attached the <b>violations-only</b> report for <b>{datetime.now():%B %d, %Y}</b>.</p>
      <p><img src="cid:{logo_cid}" style="height:30px"></p>
    </body></html>""", subtype="html")
    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH,"rb") as img:
            msg.get_payload()[-1].add_related(img.read(), maintype="image", subtype="jpeg", cid=f"<{logo_cid}>")
    with open(pdf_path,"rb") as f:
        msg.add_attachment(f.read(), maintype="application", subtype="pdf", filename=os.path.basename(pdf_path))
    if SMTP_PORT == 465:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ssl.create_default_context(), timeout=30) as s:
            s.login(EMAIL_FROM, EMAIL_PASS); s.send_message(msg)
    else:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as s:
            s.ehlo(); s.starttls(context=ssl.create_default_context()); s.login(EMAIL_FROM, EMAIL_PASS); s.send_message(msg)

def main():
    try:
        pdf = ensure_pdf_exists(); marker = sent_marker_path()
        if os.path.exists(marker) and not ALLOW_RESEND:
            log(f"[SKIP] already sent: {os.path.basename(marker)}"); return
        send_email(pdf); open(marker,"w").write(datetime.now().isoformat())
        log(f"[MAIL] Sent to {EMAIL_TO} with {os.path.basename(pdf)}")
    except Exception as e:
        log(f"[MAIL-ERR] {e}\n{traceback.format_exc()}"); sys.exit(1)

if __name__ == "__main__":
    main()

