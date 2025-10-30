from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate,
    Paragraph, Spacer, Table, Image, PageBreak
)
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from datetime import datetime
from zoneinfo import ZoneInfo
from xml.sax.saxutils import escape as xml_escape
import os
import json
import qrcode
import tempfile
import matplotlib
matplotlib.use("Agg")  # headless (Raspberry Pi safe)
import matplotlib.pyplot as plt

# ---- theme ----
BRAND_PRIMARY   = colors.HexColor("#0F172A")
BRAND_ACCENT    = colors.HexColor("#2563EB")
BRAND_MUTED     = colors.HexColor("#6B7280")
BG_HEADER       = colors.darkblue
FG_HEADER       = colors.whitesmoke
ROW_ALT         = colors.HexColor("#F8FAFF")
ROW_SPEEDING_BG = colors.HexColor("#FFF0F0")
SPEED_TEXT      = colors.HexColor("#B00020")

IST = ZoneInfo("Asia/Kolkata")
# ---- helpers ----
def safe_float(val, decimals=2):
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return f"{0:.{decimals}f}"

def _to_float(val, default=0.0):
    try: return float(val)
    except Exception: return float(default)

def _tmp_png():
    return tempfile.NamedTemporaryFile(delete=False, suffix=".png").name

def P(text, style=None):
    """Always escape before putting text in a Paragraph."""
    styles = getSampleStyleSheet()
    style = style or styles["BodyText"]
    return Paragraph(xml_escape(str(text if text is not None else "N/A")), style)

def _qr_png(text: str, box_px: int = 140) -> str | None:
    """Create a temp PNG QR code for given text; returns file path or None."""
    if qrcode is None:
        return None
    try:
        img = qrcode.QRCode(border=0, box_size=2)
        img.add_data(text)
        img.make(fit=True)
        pil = img.make_image(fill_color="black", back_color="white").convert("RGB")
        # resize to consistent box
        try:
            from PIL import Image as _PILImage
            pil = pil.resize((box_px, box_px), _PILImage.LANCZOS)
        except Exception:
            pass
        path = _tmp_png()
        pil.save(path, format="PNG")
        return path
    except Exception:
        return None

def _seal_from_bundle(bundle_dir: str) -> dict | None:
    """Load seal.json from bundle_dir, if present."""
    try:
        sp = os.path.join(bundle_dir, "seal.json")
        if os.path.exists(sp):
            return json.load(open(sp, "r"))
    except Exception:
        return None
    return None

def _verify_badge_paragraph(bundle_dir: str):
    """Return a small ✓/✗ Paragraph after verifying seal.json (or a neutral note)."""
    try:
        from evidence_seal import verify_seal  # local helper
        ok, _details, reason = verify_seal(bundle_dir)
        if ok:
            return P('<font color="#2e7d32">✓ Verified</font>')
        else:
            # concise reason
            return P(f'<font color="#c62828">✗ Tampered</font> <font color="#6B7280">({xml_escape(str(reason))})</font>')
    except Exception:
        return P('<font color="#6B7280">Signature check unavailable</font>')

def _qr_panel(seal_id: str, bundle_dir: str):
    """Build a right-aligned panel containing a QR and status for /verify/<seal_id>."""
    from reportlab.platypus import Table, Image, Spacer
    url = f"/verify/{seal_id}"
    qr_path = _qr_png(url, box_px=120)
    elems = []
    if qr_path and os.path.exists(qr_path):
        elems.append(Image(qr_path, width=0.95*inch, height=0.95*inch))
    elems.append(Spacer(1, 2))
    elems.append(P("<b>Verify evidence</b>"))
    elems.append(P(f"<font color='#6B7280'>{xml_escape(url)}</font>"))
    elems.append(_verify_badge_paragraph(bundle_dir))
    # single-cell table to keep the pack together, align right
    t = Table([[e for e in elems]],
              style=[("ALIGN",(0,0),(-1,-1),"RIGHT"),
                     ("VALIGN",(0,0),(-1,-1),"TOP")],
              hAlign="RIGHT", colWidths=[2.0*inch])
    # stash path so we can clean later
    t._qr_tmp_path = qr_path
    return t

def draw_chart_image(title, labels, data, *, max_bars=12):
    """Renders a compact bar chart to a temp PNG; returns path or None.
    - Caps bars to top-N to avoid OOM on Pi.
    - Skips empty/zero-only series.
    - Always closes figures to free memory.
    """
    try:
        if not labels or not data or len(labels) != len(data):
            return None

        # Coerce numeric and pair up
        pairs = []
        for l, v in zip(labels, data):
            try:
                pairs.append((str(l), float(v)))
            except Exception:
                continue
        if not pairs:
            return None

        # Keep only top-N bars, aggregate the rest into "Others"
        pairs.sort(key=lambda x: x[1], reverse=True)
        head = pairs[:max_bars]
        if len(pairs) > max_bars:
            others = sum(v for _, v in pairs[max_bars:])
            head.append(("Others", float(others)))
        labels = [l[:18] for l, _ in head]        # shorten labels
        vals   = [float(v) for _, v in head]

        if not any(v > 0 for v in vals):
            return None

        fig, ax = plt.subplots(figsize=(6.4, 2.2), dpi=96)
        ax.bar(range(len(labels)), vals)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=7)
        ax.set_title(str(title), fontsize=10, pad=6)
        ax.set_ylabel("Count", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")
        fig.tight_layout()

        # Save to a temp file and close
        f = _tmp_png()
        fig.savefig(f, dpi=96, bbox_inches="tight")
        plt.close(fig)
        return f
    except MemoryError:
        try: plt.close("all")
        except Exception: pass
        return None
    except Exception:
        try: plt.close("all")
        except Exception: pass
        return None

def _speed_limits_from(summary):
    raw = (summary or {}).get("speed_limits", {}) or {}
    try:
        lim = {str(k).upper(): float(v) for k, v in raw.items()}
    except Exception:
        lim = {}
    if not lim:
        lim = {"HUMAN": 8, "CAR": 60, "TRUCK": 50, "BUS": 50, "BIKE": 60, "BICYCLE": 10, "UNKNOWN": 50}
    if "DEFAULT" not in lim:
        lim["DEFAULT"] = lim.get("UNKNOWN", 50)
    return lim

class NumberedCanvas(canvas.Canvas):
    """Footer: centered ‘Generated on … | Page X of Y’"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_footer(total)
            super().showPage()
        super().save()

    def _draw_footer(self, page_count):
        """Center footer regardless of page size."""
        self.setFont("Helvetica", 8)
        w, _ = self._pagesize  # (width, height)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = f"Generated on {ts}  |  Page {self._pageNumber} of {page_count}"
        self.drawCentredString(w * 0.5, 18, text)

def add_header_footer(c, doc, logo_path):
    c.saveState()
    if logo_path and os.path.exists(logo_path):
        try:
            c.drawImage(logo_path, x=doc.leftMargin, y=doc.height + doc.bottomMargin + 6,
                        width=60, height=22, preserveAspectRatio=True, mask='auto')
        except Exception:
            pass
    c.restoreState()

# ---- centered section helpers ----
def _centered_title(text, styles):
    title_style = ParagraphStyle(
        'CenteredTitle', parent=styles['Title'], alignment=TA_CENTER,
        fontSize=22, textColor=BRAND_PRIMARY, spaceAfter=6
    )
    return Paragraph(xml_escape(text), title_style)

def _centered_subtle(text):
    return Paragraph(
        xml_escape(text),
        ParagraphStyle("subtle", alignment=TA_CENTER, fontSize=10, textColor=BRAND_MUTED)
    )

def _section_heading(text):
    h = Paragraph(
        f"<b>{xml_escape(text)}</b>",
        ParagraphStyle("H3Centered", alignment=TA_CENTER, fontSize=14, textColor=BRAND_PRIMARY, spaceAfter=4)
    )
    rule = HRFlowable(width="40%", thickness=1.2, color=BRAND_ACCENT, spaceBefore=2, spaceAfter=6, hAlign="CENTER")
    return [h, rule]

def _site_block(location: dict):
    loc = location or {}
    name    = (loc.get("name") or loc.get("site_name") or "N/A")
    address = (loc.get("address") or "N/A")
    lat     = loc.get("lat")
    lon     = loc.get("lon")
    source  = (loc.get("source") or "static").upper()

    coords  = f"{float(lat):.6f}, {float(lon):.6f}" if (lat is not None and lon is not None) else "N/A"

    styles = getSampleStyleSheet()
    title  = ParagraphStyle('site_title', parent=styles["BodyText"], textColor=BRAND_MUTED, fontSize=9)
    value  = ParagraphStyle('site_value', parent=styles["BodyText"], fontSize=10)

    data = [
        [Paragraph("Site", title),    Paragraph(str(name), value),     Paragraph("Coords", title),  Paragraph(coords, value)],
        [Paragraph("Address", title), Paragraph(str(address), value),  Paragraph("Source", title),  Paragraph(source, value)],
    ]
    t = Table(data, colWidths=[0.9*inch, 3.1*inch, 0.9*inch, 2.0*inch])
    t.setStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F8FAFC")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.HexColor("#D1D5DB")),
        ("BOX", (0,0), (-1,-1), 0.25, colors.HexColor("#D1D5DB")),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING",(0,0), (-1,-1), 6),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
    ])
    return t

def _kpi_cards(summary):
    styles = getSampleStyleSheet()
    n = styles["Normal"]; n.fontSize = 9
    center = ParagraphStyle('center', parent=n, alignment=TA_CENTER)
    big = ParagraphStyle('big', parent=center, fontSize=16, textColor=BRAND_PRIMARY)
    muted = ParagraphStyle('muted', parent=center, fontSize=8, textColor=BRAND_MUTED)

    def tile(label, value, sub=""):
        title_par = Paragraph(f"<b>{xml_escape(str(label))}</b>", center)
        big_par = Paragraph(f"<b>{xml_escape(str(value))}</b>", big)
        sub_par = Paragraph(xml_escape(str(sub)), muted)
        return Table([[title_par],[big_par],[sub_par]],
                     style=[("BACKGROUND",(0,0),(-1,-1), colors.white),
                            ("BOX",(0,0),(-1,-1),0.6, colors.HexColor("#D0D7DE")),
                            ("VALIGN",(0,0),(-1,-1),"MIDDLE")],
                     colWidths=[2.2*inch])

    total = summary.get("total_records", 0)
    avg = safe_float(summary.get("avg_speed", 0.0))
    top = safe_float(summary.get("top_speed", 0.0))
    low = safe_float(summary.get("lowest_speed", 0.0))
    auto = summary.get("auto_snapshots", 0)
    manual = summary.get("manual_snapshots", 0)
    approach   = summary.get("approaching_count", 0)
    depart     = summary.get("departing_count", 0)
    stationary = summary.get("stationary_count", 0)
    right_     = summary.get("right_count", 0)
    left_      = summary.get("left_count", 0)
    unknown_   = summary.get("unknown_count", 0)

    tiles = [
        tile("Total Records", total),
        tile("Average Speed (km/h)", avg),
        tile("Top Speed (km/h)", top),
        tile("Auto / Manual", f"{auto} / {manual}"),
        # two concise direction tiles
        tile("Approaching / Departing / Stationary", f"{approach} / {depart} / {stationary}"),
        tile("Right / Left / Unknown", f"{right_} / {left_} / {unknown_}"),
    ]

    rows = [tiles[i:i+3] for i in range(0, len(tiles), 3)]
    for r in rows:
        while len(r) < 3: r.append(Spacer(1,1))
    return Table(rows, colWidths=[2.4*inch]*3, hAlign="CENTER",
                 style=[("BOTTOMPADDING",(0,0),(-1,-1),6)])

def _filters_table(filters):
    styles = getSampleStyleSheet()
    if not filters:
        return Paragraph("<i>No filters applied (full dataset).</i>", styles["Italic"])
    n = styles["Normal"]
    rows = [[Paragraph("<b>Filter</b>", n), Paragraph("<b>Value</b>", n)]]
    for k, v in filters.items():
        rows.append([P(str(k).replace("_"," ").title(), n), P(v, n)])
    t = Table(rows, colWidths=[3.2*inch, 5.0*inch], hAlign="CENTER")
    t.setStyle([
        ("GRID",(0,0),(-1,-1),0.4,colors.lightgrey),
        ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("FONTSIZE",(0,0),(-1,-1),9),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("LEFTPADDING",(0,0),(-1,-1),6),
        ("RIGHTPADDING",(0,0),(-1,-1),6),
        ("WORDWRAP",(0,0),(-1,-1),"CJK"),
    ])
    return t

def _speed_limits_table(limits):
    n = getSampleStyleSheet()["Normal"]
    rows = [[Paragraph("<b>Object Type</b>", n), Paragraph("<b>Speed Limit (km/h)</b>", n)]]
    order = ["HUMAN","CAR","TRUCK","BUS","BIKE","BICYCLE","UNKNOWN","DEFAULT"]
    for k in order:
        val = limits.get(k, limits.get("DEFAULT", 50))
        rows.append([P(k.title(), n), P(safe_float(val, 0), n)])
    t = Table(rows, colWidths=[3.2*inch, 2.6*inch], hAlign="CENTER")
    t.setStyle([
        ("GRID",(0,0),(-1,-1),0.4,colors.lightgrey),
        ("BACKGROUND",(0,0),(-1,0),colors.lightblue),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("FONTSIZE",(0,0),(-1,-1),9),
        ("WORDWRAP",(0,0),(-1,-1),"CJK"),
    ])
    return t

def _snapshot_gallery(data, max_items=12, cols=3):
    items = []
    for r in (data or []):
        p = r.get("snapshot_path")
        if p and os.path.exists(p):
            dirn = str(r.get("direction", "N/A")).title()
            cap = (
                f"{str(r.get('type','UNK')).upper()}"
                f" | {safe_float(r.get('speed_kmh'),1)} km/h"
                f" | {safe_float(r.get('distance'),1)} m"
                f" | {dirn}"
                f" | {r.get('datetime','N/A')}"
            )
            items.append((p, cap))
        if len(items) >= max_items:
            break
    if not items:
        return None

    cells, row = [], []
    col_w = 3.2*inch
    for i, (path, caption) in enumerate(items, 1):
        try:
            img = Image(path, width=col_w*0.95, height=col_w*0.6, kind='proportional')
        except Exception:
            img = P("N/A")
        cap = P(caption, ParagraphStyle('cap',
                 parent=getSampleStyleSheet()["BodyText"],
                 fontSize=7, alignment=TA_CENTER))
        cell_content = [img, Spacer(1,2), cap]
        row.append(cell_content)
        if i % cols == 0:
            cells.append(row); row = []
    if row:
        while len(row) < cols: row.append(Spacer(1,1))
        cells.append(row)

    t = Table(cells, colWidths=[col_w]*cols, hAlign="CENTER")
    t.setStyle([("ALIGN",(0,0),(-1,-1),"CENTER"),
                ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                ("BOTTOMPADDING",(0,0),(-1,-1),6),
                ("WORDWRAP",(0,0),(-1,-1),"CJK")])
    return t

# ---- single detection (centered) ----
def generate_single_detection_pdf(filepath, record, logo_path="/home/pi/iwr6843isk/static/essi_logo.jpeg"):
    styles = getSampleStyleSheet()
    p = []

    if logo_path and os.path.exists(logo_path):
        lg = Image(logo_path, width=2.4*inch, height=0.85*inch); lg.hAlign = "CENTER"
        p.append(lg)

    p += [Spacer(1,8), _centered_title("Detection Report", styles),
          _centered_subtle(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), Spacer(1,16)]

    # ── Evidence Seal: If seal.json exists next to snapshot (or seal_id present), add QR panel ──
    _tmp_paths_to_cleanup = []
    try:
        snap = (record.get("snapshot_path") or "").strip()
        bundle_dir = os.path.dirname(snap) if snap else ""
        seal_id = (record.get("seal_id") or "").strip()
        if not seal_id and bundle_dir:
            seal = _seal_from_bundle(bundle_dir)
            if seal:
                seal_id = (seal.get("payload", {}) or {}).get("seal_id") or ""
        if seal_id and bundle_dir and os.path.isdir(bundle_dir) and os.path.exists(os.path.join(bundle_dir, "seal.json")):
            panel = _qr_panel(seal_id, bundle_dir)
            # capture tmp png to delete later
            if getattr(panel, "_qr_tmp_path", None):
                _tmp_paths_to_cleanup.append(panel._qr_tmp_path)
            p.append(panel)
            p.append(Spacer(1, 10))
    except Exception:
        pass

    t = str(record.get("type","UNKNOWN")).upper()
    s = f"{safe_float(record.get('speed_kmh'))} km/h"
    d = str(record.get("direction","N/A")).title()

    # --- centered KPI row ----------------------------------------------------
    styles_local = getSampleStyleSheet()
    _kpi_title = ParagraphStyle(
        'kpi_title', parent=styles_local["BodyText"],
        fontSize=10, alignment=TA_CENTER, textColor=BRAND_MUTED, leading=12
    )
    _kpi_value = ParagraphStyle(
        'kpi_value', parent=styles_local["BodyText"],
        fontSize=16, alignment=TA_CENTER, leading=18
    )
    def _tile(label, value):
        return Table(
            [[Paragraph(label, _kpi_title)],
             [Paragraph(value, _kpi_value)]],
            colWidths=[2.6*inch],
            style=[
                ("ALIGN", (0,0), (-1,-1), "CENTER"),
                ("VALIGN",(0,0), (-1,-1), "MIDDLE"),
                ("BOTTOMPADDING",(0,0), (-1,-1), 8),
            ],
        )
    kpi_row = [
        _tile("Object", t),
        _tile("Speed", s),
        _tile("Direction", d),
    ]
    p.append(Table(
        [kpi_row],
        colWidths=[2.8*inch, 2.8*inch, 2.8*inch],
        hAlign="CENTER",
        style=[
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("VALIGN",(0,0), (-1,-1), "MIDDLE"),
            ("LEFTPADDING", (0,0), (-1,-1), 0),
            ("RIGHTPADDING",(0,0), (-1,-1), 0),
        ],
    ))
    p.append(Spacer(1, 12))

    # hero image
    img_w = 6.5*inch
    try:
        if record.get("snapshot_path") and os.path.exists(record["snapshot_path"]):
            hero_img = Image(record["snapshot_path"], width=img_w, height=img_w*0.6, kind='proportional')
            hero_img.hAlign = "CENTER"
            p.append(hero_img)
        else:
            p.append(P("<i>Image unavailable</i>", ParagraphStyle('unavailable', parent=styles["Italic"], alignment=TA_CENTER)))
    except Exception:
        p.append(P("<i>Image unavailable</i>", ParagraphStyle('unavailable', parent=styles["Italic"], alignment=TA_CENTER)))
    p.append(Spacer(1, 10))

    # Always show plate text (even if no crop is available)
    try:
        _plate_txt  = (record.get("plate_text") or "").strip()
        _plate_conf = safe_float(record.get("plate_conf"), 2)
        if _plate_txt:
            p.append(P(f"<b>Plate:</b> {_plate_txt} (conf: {_plate_conf})",
                       ParagraphStyle('plate', alignment=TA_CENTER)))
            p.append(Spacer(1, 8))
    except Exception:
        pass

    # OPTIONAL: plate crop image (if present)
    try:
        _plate_crop = (record.get("plate_crop_path") or "").strip()
        if _plate_crop and os.path.exists(_plate_crop):
            plate_img = Image(_plate_crop, width=3.2*inch, height=0.85*inch, kind='proportional')
            plate_img.hAlign = "CENTER"
            p.append(plate_img)
            p.append(Spacer(1, 6))
    except Exception:
        pass

    # details (all escaped)
    n = styles["Normal"]
    hdr = [P("Property", n), P("Value", n), P("Property", n), P("Value", n)]
    def fmt(key):
        v = record.get(key)
        if v is None or v == "": return "N/A"
        if isinstance(v, (int, float)): return safe_float(v)
        return str(v)
    rows = [hdr] + [[P(c, n) for c in row] for row in [
        ("Datetime", fmt("datetime"), "Object ID", fmt("object_id")),
        ("Sensor", fmt("sensor"), "Snapshot", os.path.basename(record.get("snapshot_path") or "")),
        ("Type", t, "Confidence", safe_float(record.get("confidence"))),
        ("Speed (km/h)", safe_float(record.get("speed_kmh")), "Velocity (m/s)", safe_float(record.get("velocity"))),
        ("Distance (m)", safe_float(record.get("radar_distance","distance")), "Direction", d),
        ("Motion State", fmt("motion_state"), "Doppler (Hz)", safe_float(record.get("doppler_frequency"))),
        ("Signal (dB)", safe_float(record.get("signal_level"), 1), "Reviewed", "Yes" if record.get("reviewed") else "No"),
        ("Flagged", "Yes" if record.get("flagged") else "No", "Plate", str(record.get("plate_text") or "N/A"))
    ]]
    details_table = Table(rows, colWidths=[1.8*inch, 2.2*inch, 1.8*inch, 2.2*inch], hAlign="CENTER")
    details_table.setStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F8FAFC")),
        ("TEXTCOLOR",  (0,0), (-1,0), BRAND_PRIMARY),
        ("FONTNAME",   (0,0), (-1,0), 'Helvetica-Bold'),
        ("FONTSIZE",   (0,0), (-1,0), 9),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#D1D5DB")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("FONTSIZE", (0,1), (-1,-1), 9),
        ("WORDWRAP",(0,0),(-1,-1),"CJK"),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING",(0,0), (-1,-1), 8),
        ("TOPPADDING",  (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
        ("ALIGN", (0,0), (-1,0), "CENTER"),
    ])
    p.append(details_table)

    doc = BaseDocTemplate(filepath, pagesize=A4, leftMargin=40, rightMargin=40, topMargin=30, bottomMargin=40)
    frame = Frame(doc.leftMargin, doc.bottomMargin + 20, doc.width, doc.height - 20, id="main_frame")
    doc.addPageTemplates(PageTemplate(id="main", frames=[frame],
                                      onPage=lambda canv, d: add_header_footer(canv, d, logo_path)))
    doc.build(p, canvasmaker=NumberedCanvas)
    try:
        for _p in _tmp_paths_to_cleanup:
            if _p and os.path.exists(_p):
                os.remove(_p)
    except Exception:
        pass

# ---- full report ----
def generate_pdf_report(filepath, title="Radar Based Speed Detection Report",
                        logo_path="/home/pi/iwr6843isk/static/essi_logo.jpeg",
                        summary=None, data=None, filters=None, charts=None):

    styles = getSampleStyleSheet()
    content = []

    if logo_path and os.path.exists(logo_path):
        logo = Image(logo_path, width=2.4*inch, height=0.8*inch); logo.hAlign = 'CENTER'
        content.append(logo)
    content += [Spacer(1, 6),
                _centered_title(title, styles),
                _centered_subtle(datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                Spacer(1, 8)]

    try:
        seal_id_sum = ((summary or {}).get("seal_id") or "").strip()
        snap_sum = ((summary or {}).get("snapshot_path") or "").strip()
        bdir_sum = os.path.dirname(snap_sum) if snap_sum else ((summary or {}).get("bundle_dir") or "")
        if seal_id_sum and bdir_sum and os.path.exists(os.path.join(bdir_sum, "seal.json")):
            panel = _qr_panel(seal_id_sum, bdir_sum)
            content.append(panel)
            content.append(Spacer(1, 10))
    except Exception:
        pass

    content.append(Spacer(1, 6))

    limits = _speed_limits_from(summary or {})
    if summary:
        content.append(_kpi_cards(summary))
        content.append(Spacer(1, 12))
        content += _section_heading("Configured Speed Limits")
        content.append(_speed_limits_table(limits))
        content.append(Spacer(1, 12))

    if filters:
        content.append(PageBreak())
        content += _section_heading("Filters")
        content.append(_filters_table(filters))
        content.append(Spacer(1, 12))

    # charts
    if charts and isinstance(charts, dict):
        imgs = []
        # Sort charts by total value and cap number of charts
        MAX_CHARTS = int(os.environ.get("PDF_MAX_CHARTS", 6))
        def _score(item):
            vals = (item[1] or {}).get("data", []) or []
            s = 0.0
            for v in vals:
                try: s += float(v)
                except Exception: pass
            return s
        items = sorted(list(charts.items()), key=_score, reverse=True)[:MAX_CHARTS]
        for chart_title, chart_data in items:
            labels = (chart_data or {}).get("labels", []) or []
            vals   = (chart_data or {}).get("data", []) or []
            img = draw_chart_image(
                chart_title.replace("_"," ").title(),
                labels, vals,
                max_bars=int(os.environ.get("PDF_MAX_BARS", 12))
            )
            if img and os.path.exists(img):
                imgs.append(img)
        if imgs:
            content.append(PageBreak())
            content += _section_heading("Analytics Charts")
            for fp in imgs:
                im = Image(fp, width=6.9*inch, height=2.2*inch); im.hAlign = "CENTER"
                content.append(im)
                content.append(Spacer(1,8))

    # gallery
    gal = _snapshot_gallery(data or [])
    if gal:
        content.append(PageBreak())
        content += _section_heading("Recent Snapshots")
        content.append(gal)
        content.append(Spacer(1, 8))

    # detection data
    if data:
        content.append(PageBreak())
        content += _section_heading("Detection Data")

        header_labels = [
            "Snapshot", "Datetime", "Sensor", "Object ID", "Type", "Confidence",
            "Speed (km/h)", "Velocity (m/s)", "Distance (m)",
            "Direction", "Motion State", "Signal (dB)", "Doppler (Hz)",
            "Reviewed", "Flagged", "Plate", "Plate Conf"
        ]

        # all headers are Paragraphs
        header_cells = [P(h, ParagraphStyle('hdr', parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=8)) for h in header_labels]
        table_data = [header_cells]

        speeding_rows = []
        nstyle = ParagraphStyle('cell', parent=styles["BodyText"], fontSize=7)

        def is_speeding(row):
            t = str(row.get("type","UNKNOWN")).upper()
            spd = _to_float(row.get("speed_kmh"), 0)
            return spd > limits.get(t, limits.get("DEFAULT", 50))

        for row in data:
            # Snapshot
            thumb_path = row.get("snapshot_path")
            try:
                if thumb_path and os.path.exists(thumb_path):
                    img = Image(thumb_path, width=0.75*inch, height=0.52*inch)
                else:
                    img = P("N/A", nstyle)
            except Exception:
                img = P("N/A", nstyle)

            if is_speeding(row):
                speeding_rows.append(len(table_data))

            table_data.append([
                img,
                P(row.get("datetime", "N/A"), nstyle),
                P(row.get("sensor", "N/A"), nstyle),
                P(row.get("object_id", "N/A"), nstyle),
                P(str(row.get("type", "UNKNOWN")).upper(), nstyle),
                P(safe_float(row.get('confidence')), nstyle),
                P(safe_float(row.get('speed_kmh')), nstyle),
                P(safe_float(row.get('velocity')), nstyle),
                P(safe_float(row.get('distance')), nstyle),
                P(str(row.get("direction","N/A")).title(), nstyle),
                P(row.get("motion_state","N/A"), nstyle),
                P(safe_float(row.get('signal_level'), 1), nstyle),
                P(safe_float(row.get('doppler_frequency')), nstyle),
                P("Yes" if row.get("reviewed") else "No", nstyle),
                P("Yes" if row.get("flagged") else "No", nstyle),
                P(row.get("plate_text","") or "N/A", nstyle),
                P(safe_float(row.get("plate_conf"),2), nstyle),
            ])

        table = Table(table_data, repeatRows=1, colWidths=[0.85*inch] + [None]*(len(header_labels)-1), hAlign="CENTER")
        style = [
            ("BACKGROUND",(0,0),(-1,0), BG_HEADER),
            ("TEXTCOLOR",(0,0),(-1,0), FG_HEADER),
            ("GRID",(0,0),(-1,-1),0.25, colors.grey),
            ("FONTSIZE",(0,0),(-1,-1),7),
            ("ALIGN",(0,0),(-1,-1),"CENTER"),
            ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("LEFTPADDING",(0,0),(-1,-1),2),
            ("RIGHTPADDING",(0,0),(-1,-1),2),
            ("TOPPADDING",(0,0),(-1,-1),2),
            ("BOTTOMPADDING",(0,0),(-1,-1),2),
            ("WORDWRAP",(0,0),(-1,-1),"CJK"),
        ]
        # zebra striping + highlight speeding rows + emphasize speed column
        for i in range(1, len(table_data)):
            if i % 2 == 1:
                style.append(("BACKGROUND",(0,i),(-1,i), ROW_ALT))
        for r in speeding_rows:
            style.append(("BACKGROUND",(0,r),(-1,r), ROW_SPEEDING_BG))
            style.append(("TEXTCOLOR",(6,r),(6,r), SPEED_TEXT))

        table.setStyle(style)
        content.append(table)

    # build
    doc = BaseDocTemplate(
        filepath, pagesize=landscape(A4),
        rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='normal')
    template = PageTemplate(id='withHeaderFooter', frames=frame,
                            onPage=lambda c, d: add_header_footer(c, d, logo_path))
    doc.addPageTemplates([template])
    doc.build(content, canvasmaker=NumberedCanvas)
    try:
        for _fp in locals().get('imgs', []):
            if _fp and os.path.exists(_fp):
                os.remove(_fp)
    except Exception:
        pass
    return filepath
