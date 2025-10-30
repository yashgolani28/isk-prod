from __future__ import annotations
import os, re
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np

# ---- OCR (fast, angle-robust, CPU) ----
from rapidocr_onnxruntime import RapidOCR

# ---- Optional YOLO (char model + plate detector if present) ----
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

__all__ = ["run_anpr"]

# =========================
# Config
# =========================
_LP_DET_WEIGHTS = os.path.join("weights", "vehicle_plate_det_bike_roi.pt")  # optional; if missing we use color/edge fallback
_MIN_PLATE_W, _MIN_PLATE_H = 64, 20                      # gates for candidates
_GAMMAS = (0.8, 1.0, 1.2)                                # exposures for TTA
_ROT_SWEEP = (-8, -4, 0, 4, 8)                           # degrees for TTA

# Character detector 
_CHAR_DET_WEIGHTS_CANDIDATES = [
    os.path.join("weights", "plate_reco_best_v2.pt"),
    os.path.join(os.path.dirname(__file__), "plate_reco_best_v2.pt"),
    "plate_reco_best_v2.pt",  # allow cwd
]

# index → char; 26 looked off in your original mapping; default to 'Q' but allow override via env
_CLASS_MAP = {
    0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7',
    8:'8', 9:'9', 10:'A', 11:'B', 12:'C', 13:'D', 14:'E',
    15:'F', 16:'G', 17:'H', 18:'I', 19:'J', 20:'K', 21:'L',
    22:'M', 23:'N', 24:'O', 25:'P', 26:os.environ.get("ANPR_CLASS26", "Q"),
    27:'R', 28:'S', 29:'T', 30:'U', 31:'V', 32:'W', 33:'X', 34:'Y', 35:'Z'
}

# Indian state/UT codes
INDIA_STATE_CODES = {
    'AN','AP','AR','AS','BR','CH','DD','DL','DN','GA','GJ','HP','HR','JH','JK','KA','KL','LA',
    'LD','MH','ML','MN','MP','MZ','NL','OD','PB','PY','RJ','SK','TN','TS','TR','UK','UP','WB'
}

# Common letter↔digit confusions
_DIGIT_FOR_LETTER = {"O":"0","Q":"0","D":"0","I":"1","L":"1","Z":"2","S":"5","B":"8","G":"6","T":"7"}
_LETTER_FOR_DIGIT = {"0":"O","1":"I","2":"Z","4":"A","5":"S","6":"G","7":"T","8":"B"}

# Indian plate colors & blue strip (HSV)
_HSV_GREEN  = ((35,  40,  40), (85, 255, 255))   # EV green BG
_HSV_YELLOW = ((15,  60,  60), (35, 255, 255))   # commercial yellow BG
_HSV_WHITE  = ((0,    0, 160), (180,  60, 255))  # private white BG (bright)
_HSV_BLUE   = ((95,  80,  40), (130, 255, 255))  # left blue state band

# =========================
# Lazy singletons
# =========================
_rapid = RapidOCR()
_lp_det_model = None
_char_det_model = None

def _get_lp_detector():
    """Load YOLO plate detector once if weights exist; else None."""
    global _lp_det_model
    if YOLO is None:
        return None
    if _lp_det_model is not None:
        return _lp_det_model
    if os.path.exists(_LP_DET_WEIGHTS):
        try:
            _lp_det_model = YOLO(_LP_DET_WEIGHTS)
            return _lp_det_model
        except Exception:
            return None
    return None

def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points (x,y) as tl,tr,br,bl for homography."""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def _deskew_crop(img: np.ndarray, quad: np.ndarray, pad: float = 0.10) -> np.ndarray:
    """
    Perspective-rectify the plate polygon to a tight, padded crop.
    quad: 4x2 polygon in image coords.
    pad:  % padding around the rectified ROI.
    """
    H, W = img.shape[:2]
    quad = np.clip(quad, [0, 0], [W - 1, H - 1]).astype(np.float32)
    quad = _order_quad(quad)
    # target width/height from side lengths
    wA = np.linalg.norm(quad[1] - quad[0])
    wB = np.linalg.norm(quad[2] - quad[3])
    hA = np.linalg.norm(quad[3] - quad[0])
    hB = np.linalg.norm(quad[2] - quad[1])
    width = int(max(wA, wB))
    height = int(max(hA, hB))
    # enforce plate-ish aspect
    asp = max(2.0, min(6.0, (width + 1e-6) / (height + 1e-6)))
    height = max(20, int(width / asp))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_CUBIC)
    # padding
    if pad > 0:
        px = int(round(width * pad))
        py = int(round(height * pad))
        warped = cv2.copyMakeBorder(warped, py, py, px, px, cv2.BORDER_REPLICATE)
    return warped

def _pick_best_plate_polygon(ocr_boxes: List[np.ndarray], img_shape) -> Optional[np.ndarray]:
    """From OCR quads, pick the most plate-like polygon."""
    H, W = img_shape[:2]
    best = None
    best_score = -1.0
    for quad in ocr_boxes:
        quad = np.asarray(quad, dtype=np.float32)
        x_min = float(np.min(quad[:, 0])); x_max = float(np.max(quad[:, 0]))
        y_min = float(np.min(quad[:, 1])); y_max = float(np.max(quad[:, 1]))
        w = x_max - x_min; h = y_max - y_min
        if w < _MIN_PLATE_W or h < _MIN_PLATE_H:
            continue
        asp = w / (h + 1e-6)
        if not (1.8 <= asp <= 7.0):
            continue
        area = w * h
        score = area * (1.0 - abs(asp - 4.0) / 4.0)  # prefer ~4:1 and bigger
        if score > best_score:
            best_score, best = score, quad
    return best

def _get_char_detector():
    """Load custom char detector once if weights exist; else None."""
    global _char_det_model
    if YOLO is None:
        return None
    if _char_det_model is not None:
        return _char_det_model
    for p in _CHAR_DET_WEIGHTS_CANDIDATES:
        if os.path.exists(p):
            try:
                _char_det_model = YOLO(p)
                return _char_det_model
            except Exception:
                continue
    return None

# =========================
# Helpers
# =========================
def _enhance_low_light(img: np.ndarray) -> np.ndarray:
    """Mild y-CLAHE + bilateral + unsharp + gamma TTA (pick best)."""
    def _proc(im, gamma=1.0):
        ycc = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycc)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        y = clahe.apply(y)
        ycc = cv2.merge((y,cr,cb))
        out = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
        out = cv2.bilateralFilter(out, 5, 40, 40)
        blur = cv2.GaussianBlur(out, (0,0), 1.0)
        out = cv2.addWeighted(out, 1.6, blur, -0.6, 0)
        if abs(gamma-1.0) > 1e-3:
            lut = np.array([np.clip((i/255.0)**(1.0/gamma)*255,0,255) for i in range(256)], dtype=np.uint8)
            out = cv2.LUT(out, lut)
        return out
    best, score, best_g = img, -1e9, 1.0
    for g in _GAMMAS:
        t = _proc(img, g)
        # simple sharpness score
        s = cv2.Laplacian(cv2.cvtColor(t, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        if s > score:
            best, score, best_g = t, s, g
    return best

def _bgr_to_hsv_mask(bgr: np.ndarray, lo_hi) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    (lh,ls,lv),(hh,hs,hv) = lo_hi
    return cv2.inRange(hsv, (lh,ls,lv), (hh,hs,hv))

def _largest_box_from_mask(mask: np.ndarray, min_wh=(60,18)) -> Optional[tuple]:
    """Return xyxy of the largest reasonably rectangular contour, or None."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_a = None, 0
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w < min_wh[0] or h < min_wh[1]:
            continue
        ar = w / max(1,h)
        if 1.8 <= ar <= 6.5:
            a = w*h
            if a > best_a:
                best, best_a = (x, y, x+w, y+h), a
    return best

def _trim_left_blue_strip(crop: np.ndarray) -> np.ndarray:
    """Remove IN blue state band if present to help OCR."""
    h, w = crop.shape[:2]
    if w < 50:
        return crop
    band = crop[:, :min(int(0.18*w), 80)]
    m = _bgr_to_hsv_mask(band, _HSV_BLUE)
    if (m > 0).mean() > 0.25:
        return crop[:, band.shape[1]:]
    return crop

def _color_guided_plate_box(img: np.ndarray) -> Optional[tuple]:
    """Fast color heuristic for Indian plates; works when YOLO is absent/slow."""
    masks = [
        _bgr_to_hsv_mask(img, _HSV_GREEN),
        _bgr_to_hsv_mask(img, _HSV_YELLOW),
        _bgr_to_hsv_mask(img, _HSV_WHITE),
    ]
    mask = np.clip(masks[0] + masks[1] + masks[2], 0, 255).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (9,3)),
                            iterations=2)
    return _largest_box_from_mask(mask)

def _crop(img, box):
    x1, y1, x2, y2 = map(int, box)
    h, w = img.shape[:2]
    x1 = max(0, min(w-1, x1)); x2 = max(0, min(w, x2))
    y1 = max(0, min(h-1, y1)); y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    pad = max(2, int(0.04 * max(x2-x1, y2-y1)))  # small context helps OCR
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
    return img[y1:y2, x1:x2].copy()

def _ocr_with_char_detector(img: np.ndarray) -> tuple[str, float]:
    """Detect characters with YOLO, sort by x (two lines handled), decode via _CLASS_MAP."""
    det = _get_char_detector()
    if det is None:
        return "", 0.0
    try:
        r = det.predict(source=img, imgsz=640, conf=0.25, iou=0.5, verbose=False)[0]
    except Exception:
        return "", 0.0
    if not getattr(r, "boxes", None) or len(r.boxes) == 0:
        return "", 0.0
    xyxy = r.boxes.xyxy.cpu().numpy()
    cls  = r.boxes.cls.cpu().numpy().astype(int)
    sc   = r.boxes.conf.cpu().numpy()
    centers = np.column_stack([(xyxy[:,1]+xyxy[:,3])*0.5, (xyxy[:,0]+xyxy[:,2])*0.5, sc, cls])
    ys = centers[:,0]
    rows = [centers]
    # naive two-line split by median if vertical spread large
    if np.std(ys) > 12 and len(ys) >= 6:
        m = np.median(ys)
        rows = [centers[ys <= m], centers[ys > m]]
        rows = sorted(rows, key=lambda a: a[:,0].mean())  # top then bottom
    decoded, confs = [], []
    for row in rows:
        row = row[np.argsort(row[:,1])]
        for _,_, s, c in row:
            ch = _CLASS_MAP.get(int(c), "")
            if not ch:
                continue
            decoded.append(ch)
            confs.append(float(s))
        decoded.append(" ")
    if decoded and decoded[-1] == " ":
        decoded.pop()
    txt = "".join(decoded).replace(" ", "")
    if not txt:
        return "", 0.0
    if confs:
        k = max(1, int(len(confs)*0.8))
        conf = float(np.median(sorted(confs, reverse=True)[:k]))
    else:
        conf = 0.0
    return txt, conf

def _ocr_plate(img: np.ndarray) -> tuple[str, float]:
    """
    Try custom YOLO char-detector first; fall back to RapidOCR. Ensures conf=0.0 if text is empty.
    """
    # 1) char detector
    txt, conf = _ocr_with_char_detector(img)
    txt = "".join(ch for ch in (txt or "").upper() if ch.isalnum())
    if txt:
        return txt, float(conf)
    # 2) RapidOCR
    result, _ = _rapid(img)
    if not result:
        return "", 0.0
    best = max(result, key=lambda x: (len(x[1] or ""), x[2] or 0.0))
    txt = "".join(ch for ch in (best[1] or "").upper() if ch.isalnum())
    return (txt, float(best[2] or 0.0)) if txt else ("", 0.0)

def _postfix_india(text: str) -> str:
    """Heuristics for Indian plates (state code, district digits, swaps)."""
    t = re.sub(r"[^A-Z0-9]", "", (text or "").upper())
    if len(t) < 6:
        return t
    # Fix common swaps by position: AA00AA0000 pattern-ish
    # 0-1 state
    s = t[:2]
    if s not in INDIA_STATE_CODES and len(t) >= 2:
        s = "".join(_LETTER_FOR_DIGIT.get(ch, ch) for ch in s)
    # 2-3 district (digits)
    d = t[2:4]
    d = "".join(_DIGIT_FOR_LETTER.get(ch, ch) for ch in d)
    # 4-5 series letters
    ser = t[4:6]
    ser = "".join(_LETTER_FOR_DIGIT.get(ch, ch) for ch in ser)
    # rest mostly digits
    rest = t[6:]
    rest = "".join(_DIGIT_FOR_LETTER.get(ch, ch) for ch in rest)
    return (s + d + ser + rest)[:10]

# =========================
# Main entry
# =========================
def run_anpr(image_path: str, roi=None, save_dir="snapshots") -> Dict:
    """
    Returns dict: {plate_text, plate_conf, plate_bbox, crop_path}
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"plate_text":"", "plate_conf":0.0, "plate_bbox":None, "crop_path":None}

    # 1) Find plate box
    cand_box: Optional[Tuple[int,int,int,int]] = None
    cand_quad: Optional[np.ndarray] = None

    # Search image + offset: if ROI (e.g., vehicle bbox) is provided,
    # restrict detection to that region and later offset back to full-image coords.
    search_img = img
    off_x, off_y = 0, 0
    if roi and isinstance(roi, (list, tuple)) and len(roi) == 4:
        rx1, ry1, rx2, ry2 = map(int, roi)
        h, w = img.shape[:2]
        rx1 = max(0, min(w-1, rx1)); rx2 = max(0, min(w, rx2))
        ry1 = max(0, min(h-1, ry1)); ry2 = max(0, min(h, ry2))
        if rx2 > rx1 and ry2 > ry1:
            search_img = img[ry1:ry2, rx1:rx2]
            off_x, off_y = rx1, ry1

    # (a) OCR-driven polygon first: get RapidOCR quads, pick the most plate-like, and deskew
    try:
        ocr_raw, _ = _rapid(search_img)
        if ocr_raw:
            ocr_quads = []
            for item in ocr_raw:
                # RapidOCR returns [quad, text, score]; quad -> [[x,y],...x4]
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    quad_pts = np.array(item[0], dtype=np.float32).reshape(-1, 2)
                    if quad_pts.shape == (4, 2):
                        ocr_quads.append(quad_pts)
            picked = _pick_best_plate_polygon(ocr_quads, search_img.shape)
            if picked is not None:
                cand_quad = picked.astype(np.float32)
                # also compute a bbox for metadata/UI (offset back to full image coords)
                x_min = int(np.min(cand_quad[:, 0])) + off_x
                x_max = int(np.max(cand_quad[:, 0])) + off_x
                y_min = int(np.min(cand_quad[:, 1])) + off_y
                y_max = int(np.max(cand_quad[:, 1])) + off_y
                cand_box = (x_min, y_min, x_max, y_max)
    except Exception:
        pass

    # (b) YOLO plate detector if available
    if cand_box is None and cand_quad is None:
        try:
            det = _get_lp_detector()
            if det is not None:
                r = det.predict(source=search_img, imgsz=640, conf=0.25, iou=0.5, verbose=False)[0]
                if getattr(r, "boxes", None) and len(r.boxes):
                    b = r.boxes.xyxy.cpu().numpy()
                    s = r.boxes.conf.cpu().numpy()
                    i = int(s.argmax())
                    x1, y1, x2, y2 = map(int, b[i].tolist())
                    cand_box = (x1 + off_x, y1 + off_y, x2 + off_x, y2 + off_y)
        except Exception:
            pass

    # (c) color-guided finder for Indian plates
    if cand_box is None and cand_quad is None:
        tmp = _color_guided_plate_box(search_img)
        if tmp:
            x1, y1, x2, y2 = tmp
            cand_box = (x1 + off_x, y1 + off_y, x2 + off_x, y2 + off_y)

    # (d) classic edge/morph fallback
    if cand_box is None and cand_quad is None:
        gray = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(gray, 60, 140)
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        dil = cv2.dilate(edges, ker, iterations=1)
        cnts, _ = cv2.findContours(dil, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        best, best_a = None, 0
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            ar = w / max(1,h)
            if w >= _MIN_PLATE_W and h >= _MIN_PLATE_H and 1.6 <= ar <= 6.8:
                a = w*h
                if a > best_a:
                    best, best_a = (x,y,x+w,y+h), a
        cand_box = best

    # 2) Crop + enhance + OCR (+ always save crop)
    if cand_box is None and cand_quad is None:
        return {"plate_text":"", "plate_conf":0.0, "plate_bbox":None, "crop_path":None}
    if cand_quad is not None:
        # Perspective-rectified crop straight from the polygon (highest quality)
        crop = _deskew_crop(search_img, cand_quad, pad=0.08)
    else:
        crop = _crop(img, cand_box)
        if crop is None or crop.size == 0:
            return {"plate_text":"", "plate_conf":0.0, "plate_bbox":cand_box, "crop_path":None}

    # Trim blue state band; enhance
    crop = _trim_left_blue_strip(crop)
    crop2 = _enhance_low_light(crop)

    # Save crop (debug/overlay); ignore failures
    crop_path = None
    try:
        os.makedirs(save_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        crop_path = os.path.join(save_dir, f"{stem}_plate.jpg")
        cv2.imwrite(crop_path, crop)
    except Exception:
        pass

    # Small rotation sweep for robustness
    best_txt, best_conf = "", 0.0
    for deg in _ROT_SWEEP:
        M = cv2.getRotationMatrix2D((crop2.shape[1]/2, crop2.shape[0]/2), deg, 1.0)
        rot = cv2.warpAffine(crop2, M, (crop2.shape[1], crop2.shape[0]),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        txt, conf = _ocr_plate(rot)
        txt = _postfix_india(txt)
        if conf > best_conf or (conf >= best_conf and len(txt) > len(best_txt)):
            best_txt, best_conf = txt, conf

    # Never return nonzero confidence with empty text
    if not best_txt:
        best_conf = 0.0

    return {
        "plate_text": best_txt,
        "plate_conf": float(best_conf),
        "plate_bbox": cand_box,
        "crop_path": crop_path
    }

if __name__ == "__main__":
    import sys, json
    p = sys.argv[1] if len(sys.argv) > 1 else "snapshots/sample.jpg"
    out = run_anpr(p, roi=None, save_dir="snapshots")
    print(json.dumps(out, indent=2))
