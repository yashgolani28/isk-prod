import os, json, base64, hashlib, uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Tuple, Optional, TYPE_CHECKING

try:
    import nacl.signing as _nacl_signing
    import nacl.exceptions as _nacl_exceptions
except Exception:  # PyNaCl not available at runtime
    _nacl_signing = None
    _nacl_exceptions = None

# For type checkers only (Pylance/Pyright); not evaluated at runtime.
if TYPE_CHECKING:
    from nacl.signing import SigningKey, VerifyKey
    from nacl.exceptions import BadSignatureError

KEYS_DIR = "keys"
PRIV_PATH = os.environ.get("EVISEAL_PRIVATE", os.path.join(KEYS_DIR, "ed25519_private.pem"))
PUB_PATH  = os.environ.get("EVISEAL_PUBLIC",  os.path.join(KEYS_DIR, "ed25519_public.pem"))

def _b64(x: bytes) -> str: 
    return base64.b64encode(x).decode("ascii")
logger = logging.getLogger(__name__)
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_keys(autogen: bool=True) -> Tuple[Optional["SigningKey"], Optional["VerifyKey"]]:
    os.makedirs(KEYS_DIR, exist_ok=True)
    sk = None; vk = None
    if _nacl_signing is None:
        return None, None
    try:
        if os.path.exists(PRIV_PATH):
            raw = open(PRIV_PATH, "rb").read().strip()
            # Accept raw 32-byte seed or PEM-like base64
            if raw.startswith(b"-----BEGIN"):
                b64 = b"".join(line.strip() for line in raw.splitlines() if b"-----" not in line)
                seed = base64.b64decode(b64)
            else:
                seed = raw
            sk = _nacl_signing.SigningKey(seed)
            vk = sk.verify_key
        elif autogen:
            sk = _nacl_signing.SigningKey.generate()
            vk = sk.verify_key
            with open(PRIV_PATH, "wb") as f: f.write(sk._seed)
            with open(PUB_PATH,  "wb") as f: f.write(bytes(vk))
    except Exception:
        sk = None
        try:
            if os.path.exists(PUB_PATH):
                vk = VerifyKey(open(PUB_PATH,"rb").read())
        except Exception:
            vk = None
    return sk, vk

def pubkey_fingerprint(vk: "VerifyKey") -> str:
    return hashlib.sha256(bytes(vk)).hexdigest()[:16]

def make_seal(bundle_dir: str, site: str, seal_id: Optional[str]=None, extra: Optional[Dict]=None) -> Dict:
    """Compute hashes, sign payload, write bundle_dir/seal.json. Returns seal dict."""
    img  = os.path.join(bundle_dir, "image.jpg")
    meta = os.path.join(bundle_dir, "meta.json")
    clip = os.path.join(bundle_dir, "clip.mp4")
    try:
        qr_text = None
        if isinstance(extra, dict):
            qr_text = extra.get("qr_text") or extra.get("verify_url")
        if not qr_text:
            qr_text = os.environ.get("EVISEAL_QR")
        if qr_text and os.path.exists(img):
            # overlay_qr is idempotent and will skip if the bundle is already sealed.
            overlay_qr(img, qr_text)
    except Exception as e:
        logger.warning(f"[SEAL] QR overlay skipped: {e}")
    payload = {
        "seal_id":    seal_id or uuid.uuid4().hex[:12],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "site":       site,
        "files": {
            "image.jpg": sha256_file(img)  if os.path.exists(img)  else None,
            "meta.json": sha256_file(meta) if os.path.exists(meta) else None,
            "clip.mp4":  sha256_file(clip) if os.path.exists(clip) else None,
        }
    }
    if extra:
        payload["extra"] = extra
    sk, vk = ensure_keys()
    sig_b64 = None; vk_fpr = None
    if sk and vk:
        m = json.dumps(payload, sort_keys=True, separators=(",",":")).encode("utf-8")
        sig_b64 = _b64(sk.sign(m).signature)
        vk_fpr  = pubkey_fingerprint(vk)
    data = {"payload": payload, "sig_b64": sig_b64, "pubkey_fingerprint": vk_fpr, "alg": "ed25519"}
    with open(os.path.join(bundle_dir, "seal.json"), "w") as f:
        json.dump(data, f, indent=2)
    return data

def verify_seal(bundle_dir: str) -> Tuple[bool, Dict, str]:
    """Recompute hashes and verify signature in bundle_dir/seal.json."""
    path = os.path.join(bundle_dir, "seal.json")
    if not os.path.exists(path):
        return False, {}, "seal.json missing"
    data = json.load(open(path, "r"))
    payload = data.get("payload", {})
    files   = payload.get("files", {})

    # recompute
    recomputed = {}
    for name, old in files.items():
        p = os.path.join(bundle_dir, name)
        recomputed[name] = sha256_file(p) if (old and os.path.exists(p)) else None

    # signature
    sig_ok = False; reason = ""
    sig_b64 = data.get("sig_b64")
    try:
        if (_nacl_signing is not None) and sig_b64:
            vk = _nacl_signing.VerifyKey(open(PUB_PATH, "rb").read())
            m = json.dumps(payload, sort_keys=True, separators=(",",":")).encode("utf-8")
            vk.verify(m, base64.b64decode(sig_b64))
            sig_ok = True
        else:
            reason = "signature missing or library unavailable"
    except Exception as e:
        reason = f"signature invalid: {e}"
        sig_ok = False

    ok = sig_ok and all(files.get(k) == recomputed.get(k) for k in files.keys())
    details = {"payload": payload, "recomputed": recomputed, "sig_ok": sig_ok}
    if not ok and not reason:
        bad = [k for k in files.keys() if files.get(k)!=recomputed.get(k)]
        reason = f"hash mismatch: {', '.join(bad) or 'none'}"
    return ok, details, reason

def overlay_qr(image_path: str, qr_text: str, box_px: int=110, margin_px: int=8, corner: str="tr") -> bool:
    """Stamp a small QR and label ('Verify evidence') onto image_path in-place."""
    import qrcode
    from PIL import Image
    import cv2, numpy as np

    bundle_dir = os.path.dirname(image_path) or "."
    if os.path.exists(os.path.join(bundle_dir, "seal.json")):
        return True

    img = cv2.imread(image_path)
    if img is None:
        return False
    qr = qrcode.QRCode(border=0, box_size=2)
    qr.add_data(qr_text); qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    qr_img = qr_img.resize((box_px, box_px))
    qr_np  = np.array(qr_img)[:, :, ::-1]  # RGB->BGR
    H, W = img.shape[:2]
    x = W - margin_px - box_px if corner.endswith("r") else margin_px
    y = margin_px if corner.startswith("t") else H - margin_px - box_px
    roi = img[y:y+box_px, x:x+box_px]
    if roi.shape == qr_np.shape:
        # JPEG introduces tiny variations; allow a small tolerance.
        if np.allclose(roi.astype(np.int16), qr_np.astype(np.int16), atol=3):
            return True
    img[y:y+box_px, x:x+box_px] = qr_np
    cv2.putText(img, "Verify evidence", (x, y+box_px+16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(img, "Verify evidence", (x, y+box_px+16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    cv2.imwrite(image_path, img)
    return True