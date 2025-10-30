import os
import re
import io
import time
import json
import numpy as np
import math
import queue
import hashlib
import threading
import subprocess
import shutil
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, List, Optional, Tuple, Callable, Dict, Any

import requests
from requests.auth import HTTPDigestAuth
import urllib3
from urllib.parse import urlparse, urlunparse, quote

# Optional fallback for encoding if ffmpeg is not available
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# =====================================================================================
#  SNAPSHOT (existing API preserved)
# =====================================================================================

def _ffmpeg_bin():
    """Resolve ffmpeg path from env or PATH (Windows-safe)."""
    p = os.environ.get("FFMPEG_BIN") or os.environ.get("FFMPEG_PATH")
    if p and os.path.isfile(p):
        return p
    w = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if w:
        return w
    for g in (r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
              r"C:\ffmpeg\bin\ffmpeg.exe"):
        if os.path.isfile(g):
            return g
    return None

def capture_snapshot(camera_url: str,
                     output_dir: str = "snapshots",
                     username: Optional[str] = None,
                     password: Optional[str] = None,
                     timeout: float = 5.0,
                     verify: bool = False) -> Optional[str]:
    """
    One-off JPEG fetch from camera snapshot endpoint.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        # --- RTSP single-frame grab (Axis or any RTSP source) ---
        url = (camera_url or "").strip()
        if url.lower().startswith("rtsp://"):
            # embed auth if not already present
            try:
                p = urlparse(url)
                if (username and password) and not (p.username or p.password):
                    user = quote(username, safe="")
                    pwd = quote(password, safe="")
                    netloc = f"{user}:{pwd}@{p.hostname or ''}"
                    if p.port:
                        netloc += f":{p.port}"
                    url = urlunparse((p.scheme, netloc, p.path or "", p.params or "", p.query or "", p.fragment or ""))
            except Exception:
                pass

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            path = os.path.join(output_dir, f"speeding_{ts}.jpg")
            ff = _ffmpeg_bin()
            if not ff and _HAS_CV2:
                try:
                    cap = cv2.VideoCapture(url)
                    ok, frame = cap.read()
                    cap.release()
                    if ok and frame is not None:
                        ok2, enc = cv2.imencode(".jpg", frame)
                        if ok2:
                            with open(path, "wb") as f:
                                f.write(bytes(enc))
                            return path if os.path.exists(path) and os.path.getsize(path) >= 1024 else None
                except Exception as e:
                    print(f"[CAMERA EXCEPTION] OpenCV RTSP fallback failed: {e}")
                return None
            if not ff:
                print("[CAMERA ERROR] ffmpeg not found for RTSP snapshot")
                return None
            cmd = [
                ff, "-y",
                "-hide_banner", "-loglevel", "quiet",
                "-rtsp_transport", "tcp",
                "-stimeout", str(int(timeout * 1_000_000)),  # µs
                "-i", url,
                "-vframes", "1", "-q:v", "2", path
            ]
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=max(8, int(timeout) + 3), check=False)
            except Exception as e:
                print(f"[CAMERA EXCEPTION] ffmpeg: {e}")
                return None
            if os.path.exists(path) and os.path.getsize(path) >= 1024:
                return path
            if os.path.exists(path):
                try: os.remove(path)
                except Exception: pass
            print("[CAMERA ERROR] RTSP snapshot failed or too small")
            return None
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "image/jpeg,image/png,image/*,*/*",
            "Connection": "close",
        }
        resp = requests.get(
            camera_url,
            auth=HTTPDigestAuth(username, password) if username and password else None,
            timeout=timeout,
            headers=headers,
            verify=verify,
            stream=True,
        )
        if resp.status_code != 200:
            print(f"[CAMERA ERROR] Snapshot HTTP {resp.status_code}")
            return None

        ctype = resp.headers.get("content-type", "").lower()
        if "image" not in ctype:
            # Support MJPEG by extracting the first JPEG part
            if "multipart/x-mixed-replace" in ctype:
                boundary = None
                for part in ctype.split(";"):
                    part = part.strip()
                    if part.lower().startswith("boundary="):
                        boundary = part.split("=",1)[1].strip()
                        if boundary.startswith('"') and boundary.endswith('"'):
                            boundary = boundary[1:-1]
                if not boundary:
                    print("[CAMERA ERROR] MJPEG without boundary")
                    return None
                boundary_bytes = ("--" + boundary).encode()
                buf = b""
                jpeg = b""
                start = end = -1
                for chunk in resp.iter_content(chunk_size=8192):
                    if not chunk: break
                    buf += chunk
                    # Find JPEG SOI/EOI
                    if start < 0:
                        s = buf.find(b"\xff\xd8")
                        if s >= 0:
                            start = s
                    if start >= 0:
                        e = buf.find(b"\xff\xd9", start)
                        if e >= 0:
                            end = e + 2
                            jpeg = buf[start:end]
                            break
                    # avoid unbounded growth
                    if len(buf) > 4_000_000:
                        buf = buf[-1_000_000:]
                if not jpeg:
                    print("[CAMERA ERROR] Could not extract JPEG from MJPEG")
                    return None
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                path = os.path.join(output_dir, f"speeding_{ts}.jpg")
                with open(path, "wb") as f:
                    f.write(jpeg)
                if os.path.getsize(path) < 1024:
                    os.remove(path)
                    print("[CAMERA ERROR] Snapshot too small (mjpeg)")
                    return None
                return path
            # not image and not mjpeg
            print(f"[CAMERA ERROR] Unexpected content-type: {ctype}")
            try:
                print(f"[CAMERA DEBUG] Body: {resp.text[:200]}")
            except Exception:
                pass
            return None

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = os.path.join(output_dir, f"speeding_{ts}.jpg")

        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=4096):
                if chunk:
                    f.write(chunk)

        if not os.path.exists(path):
            return None
        if os.path.getsize(path) < 1024:
            os.remove(path)
            print("[CAMERA ERROR] Snapshot too small")
            return None
        return path

    except Exception as e:
        print(f"[CAMERA EXCEPTION] {e}")
        return None

def get_latest_jpeg() -> Optional[bytes]:
    """Returns latest JPEG bytes from the global ring, or None if not ready."""
    if _global_ring is None:
        return None
    fr = _global_ring.latest()
    return fr.jpeg if fr else None

def _si_no_window():
    try:
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0
        return si
    except Exception:
        return None

def _cf_no_window():
    return 0x08000000 if os.name == "nt" else 0

# =====================================================================================
#  VIDEO RING BUFFER (bytes only; minimal CPU)
# =====================================================================================

@dataclass
class RingFrame:
    ts_wall: float           # POSIX seconds (local)
    ts_mono: float           # monotonic seconds
    jpeg: bytes              # raw JPEG bytes
    size: int                # bytes


class VideoRingBuffer:
    """
    Background thread that polls snapshot (or MJPEG) endpoint at a capped FPS,
    keeping the last N seconds of JPEG frames in RAM (bytes only).
    """
    def __init__(self,
                 camera_url: str,
                 username: Optional[str],
                 password: Optional[str],
                 target_fps: float,
                 buffer_seconds: int,
                 timeout: float,
                 verify: bool):
        self.camera_url = camera_url
        self.username = username
        self.password = password
        self.target_fps = max(0.5, float(target_fps))
        self.buffer_seconds = max(1, int(buffer_seconds))
        self.timeout = timeout
        self.verify = verify

        # extra slack factor (2x) so we never underflow when slicing windows
        self._buf: Deque[RingFrame] = deque(maxlen=int(self.target_fps * self.buffer_seconds * 2))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None

        self._session = requests.Session()
        if username and password:
            self._session.auth = HTTPDigestAuth(username, password)
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0",
            "Accept": "image/jpeg,image/*,*/*",
            "Connection": "keep-alive",
        })

        self._last_err: Optional[str] = None
        self._last_ok_ts: float = 0.0
        self._cap: Optional["cv2.VideoCapture"] = None
        self._ffmpeg_proc: Optional[subprocess.Popen] = None

    @property
    def is_running(self) -> bool:
        return self._thr is not None and self._thr.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop.clear()
        self._thr = threading.Thread(target=self._run, name="VideoRingBuffer", daemon=True)
        self._thr.start()

    def stop(self, join_timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thr:
            self._thr.join(join_timeout)
        self._thr = None
        # Cleanup resources robustly (release both CV2 and ffmpeg if present)
        try:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
        except Exception:
            pass
        try:
            if self._ffmpeg_proc is not None:
                try:
                    self._ffmpeg_proc.kill()
                except Exception:
                    pass
                self._ffmpeg_proc = None
        except Exception:
            pass

    def _run(self) -> None:
        url = (self.camera_url or "").strip()
        if url.lower().startswith("rtsp://"):
            # Prefer FFmpeg (fully silenced); OpenCV can emit noisy decoder logs.
            self._run_rtsp_ffmpeg(url)
        else:
            # HTTP snapshot/MJPEG path 
            self._run_http_snapshot(url)

    # ---------------- HTTP snapshot/MJPEG  ---------------
    def _run_http_snapshot(self, url: str) -> None:
        """
        Supports two HTTP modes:
        1) Snapshot endpoints (single JPEG per GET) — paced to target_fps.
        2) MJPEG endpoints (multipart/x-mixed-replace) — continuous stream parsed by SOI/EOI.
        """
        interval = 1.0 / self.target_fps
        backoff = 0.5
        SOI = b"\xff\xd8"
        EOI = b"\xff\xd9"

        while not self._stop.is_set():
            t0 = time.monotonic()
            resp = None
            try:
                resp = self._session.get(
                    url, timeout=self.timeout, verify=self.verify, stream=True
                )
                ctype = (resp.headers.get("content-type") or "").lower()
                if resp.status_code != 200:
                    self._last_err = f"HTTP {resp.status_code} ctype={ctype}"
                    time.sleep(min(backoff, 1.5))
                    backoff = min(backoff * 1.6, 1.5)
                    continue

                # Mode A: MJPEG stream
                if "multipart" in ctype or "x-mixed-replace" in ctype:
                    buf = bytearray()
                    raw = resp.raw
                    raw.decode_content = True
                    while not self._stop.is_set():
                        chunk = raw.read(4096)
                        if not chunk:
                            break
                        buf.extend(chunk)
                        # Extract complete JPEGs by SOI/EOI
                        while True:
                            i = buf.find(SOI)
                            if i < 0:
                                buf.clear()
                                break
                            j = buf.find(EOI, i + 2)
                            if j < 0:
                                # keep from SOI onward
                                buf[:] = buf[i:]
                                break
                            j += 2
                            frame_bytes = bytes(buf[i:j])
                            del buf[:j]
                            if len(frame_bytes) < 1024:
                                continue
                            now_wall = time.time()
                            rf = RingFrame(ts_wall=now_wall, ts_mono=time.monotonic(), jpeg=frame_bytes, size=len(frame_bytes))
                            with self._lock:
                                self._buf.append(rf)
                            self._last_ok_ts = now_wall
                            self._last_err = None
                    # loop will reconnect if stream ends
                    continue

                # Mode B: Single-image snapshot
                if "image" in ctype:
                    data = resp.content
                    if data and len(data) >= 1024:
                        now_wall = time.time()
                        frame = RingFrame(ts_wall=now_wall, ts_mono=t0, jpeg=data, size=len(data))
                        with self._lock:
                            self._buf.append(frame)
                        self._last_ok_ts = now_wall
                        self._last_err = None
                        backoff = 0.5
                else:
                    self._last_err = f"Unexpected content-type: {ctype}"
            except Exception as e:
                self._last_err = f"http_snapshot:{e}"
            finally:
                try:
                    if resp is not None:
                        resp.close()
                except Exception:
                    pass

            # pacing only for snapshot mode
            elapsed = time.monotonic() - t0
            sleep_for = max(0.0, interval - elapsed)
            if sleep_for == 0.0 and elapsed > (interval * 4):
                time.sleep(min(backoff, 1.5))
                backoff = min(backoff * 1.6, 1.5)
            else:
                time.sleep(sleep_for)

    # ------------------------- RTSP helpers ----------------------------------
    def _rtsp_with_auth(self, url: str) -> str:
        if not (self.username and self.password):
            return url
        try:
            p = urlparse(url)
            if p.username or p.password:
                return url  # already embedded
            user = quote(self.username, safe="")
            pwd = quote(self.password, safe="")
            netloc = f"{user}:{pwd}@{p.hostname or ''}"
            if p.port:
                netloc += f":{p.port}"
            return urlunparse((p.scheme, netloc, p.path or "", p.params or "", p.query or "", p.fragment or ""))
        except Exception:
            return url

    # ------------------------- RTSP via OpenCV --------------------------------
    def _run_rtsp_cv2(self, url: str) -> None:
        rtsp = self._rtsp_with_auth(url)
        try:
            self._cap = cv2.VideoCapture(rtsp)
        except Exception as e:
            self._last_err = f"cv2_open:{e}"
            # Fallback to ffmpeg loop
            self._run_rtsp_ffmpeg(url)
            return

        if not self._cap or not self._cap.isOpened():
            self._last_err = "cv2_open_failed"
            self._run_rtsp_ffmpeg(url)
            return

        # Pace to target FPS
        interval = max(0.0, 1.0 / self.target_fps)
        next_t = time.monotonic()
        while not self._stop.is_set():
            # wait until next slot
            now = time.monotonic()
            if now < next_t:
                time.sleep(max(0.0, next_t - now))
            next_t += interval
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._last_err = "cv2_read_failed"
                time.sleep(0.1)
                continue
            # Encode to JPEG bytes
            try:
                ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if not ok2:
                    self._last_err = "cv2_imencode_failed"
                    continue
                data = buf.tobytes()
                if len(data) < 1024:
                    continue
                now_wall = time.time()
                rf = RingFrame(ts_wall=now_wall, ts_mono=now, jpeg=data, size=len(data))
                with self._lock:
                    self._buf.append(rf)
                self._last_ok_ts = now_wall
                self._last_err = None
            except Exception as e:
                self._last_err = f"cv2_encode:{e}"
                continue

    # ------------------------- RTSP via FFmpeg --------------------------------
    def _run_rtsp_ffmpeg(self, url: str) -> None:
        rtsp = self._rtsp_with_auth(url)
        ff = _ffmpeg_bin()
        if not ff:
            # If ffmpeg isn't available, try OpenCV path
            if _HAS_CV2:
                self._run_rtsp_cv2(url)
                return
            self._last_err = "ffmpeg_not_found"
            return
        cmd = [
            ff,
            "-loglevel", "quiet",
            "-hide_banner", "-nostats",
            "-rtsp_transport", "tcp",
            "-stimeout", str(int(self.timeout * 1_000_000)),  # microseconds
            "-i", rtsp,
            "-vf", f"fps={self.target_fps:.3f}",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-q:v", "3",
            "pipe:1",
        ]
        try:
            self._ffmpeg_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
                startupinfo=_si_no_window(),
                creationflags=_cf_no_window(),
            )
        except Exception as e:
            self._last_err = f"ffmpeg_spawn:{e}"
            return

        SOI = b"\xff\xd8"
        EOI = b"\xff\xd9"
        buf = bytearray()
        out = self._ffmpeg_proc.stdout
        if out is None:
            self._last_err = "ffmpeg_no_stdout"
            return

        try:
            while not self._stop.is_set():
                chunk = out.read(4096)
                if not chunk:
                    # process ended
                    rc = self._ffmpeg_proc.poll()
                    if rc is not None:
                        self._last_err = f"ffmpeg_rc:{rc}"
                        break
                    time.sleep(0.01)
                    continue
                buf.extend(chunk)
                # extract complete JPEGs
                while True:
                    i = buf.find(SOI)
                    if i < 0:
                        buf.clear()
                        break
                    j = buf.find(EOI, i + 2)
                    if j < 0:
                        # wait for more data
                        buf[:] = buf[i:]  # keep from SOI
                        break
                    j += 2
                    frame_bytes = bytes(buf[i:j])
                    del buf[:j]
                    if len(frame_bytes) < 1024:
                        continue
                    now_wall = time.time()
                    rf = RingFrame(ts_wall=now_wall, ts_mono=time.monotonic(),
                                   jpeg=frame_bytes, size=len(frame_bytes))
                    with self._lock:
                        self._buf.append(rf)
                    self._last_ok_ts = now_wall
                    self._last_err = None
        finally:
            try:
                if self._ffmpeg_proc:
                    self._ffmpeg_proc.kill()
            except Exception:
                pass
            self._ffmpeg_proc = None

    def stats(self) -> dict:
        with self._lock:
            n = len(self._buf)
            latest = self._buf[-1].ts_wall if n else 0.0
        return {
            "frames": n,
            "buffer_seconds": self.buffer_seconds,
            "target_fps": self.target_fps,
            "last_ok_ts": latest,
            "last_error": self._last_err,
        }

    def latest(self) -> Optional[RingFrame]:
        with self._lock:
            return self._buf[-1] if self._buf else None
    
    def _slice_indices(self, start_wall: float, end_wall: float) -> Tuple[int, int, List[RingFrame]]:
        with self._lock:
            frames = list(self._buf)
        if not frames:
            return 0, 0, []
        idx_start = 0
        idx_end = len(frames)
        for i, fr in enumerate(frames):
            if fr.ts_wall >= start_wall:
                idx_start = i
                break
        for j in range(idx_start, len(frames)):
            if frames[j].ts_wall > end_wall:
                idx_end = j
                break
        return idx_start, idx_end, frames

    def collect_window(self,
                       event_time: datetime,
                       pre_seconds: float = 3.0,
                       post_seconds: float = 2.0,
                       block: bool = False,
                       block_timeout: float = 3.0) -> List[RingFrame]:
        """
        Return frames within [event_time - pre, event_time + post].
        If block=True, wait up to block_timeout for post window to pass.
        """
        if event_time.tzinfo is None:
            event_wall = time.mktime(event_time.timetuple()) + event_time.microsecond / 1e6
        else:
            event_wall = event_time.timestamp()

        end_wall = event_wall + float(post_seconds)
        if block:
            t_deadline = time.monotonic() + max(0.0, float(block_timeout))
            while time.time() < end_wall and time.monotonic() < t_deadline and not self._stop.is_set():
                time.sleep(0.05)

        start_wall = event_wall - float(pre_seconds)
        _, _, frames = self._slice_indices(start_wall, end_wall)
        return [fr for fr in frames if start_wall <= fr.ts_wall <= end_wall]


# =====================================================================================
#  CLIP WORKER (async; ffmpeg pipe with atomic writes; manifest optional)
# =====================================================================================

@dataclass
class ClipJob:
    job_id: str
    cam_id: str
    event_id: int
    event_time: datetime
    object_id: Optional[str]
    obj_class: Optional[str]
    speed_kmh: Optional[float]
    pre_s: float
    post_s: float
    fps: float
    quality: str
    output_dir: str
    tmp_dir: str
    poster_jpeg: Optional[bytes] = None  # optional cover (not required)


@dataclass
class ClipResult:
    job_id: str
    event_id: int
    status: str  # 'ready' | 'failed'
    reason: Optional[str]
    clip_path: Optional[str]
    duration_s: Optional[float]
    fps: Optional[float]
    frames: int
    size_bytes: Optional[int]
    sha256: Optional[str]
    started_at: float
    finished_at: float
    meta: Dict[str, Any]


class ClipWorker:
    """
    Dedicated thread that turns ring-buffer JPEG frames into MP4 clips.
    Entirely asynchronous; main pipeline never blocks.
    """
    def __init__(self,
                 ring: VideoRingBuffer,
                 on_result: Optional[Callable[[ClipResult], None]] = None):
        self.ring = ring
        self.on_result = on_result
        self._q: "queue.Queue[ClipJob]" = queue.Queue(maxsize=256)
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None

        # status map by event_id and job_id
        self._status_by_id: Dict[str, ClipResult] = {}
        self._status_by_event: Dict[int, ClipResult] = {}
        self._lock = threading.Lock()

        # directories
        self._ensure_dir("clips")
        self._ensure_dir("work/clip_tmp")

    @staticmethod
    def _ensure_dir(path: str) -> None:
        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            pass

    @property
    def is_running(self) -> bool:
        return self._thr is not None and self._thr.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop.clear()
        self._thr = threading.Thread(target=self._run, name="ClipWorker", daemon=True)
        self._thr.start()

    def stop(self, join_timeout: float = 3.0) -> None:
        self._stop.set()
        if self._thr:
            self._thr.join(join_timeout)
        self._thr = None

    # ------------------ Public API ------------------

    def enqueue(self, job: ClipJob) -> None:
        self._q.put(job, block=False)

    def get_status_by_event(self, event_id: int) -> Optional[ClipResult]:
        with self._lock:
            return self._status_by_event.get(event_id)

    def get_status(self, job_id: str) -> Optional[ClipResult]:
        with self._lock:
            return self._status_by_id.get(job_id)

    # ------------------ Worker loop -----------------

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job: ClipJob = self._q.get(timeout=0.25)
            except queue.Empty:
                continue
            self._process(job)

    def _process(self, job: ClipJob) -> None:
        t_start = time.time()

        # Collect frames (block to include post window, but bounded)
        frames = self.ring.collect_window(
            job.event_time, pre_seconds=job.pre_s, post_seconds=job.post_s,
            block=True, block_timeout=max(0.0, job.post_s + 1.0)
        )

        if len(frames) < max(2, int(job.fps * 0.8)):  # need at least ~0.8s worth
            res = self._finalize(job, status="failed",
                                 reason=f"insufficient_frames:{len(frames)}",
                                 clip_path=None, duration_s=None, fps=None,
                                 frames=len(frames), size_bytes=None, sha256=None,
                                 started_at=t_start, finished_at=time.time(), meta={})
            self._record_and_emit(res)
            return

        # Sort frames by timestamp; optionally subsample to target FPS
        frames.sort(key=lambda fr: fr.ts_wall)
        duration = frames[-1].ts_wall - frames[0].ts_wall
        if duration <= 0:
            duration = max(job.pre_s + job.post_s, 0.5)

        # Downsample to target fps if needed
        step = max(1, int(round(len(frames) / max(1.0, duration * job.fps))))
        sampled = frames[::step] if step > 1 else frames
        out_name = self._build_clip_name(job, sampled, duration)

        tmp_path = os.path.join(job.tmp_dir, out_name)
        final_path = os.path.join(job.output_dir, out_name)
        self._ensure_dir(job.tmp_dir)
        self._ensure_dir(job.output_dir)

        _q = (getattr(job, "quality", None) or "medium").lower()
        _crf = {"low": 30, "medium": 24, "high": 20}.get(_q, 24)
        ok, err = self._encode_with_ffmpeg(sampled, tmp_path, job.fps, _crf)
        if not ok:
            if _HAS_CV2:
                ok2, err2 = self._encode_with_cv2(sampled, job.fps, tmp_path)
                if not ok2:
                    res = self._finalize(job, status="failed",
                                         reason=f"encode_error:{err}|fallback:{err2}",
                                         clip_path=None, duration_s=None, fps=None,
                                         frames=len(sampled), size_bytes=None, sha256=None,
                                         started_at=t_start, finished_at=time.time(), meta={})
                    self._record_and_emit(res)
                    return
            else:
                res = self._finalize(job, status="failed",
                                     reason=f"encode_error:{err}|fallback_unavailable",
                                     clip_path=None, duration_s=None, fps=None,
                                     frames=len(sampled), size_bytes=None, sha256=None,
                                     started_at=t_start, finished_at=time.time(), meta={})
                self._record_and_emit(res)
                return

        # Atomic move
        try:
            os.replace(tmp_path, final_path)
        except Exception as e:
            res = self._finalize(job, status="failed",
                                 reason=f"atomic_move_error:{e}",
                                 clip_path=None, duration_s=None, fps=None,
                                 frames=len(sampled), size_bytes=None, sha256=None,
                                 started_at=t_start, finished_at=time.time(), meta={})
            self._record_and_emit(res)
            return

        # Compute metadata
        size_b = os.path.getsize(final_path) if os.path.exists(final_path) else None
        sha = self._sha256_file(final_path) if size_b and size_b > 0 else None
        ts_dt = (job.event_time if isinstance(job.event_time, datetime)
                 else datetime.fromtimestamp(sampled[0].ts_wall))
        # unix seconds (safe even if ts_dt is naive)
        try:
            event_ts_unix = (ts_dt.timestamp()
                             if ts_dt.tzinfo is not None
                             else time.mktime(ts_dt.timetuple()) + ts_dt.microsecond / 1e6)
        except Exception:
            event_ts_unix = sampled[0].ts_wall
        bundle_date = ts_dt.strftime("%Y%m%d")
        # using seconds precision in the key to stay aligned with clip filename and DB times
        bundle_key = f"violation_{ts_dt.strftime('%Y%m%d_%H%M%S')}_{str(job.object_id) if job.object_id is not None else 'NA'}"

        res = self._finalize(
            job,
            status="ready",
            reason=None,
            clip_path=final_path,
            duration_s=round(duration, 2),
            fps=round(job.fps, 2),
            frames=len(sampled),
            size_bytes=size_b,
            sha256=sha,
            started_at=t_start,
            finished_at=time.time(),
            meta={
                "start_ts": sampled[0].ts_wall,
                "end_ts": sampled[-1].ts_wall,
                # add robust identifiers for DB reconciliation
                "object_id": job.object_id,
                "obj_class": job.obj_class,
                "speed_kmh": int(round(job.speed_kmh)) if isinstance(job.speed_kmh, (int, float)) else None,
                "event_time": (job.event_time.isoformat() if isinstance(job.event_time, datetime)
                               else datetime.fromtimestamp(sampled[0].ts_wall).isoformat()),
                "event_ts_unix": event_ts_unix,
                "bundle_key": bundle_key,
                "bundle_date": bundle_date,
                "cam_id": job.cam_id
            }
        )
        self._write_manifest(res)  # optional manifest next to the clip
        self._record_and_emit(res)

    # ------------------ Helpers ---------------------

    @staticmethod
    def _sha256_file(path: str) -> Optional[str]:
        try:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None

    def _build_clip_name(self, job: ClipJob, frames: List[RingFrame], duration: float) -> str:
        # Name: violation_YYYYMMDD_HHMMSS_<event>_<sanitized_obj>_<class>_<speed>.mp4
        # Use the event_time to align with DB timestamps; fall back to first frame ts.
        ts_dt = job.event_time if isinstance(job.event_time, datetime) else datetime.fromtimestamp(frames[0].ts_wall)
        ts = ts_dt.strftime("%Y%m%d_%H%M%S")
        speed_tag = f"{int(round(job.speed_kmh))}kmh" if job.speed_kmh is not None else "na"
        obj_raw = str(job.object_id) if job.object_id is not None else "NA"
        # sanitize: keep [A-Za-z0-9-], replace others (including underscores/spaces) with '-'
        obj_tag = re.sub(r"[^A-Za-z0-9\-]+", "-", obj_raw)
        cls_tag = (job.obj_class or "NA").upper()
        cam_tag = re.sub(r"[^A-Za-z0-9\-]+", "-", str(job.cam_id))
        return f"violation_{ts}_{job.event_id}_{obj_tag}_{cls_tag}_{speed_tag}_CAM-{cam_tag}.mp4"

    def _encode_with_ffmpeg(self, frames: List["RingFrame"], out_path, fps, crf=23):
        """
        Encode list of JPEG bytes to H.264-in-MP4 using ffmpeg via stdin.
        Hardens for web playback:
        - even-size guarantee (scale filter)
        - moov atom at head (+faststart)
        - clean stdin close + wait
        - header sanity check (ftyp)
        """
        import subprocess, shutil, math, hashlib, time, os, tempfile

        # Backward-compat / robustness: accept old call pattern (frames, fps, out_path, quality)
        # or a string-quality instead of CRF.
        if isinstance(out_path, (int, float)) and isinstance(fps, str):
            # called as (frames, fps, out_path, quality)
            out_path, fps, crf = fps, out_path, crf
        if isinstance(crf, str):
            crf = {"low": 30, "medium": 24, "high": 20}.get(crf.lower(), 24)
        fps = float(fps)
        out_path = str(out_path)

        if not frames:
            return False, "no_frames"
        # Force even dimensions for yuv420p (handled purely inside ffmpeg)
        # libx264 + yuv420p requires width/height divisible by 2
        scale_filter = "scale=trunc(iw/2)*2:trunc(ih/2)*2"

        ffmpeg_bin = _ffmpeg_bin()
        if not ffmpeg_bin:
            return False, "ffmpeg_not_found"

        # Sane GOP for short clips: ~2s
        g = max(1, int(round(fps * 2)))

        # Write to a temp path, then atomically replace
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp_path = out_path + ".part"

        args = [
            ffmpeg_bin, "-loglevel", "quiet", "-hide_banner", "-y",
            "-f", "image2pipe", "-vcodec", "mjpeg", "-r", str(fps),  # input stream timing
            "-i", "pipe:0",
            "-vf", scale_filter,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-g", str(g), "-keyint_min", str(g),
            "-movflags", "+faststart",
            "-f", "mp4", tmp_path,
        ]

        try:
            proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            # Write all JPEG frames
            proc_stdin = proc.stdin
            for fr in frames:
                if not fr or not getattr(fr, "jpeg", None):
                    continue
                try:
                    proc_stdin.write(fr.jpeg)
                except BrokenPipeError:
                    # Encoder already errored out
                    break

            # Very important: close stdin so ffmpeg can finalize moov
            try:
                proc_stdin.close()
            except Exception:
                pass

            # Wait for completion
            try:
                _, err = proc.communicate(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                return False, "ffmpeg_timeout"

            if proc.returncode != 0:
                # bubble up ffmpeg error text for logs
                return False, f"ffmpeg_exit_{proc.returncode}: {(err or b'').decode('utf-8', 'ignore').strip()}"

            # Quick validity checks before we mark ready
            if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) < 40_000:
                return False, "tiny_or_missing_output"

            # Check MP4 brand header: should contain 'ftyp'
            with open(tmp_path, "rb") as f:
                head = f.read(16)
            if b"ftyp" not in head:
                return False, "invalid_mp4_header"

            # All good → make it final
            os.replace(tmp_path, out_path)
            return True, None

        except Exception as e:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            return False, f"ffmpeg_exception:{e}"

    def _encode_with_cv2(self, frames: List[RingFrame], fps: float, out_path: str) -> Tuple[bool, Optional[str]]:
        """
        Fallback encoder: decode each JPEG and write via cv2.VideoWriter (MJPG in MP4 may be large).
        """
        if not _HAS_CV2:
            return False, "cv2_unavailable"

        # Decode first frame to get size
        try:
            import numpy as np  # type: ignore
        except Exception:
            return False, "numpy_missing"

        try:
            first = frames[0].jpeg
            img0 = cv2.imdecode(np.frombuffer(first, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img0 is None:
                return False, "decode_fail_first"
            h, w = img0.shape[:2]

            # FourCC: fallback to mp4v; some stacks accept MJPG as well
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            if not vw.isOpened():
                return False, "videowriter_open_fail"

            vw.write(img0)
            for fr in frames[1:]:
                img = cv2.imdecode(np.frombuffer(fr.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None and img.shape[0] == h and img.shape[1] == w:
                    vw.write(img)
            vw.release()
            return True, None
        except Exception as e:
            return False, f"cv2_encode:{e}"

    def _finalize(self, job: ClipJob, **kwargs) -> ClipResult:
        return ClipResult(
            job_id=job.job_id,
            event_id=job.event_id,
            status=kwargs.get("status", "failed"),
            reason=kwargs.get("reason"),
            clip_path=kwargs.get("clip_path"),
            duration_s=kwargs.get("duration_s"),
            fps=kwargs.get("fps"),
            frames=kwargs.get("frames", 0),
            size_bytes=kwargs.get("size_bytes"),
            sha256=kwargs.get("sha256"),
            started_at=kwargs.get("started_at", 0.0),
            finished_at=kwargs.get("finished_at", time.time()),
            meta=kwargs.get("meta", {}),
        )

    def _write_manifest(self, res: ClipResult) -> None:
        try:
            if not res.clip_path:
                return
            manifest = {
                "job_id": res.job_id,
                "event_id": res.event_id,
                "status": res.status,
                "clip_path": res.clip_path,
                "duration_s": res.duration_s,
                "fps": res.fps,
                "frames": res.frames,
                "size_bytes": res.size_bytes,
                "sha256": res.sha256,
                "started_at": res.started_at,
                "finished_at": res.finished_at,
                "meta": res.meta,
            }
            mpath = res.clip_path.rsplit(".", 1)[0] + ".json"
            with open(mpath, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[CLIP MANIFEST WARN] {e}")

    def _record_and_emit(self, res: ClipResult) -> None:
        with self._lock:
            self._status_by_id[res.job_id] = res
            self._status_by_event[res.event_id] = res
        if self.on_result:
            try:
                self.on_result(res)
            except Exception as e:
                print(f"[CLIP CALLBACK ERROR] {e}")


# =====================================================================================
#  GLOBALS & PUBLIC API
# =====================================================================================

_global_ring: Optional[VideoRingBuffer] = None
_global_worker: Optional[ClipWorker] = None
_global_cfg: Dict[str, Any] = {}
_global_callback: Optional[Callable[[ClipResult], None]] = None
_pipelines: Dict[str, Dict[str, Any]] = {}

def init_video_pipeline(camera_url: str,
                        username: Optional[str] = None,
                        password: Optional[str] = None,
                        *,
                        target_fps: float = 8.0,
                        pre_seconds: float = 3.0,
                        post_seconds: float = 2.0,
                        buffer_margin: float = 6.0,
                        timeout: float = 3.0,
                        verify: bool = False,
                        output_dir: str = "clips",
                        tmp_dir: str = "work/clip_tmp",
                        quality: str = "medium",
                        on_clip_result: Optional[Callable[[ClipResult], None]] = None) -> None:
    """
    Initialize the ring-buffer reader and the async clip worker.
    - buffer keeps (pre + post + margin) seconds
    - on_clip_result: optional callback to update DB / trigger Drive sync
    """
    global _global_ring, _global_worker, _global_cfg, _global_callback
    init_camera_pipeline(
        cam_id="default",
        camera_url=camera_url,
        username=username,
        password=password,
        target_fps=target_fps,
        pre_seconds=pre_seconds,
        post_seconds=post_seconds,
        buffer_margin=buffer_margin,
        timeout=timeout,
        verify=verify,
        output_dir=output_dir,
        tmp_dir=tmp_dir,
        quality=quality,
        on_clip_result=on_clip_result,
    )
    # Mirror legacy globals for existing callers
    _set_legacy_globals_from_registry("default")
    print(f"[VIDEO PIPELINE] (default) src={camera_url} fps={target_fps}")

def _set_legacy_globals_from_registry(cam_id: str) -> None:
    global _global_ring, _global_worker, _global_cfg, _global_callback
    meta = _pipelines.get(cam_id) or {}
    _global_ring = meta.get("ring")
    _global_worker = meta.get("worker")
    _global_cfg = meta.get("cfg", {})
    _global_callback = meta.get("callback")

def init_camera_pipeline(cam_id: str,
                         camera_url: str,
                         username: Optional[str] = None,
                         password: Optional[str] = None,
                         *,
                         target_fps: float = 8.0,
                         pre_seconds: float = 3.0,
                         post_seconds: float = 2.0,
                         buffer_margin: float = 6.0,
                         timeout: float = 3.0,
                         verify: bool = False,
                         output_dir: str = "clips",
                         tmp_dir: str = "work/clip_tmp",
                         quality: str = "medium",
                         on_clip_result: Optional[Callable[[ClipResult], None]] = None) -> None:
    """
    per-camera initialization. Safe to call for many cameras.
    """
    total_buf = int(math.ceil(pre_seconds + post_seconds + max(0.0, buffer_margin)))
    ring = VideoRingBuffer(camera_url, username, password,
                          target_fps=target_fps,
                           buffer_seconds=total_buf,
                           timeout=timeout, verify=verify)
    ring.start()
    worker = ClipWorker(ring, on_result=on_clip_result)
    worker.start()
    cfg = {
        "pre_seconds": float(pre_seconds),
        "post_seconds": float(post_seconds),
        "fps": float(target_fps),
        "quality": quality,
        "output_dir": output_dir,
        "tmp_dir": tmp_dir,
    }
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    _pipelines[cam_id] = {"ring": ring, "worker": worker, "cfg": cfg, "callback": on_clip_result, "src": camera_url}
    print(f"[VIDEO PIPELINE] cam={cam_id} ring@{target_fps}fps buf={total_buf}s | out={output_dir} | src={camera_url}")

def request_violation_clip(event_id: int,
                           event_time: datetime,
                           object_id: Optional[str],
                           obj_class: Optional[str],
                           speed_kmh: Optional[float]) -> Optional[str]:
    """
    Enqueue a non-blocking clip job. Returns job_id if queued.
    """
    return request_violation_clip_multi("default", event_id, event_time, object_id, obj_class, speed_kmh)

_RECENT_CLIP_REQ: Dict[Tuple[str,int,str], float] = {}
_RECENT_CLIP_WINDOW_S = 2.0
def request_violation_clip_multi(cam_id: str,
                                 event_id: int,
                                 event_time: datetime,
                                 object_id: Optional[str],
                                 obj_class: Optional[str],
                                 speed_kmh: Optional[float]) -> Optional[str]:
    meta = _pipelines.get(cam_id)
    if not meta:
        print(f"[CLIP REQUEST] cam '{cam_id}' not initialized")
        return None
    worker: ClipWorker = meta["worker"]
    cfg = meta.get("cfg", {})
    job_id = f"{cam_id}-{event_id}_{int(time.time()*1000)}"
    try:
       sec_ts = int(event_time.timestamp()) if hasattr(event_time, "timestamp") else int(event_time)
    except Exception:
        sec_ts = int(time.time())
    key = (str(cam_id), sec_ts, str(object_id) if object_id is not None else "NA")
    now_mono = time.monotonic()
    last = _RECENT_CLIP_REQ.get(key)
    if last and (now_mono - last) < _RECENT_CLIP_WINDOW_S:
        return None
    _RECENT_CLIP_REQ[key] = now_mono
    job = ClipJob(
        job_id=job_id,
        cam_id=str(cam_id),
        event_id=event_id,
        event_time=event_time,
        object_id=object_id,
        obj_class=obj_class,
        speed_kmh=speed_kmh,
        pre_s=float(cfg.get("pre_seconds", 3.0)),
        post_s=float(cfg.get("post_seconds", 2.0)),
        fps=float(cfg.get("fps", 8.0)),
        quality=str(cfg.get("quality", "medium")),
        output_dir=str(cfg.get("output_dir", "clips")),
        tmp_dir=str(cfg.get("tmp_dir", "work/clip_tmp")),
    )
    try:
        worker.enqueue(job)
        return job_id
    except queue.Full:
        print(f"[CLIP REQUEST] queue full (cam={cam_id})")
        return None

def get_clip_status_by_event(event_id: int) -> Optional[Dict[str, Any]]:
    """
    Returns status dict for UI/API: {status, path, size, sha256, ...}
    """
    if _global_worker is None:
        return None
    res = _global_worker.get_status_by_event(event_id)
    if not res:
        return None
    return {
        "event_id": res.event_id,
        "job_id": res.job_id,
        "status": res.status,
        "reason": res.reason,
        "clip_path": res.clip_path,
        "duration_s": res.duration_s,
        "fps": res.fps,
        "frames": res.frames,
        "size_bytes": res.size_bytes,
        "sha256": res.sha256,
        "started_at": res.started_at,
        "finished_at": res.finished_at,
        "meta": res.meta,
    }


def get_ring_stats() -> Optional[Dict[str, Any]]:
    if _global_ring is None:
        # prefer default cam registry if available
        meta = _pipelines.get("default")
        if meta:
            return meta["ring"].stats()
        return None
    return _global_ring.stats()

def get_ring_stats_multi(cam_id: str) -> Optional[Dict[str, Any]]:
    meta = _pipelines.get(cam_id)
    if not meta:
        return None
    return meta["ring"].stats()

def shutdown_video_pipeline() -> None:
    """
    Graceful stop (use during service shutdown).
    """
    global _global_ring, _global_worker
    # stop legacy single pipeline if set
    try:
        if _global_worker:
            _global_worker.stop()
    except Exception:
        pass
    try:
        if _global_ring:
            _global_ring.stop()
    except Exception:
        pass
    _global_ring = None
    _global_worker = None
    # stop all multi-cam pipelines
    for cam_id, meta in list(_pipelines.items()):
        try:
            w: ClipWorker = meta.get("worker")
            r: VideoRingBuffer = meta.get("ring")
            if w:
                try: w.stop()
                except Exception: pass
            if r:
                try: r.stop()
                except Exception: pass
        finally:
            _pipelines.pop(cam_id, None)