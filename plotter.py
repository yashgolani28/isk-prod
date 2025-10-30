import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.patheffects as pe
from uuid import uuid4
from collections import defaultdict, deque
import numpy as np
import threading
import json
import time
import stat
import os
import tempfile
import shutil
import cv2

THEME = {
    "BG": "#0F172A",
    "FG": "#E5E7EB",
    "GRID": "#1F2937",
    "AX": "#94A3B8",
}
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.facecolor": THEME["BG"],
    "figure.facecolor": THEME["BG"],
    "axes.edgecolor": THEME["AX"],
    "xtick.color": THEME["AX"],
    "ytick.color": THEME["AX"],
})

class Live3DPlotter:
    """
    TM-style visualizer (no heatmaps).
    Call update(tracked_objects, points=None).
    """
    def __init__(self, az_fov_deg=120, el_fov_deg=40, sensor_height_m=2.0,
                 frame_sink=None, ws_emit_hz: int = 12, ws_quality: int = 80):
        self.lock = threading.Lock()
        self.running = True
        self.static_dir = os.environ.get("ISK_STATIC_DIR", os.path.join(os.path.dirname(__file__), "static"))
        os.makedirs(self.static_dir, exist_ok=True)
        self.STD_FIGSIZE = (8.0, 6.0)
        self.FULL_FIGSIZE = (14.0, 9.0)
        self.CUBE_ASPECT = True
        self.fig = plt.figure(figsize=self.STD_FIGSIZE, dpi=110)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.patch.set_facecolor(THEME["BG"])
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # A slightly closer, nicer viewing angle
        try:
            self.ax.view_init(elev=24, azim=-42)
        except Exception:
            pass
        self._style_axes()
        # TM params
        self.az_fov = float(az_fov_deg)
        self.el_fov = float(el_fov_deg)
        self.h = float(sensor_height_m)
        # scene data
        self.objects = []     # list of dicts with x,y,z,vx,vy,vz,type,speed
        self.points = None    # optional Nx3
        self.points2d = None  # optional Nx2
        self.lanes = []       # list of {'id', 'rect':[x1,y1,x2,y2], 'count', 'color'}
        self.zones = []       # optional polygons (not required)
        self.tracks2d = []    # cached 2-D projections for TM 2D panes
        self.tm_counts = {}   # optional per-lane counts (from tm_viz["stats"])
        self.trails = defaultdict(lambda: deque(maxlen=50))  # tid -> deque[(t, x, y)]
        self.trail_maxlen = 50
        self.trail_maxtime = 6.0  # seconds
        self._last_prune = 0.0
        # IO cadence
        self._last_scatter_save = 0.0
        self._scatter_save_every = 0.08
        self._tm2d_save_every = 0.08
        self._last_tm2d_save = 0.0
        self._last_scatter_had_content = False
        self._last_tm2d_had_content = False
        self._draw_lanes_in_3d = False
        self._frame_sink = frame_sink            # callable(kind: str, jpg_bytes: bytes)
        self._ws_quality = int(ws_quality)       # JPEG quality [0..100]
        self._ws_emit_every = 1.0 / max(1, int(ws_emit_hz))
        self._last_ws_emit = 0.0
        # start renderer
        self.thread = threading.Thread(target=self._render_loop, daemon=True)
        self.thread.start()
        self._lane_defaults = {
            "num_lanes": 4,
            "lane_width_m": 10,
            "center_x_m": 0.0,
            "start_y_m": 0.0,
            "end_y_m":   80.0,
        }
        # Full-view (enlarged) output settings for precision tracking
        cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
        _cfg = {}
        try:
            with open(cfg_path, "r") as _f:
                _cfg = json.load(_f) or {}
        except Exception:
            _cfg = {}
        # standard (dashboard card) limits — tighter → objects appear larger
        self.std_xlim = (float(_cfg.get("viz_range_x_m_std_min", -20.0)),
                         float(_cfg.get("viz_range_x_m_std_max",  20.0)))
        self.std_ylim = (0.0, float(_cfg.get("viz_range_y_m_std_max", 80.0)))
        self.std_zlim = (0.0, float(_cfg.get("viz_range_z_m_std_max", 6.0)))
        # full-view limits (slightly wider field but precise ticks)
        self.full_xlim = (float(_cfg.get("viz_range_x_m_full_min", -20.0)),
                          float(_cfg.get("viz_range_x_m_full_max",  20.0)))
        self.full_ylim = (0.0, float(_cfg.get("viz_range_y_m_full_max", 80.0)))
        self.full_zlim = (0.0, float(_cfg.get("viz_range_z_m_full_max", 6.0)))
        # defensive: if any got lost in older runtime, set sane defaults
        self.full_xlim = getattr(self, "full_xlim", (-20.0, 20.0))
        self.full_ylim = getattr(self, "full_ylim", (0.0, 80.0))
        self.full_zlim = getattr(self, "full_zlim", (0.0, 6.0))
        self._tm2d_full_path = os.path.join(self.static_dir, "tm2d_full.png")
        self._scatter_full_name = "scatter_3d_full.png" 
        self.tm2d_full_xlim = (-20, 20)
        self.tm2d_full_ylim = (0, 80)
        self._tm_stats_path = os.path.join(self.static_dir, "tm_stats.json")

        # ---- 2-D TM panes (two tall top-down subplots) ----
        self.fig2 = plt.figure(figsize=(10.0, 6.4), dpi=110)
        gs = self.fig2.add_gridspec(1, 2, wspace=0.15)
        self.ax_gate = self.fig2.add_subplot(gs[0, 0])   # "Gating & Association"
        self.ax_count = self.fig2.add_subplot(gs[0, 1])  # "Counting / Trails"
        for a in (self.ax_gate, self.ax_count):
            a.set_facecolor(THEME["BG"])
            a.grid(True, color=THEME["GRID"], alpha=0.4, linewidth=0.5)
            a.set_xlabel("X (m)", color=THEME["FG"])
            a.set_ylabel("Y (m)", color=THEME["FG"])
            a.set_xlim(-20, 20)
            a.set_ylim(0, 80)
            a.tick_params(colors=THEME["AX"])
        self.fig2.patch.set_facecolor(THEME["BG"])
        self.fig2.suptitle("Top-Down", color=THEME["FG"])
        self._tm2d_path = os.path.join(self.static_dir, "tm2d.png")

        try:
            os.makedirs(self.static_dir, exist_ok=True)
            # std (write to temp then atomically replace)
            self.fig.set_size_inches(*self.STD_FIGSIZE, forward=True)
            self._apply_limits(self.ax, full=False)
            self._apply_box_aspect(self.ax, full=False)
            fd_s, tmp_s = tempfile.mkstemp(prefix=".scatter_", suffix=".png", dir=self.static_dir); os.close(fd_s)
            self.fig.savefig(tmp_s, dpi=100, bbox_inches="tight", facecolor=self.fig.get_facecolor())
            self._replace_atomic(tmp_s, os.path.join(self.static_dir, "scatter_3d.png"))
            # full (temp → atomic replace)
            self.fig.set_size_inches(*self.FULL_FIGSIZE, forward=True)
            self._apply_limits(self.ax, full=True)
            self._apply_box_aspect(self.ax, full=True)
            fd_f, tmp_f = tempfile.mkstemp(prefix=".scatter_full_", suffix=".png", dir=self.static_dir); os.close(fd_f)
            self.fig.savefig(tmp_f, dpi=180, bbox_inches="tight", facecolor=self.fig.get_facecolor())
            self._replace_atomic(tmp_f, os.path.join(self.static_dir, self._scatter_full_name))
            # reset to std for runtime
            self.fig.set_size_inches(*self.STD_FIGSIZE, forward=True)
            self._apply_limits(self.ax, full=False)
            self._apply_box_aspect(self.ax, full=False)

            # full TM: tighter ticks (5 m) for precision view
            a_xlim, a_ylim = self.ax_gate.get_xlim(), self.ax_gate.get_ylim()
            b_xlim, b_ylim = self.ax_count.get_xlim(), self.ax_count.get_ylim()
            self.ax_gate.set_xlim(*self.tm2d_full_xlim); self.ax_gate.set_ylim(*self.tm2d_full_ylim)
            self.ax_count.set_xlim(*self.tm2d_full_xlim); self.ax_count.set_ylim(*self.tm2d_full_ylim)
            self.ax_gate.set_xticks(np.arange(self.tm2d_full_xlim[0], self.tm2d_full_xlim[1]+0.1, 5))
            self.ax_gate.set_yticks(np.arange(self.tm2d_full_ylim[0], self.tm2d_full_ylim[1]+0.1, 5))
            self.ax_count.set_xticks(np.arange(self.tm2d_full_xlim[0], self.tm2d_full_xlim[1]+0.1, 5))
            self.ax_count.set_yticks(np.arange(self.tm2d_full_ylim[0], self.tm2d_full_ylim[1]+0.1, 5))

            self.fig2.savefig(self._tm2d_full_path,
                              dpi=180, bbox_inches="tight", facecolor=self.fig2.get_facecolor())
            self.ax_gate.set_xticks(np.arange(a_xlim[0], a_xlim[1]+0.1, 10))
            self.ax_gate.set_yticks(np.arange(a_ylim[0], a_ylim[1]+0.1, 10))
            self.ax_count.set_xticks(np.arange(b_xlim[0], b_xlim[1]+0.1, 10))
            self.ax_count.set_yticks(np.arange(b_ylim[0], b_ylim[1]+0.1, 10))
        except Exception:
            pass

        self.thread2d = threading.Thread(target=self._render_loop_2d, daemon=True)
        self.thread2d.start()

# ───────────────────────── Atomic replace helper ─────────────────────────
    def _replace_atomic(self, src: str, dst: str, attempts: int = 8, backoff: float = 0.12) -> str:
        """
        Atomically replace dst with src, retrying on Windows PermissionError.
        Returns the final path written (dst, or a timestamped fallback).
        """
        for i in range(attempts):
            try:
                if os.path.exists(dst):
                    try:
                        os.chmod(dst, stat.S_IWRITE)
                    except Exception:
                        pass
                os.replace(src, dst)  # atomic on Windows/POSIX
                return dst
            except PermissionError:
                time.sleep(backoff * (i + 1))
            except Exception:
                break
        base, ext = os.path.splitext(dst)
        alt = f"{base}_{int(time.time())}{ext}"
        try:
            os.replace(src, alt)
            print(f"[SCATTER SAVE WARN] destination locked; wrote fallback {alt}")
            return alt
        except Exception as e:
            print(f"[SCATTER SAVE ERROR] fallback failed: {e}")
            # last resort: move (non-atomic)
            try:
                shutil.move(src, alt)
                return alt
            except Exception:
                return dst

    # ───────────────────────── Internal helpers ─────────────────────────
    def _apply_limits(self, ax, *, full: bool):
        if full:
            ax.set_xlim(*self.full_xlim); ax.set_ylim(*self.full_ylim); ax.set_zlim(*self.full_zlim)
        else:
            ax.set_xlim(*self.std_xlim);  ax.set_ylim(*self.std_ylim);  ax.set_zlim(*self.std_zlim)

    def _apply_box_aspect(self, ax, *, full: bool):
        # keep visually correct cube proportions
        if self.CUBE_ASPECT:
            ax.set_box_aspect((1, 1, 1))
        else:
            # fallback to proportional box aspect
            if full:
                ax.set_box_aspect((self.full_xlim[1]-self.full_xlim[0],
                                   self.full_ylim[1]-self.full_ylim[0],
                                   self.full_zlim[1]-self.full_zlim[0]))
            else:
                ax.set_box_aspect((self.std_xlim[1]-self.std_xlim[0],
                                   self.std_ylim[1]-self.std_ylim[0],
                                   self.std_zlim[1]-self.std_zlim[0]))

    def _save_scatter_images(self, points):
        """
        Render & save BOTH images:
          - static/scatter_3d.png      (standard card)
          - static/scatter_3d_full.png (full view; separate limits & DPI)
        """
        # Draw points once; save twice with distinct limits/DPI.
        self.ax.cla()
        self._style_axes()
        # standard first (bigger, tighter scale on dashboard)
        self._apply_limits(self.ax, full=False)
        self._apply_box_aspect(self.ax, full=False)
        if points is not None and len(points):
            pts = np.asarray(points, dtype=float)
            self.ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=22, depthshade=False, alpha=0.9)
        self.fig.set_size_inches(*self.STD_FIGSIZE, forward=True)
        fd_s, tmp_s = tempfile.mkstemp(prefix=".scatter_", suffix=".png", dir=self.static_dir); os.close(fd_s)
        self.fig.savefig(tmp_s, dpi=110, bbox_inches="tight", facecolor=self.fig.get_facecolor())
        self._replace_atomic(tmp_s, os.path.join(self.static_dir, "scatter_3d.png"))
        # full view: wider Y range, higher DPI; never overwrite std file
        self.ax.cla()
        self._style_axes()
        self._apply_limits(self.ax, full=True)
        self._apply_box_aspect(self.ax, full=True)
        if points is not None and len(points):
            pts = np.asarray(points, dtype=float)
            self.ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=22, depthshade=False, alpha=0.9)
        self.fig.set_size_inches(*self.FULL_FIGSIZE, forward=True)
        fd_f, tmp_f = tempfile.mkstemp(prefix=".scatter_full_", suffix=".png", dir=self.static_dir); os.close(fd_f)
        self.fig.savefig(tmp_f, dpi=180, bbox_inches="tight", facecolor=self.fig.get_facecolor())
        self._replace_atomic(tmp_f, os.path.join(self.static_dir, self._scatter_full_name))
        self.fig.set_size_inches(*self.STD_FIGSIZE, forward=True)

    def update(self, tracked_objects, points=None):
        """TM entrypoint: tracks + optional point cloud (Nx3)."""
        with self.lock:
            self.objects = []
            for o in tracked_objects or []:
                if all(k in o for k in ("x", "y", "z")):
                    obj = {
                        "x": float(o.get("x", 0.0)),
                        "y": float(o.get("y", 0.0)),
                        "z": float(o.get("z", 0.0)),
                        "vx": float(o.get("velX", 0.0)),
                        "vy": float(o.get("velY", 0.0)),
                        "vz": float(o.get("velZ", 0.0)),
                        "type": str(o.get("type", "UNKNOWN")),
                        "speed": float(o.get("speed_kmh", o.get("speed", 0.0))),
                        "tid": str(o.get("tid", "")),
                    }
                    self.objects.append(obj)
                    # ── update TM visualizer trail buffers (2-D) ────────────
                    if obj["tid"]:
                        now = time.time()
                        dq = self.trails[obj["tid"]]
                        dq.append((now, obj["x"], obj["y"]))
            # prune trail buffers periodically
            now = time.time()
            if now - self._last_prune > 0.5:
                cutoff = now - self.trail_maxtime
                dead = []
                for tid, dq in self.trails.items():
                    while dq and dq[0][0] < cutoff:
                        dq.popleft()
                    if not dq:
                        dead.append(tid)
                for tid in dead:
                    del self.trails[tid]
                self._last_prune = now
            self.tracks2d = [{"x":o["x"], "y":o["y"], "type":o["type"], "tid":o["tid"], "speed":o["speed"]} for o in self.objects]
            if points is not None:
                pts = np.asarray(points, dtype=np.float32)
                if pts.ndim == 2 and pts.shape[1] >= 3:
                    self.points = pts[:, :3]
                    self.points2d = self.points[:, :2]
                else:
                    self.points = None
                    self.points2d = None

            if not self.lanes:
                self.lanes = self._build_default_lanes()
            # Refresh live occupancy counts (heads currently inside each lane).
            if self.lanes:
                for L in self.lanes:
                    rect = L["rect"]
                    L["count"] = sum(
                        1 for o in self.tracks2d
                        if self._point_in_rect(o["x"], o["y"], rect)
                   )

    def update_tm(self, tm_viz: dict | None):
        """
        Accepts a structure like parsers emit:
          tm_viz = {
            'lanes': [{'id':1, 'rect':[x1,y1,x2,y2], 'count':3}, ...],
            'zones': [{'name':'gate', 'poly':[[x,y],...]}],   # optional
            'stats': {'lanes': {...}}                         # optional
          }
        Coordinates are expected in meters in radar X/Y on ground plane.
        """
        if tm_viz is None:
            return
        with self.lock:
            lanes = []
            for L in tm_viz.get('lanes', []) or []:
                rect = L.get('rect') or L.get('bbox') or None
                if rect and len(rect) == 4:
                    lanes.append({
                        "id": str(L.get('id', '')),
                        "rect": [float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])],
                        "count": int(L.get('count', 0)),
                        "color": "#60A5FA",  # sky-ish
                    })
            self.lanes = lanes
            zones = []
            for Z in tm_viz.get('zones', []) or []:
                poly = Z.get('poly') or Z.get('polygon')
                if isinstance(poly, (list, tuple)) and len(poly) >= 3:
                    zones.append({"name": str(Z.get('name',"zone")), "poly": np.asarray(poly, dtype=np.float32)})
            self.zones = zones
            self.tm_counts = (tm_viz.get("stats", {}) or {}).get("lanes", {})
            if not self.lanes:
                self.lanes = self._build_default_lanes()

    def update_heatmap(self, *_a, **_k):
        return

    def _draw_fov(self, r=50.0):
        # simple azimuth fan on ground plane + boresight line
        az = np.radians(np.linspace(-self.az_fov/2, self.az_fov/2, 64))
        y = np.cos(az) * r
        x = np.sin(az) * r
        z = np.zeros_like(x) + self.h
        self.ax.plot([0, 0], [0, r], [self.h, self.h], lw=1, color=THEME["AX"], ls='-')
        self.ax.plot(x, y, z, lw=1, color=THEME["AX"], ls='--')
        # ground
        self.ax.plot([-r, r], [0, 0], [0, 0], lw=1, color=THEME["AX"], alpha=0.5)

    def stop(self):
        self.running = False
        try:
            self.thread.join(timeout=1.0)
        except Exception:
            pass
        plt.close(self.fig)

    def _render_loop(self):
        while self.running:
            try:
                self._render_once()
            except Exception as e:
                print(f"[PLOTTER ERROR] render loop: {e}")
            time.sleep(0.1)

    def _render_once(self):
        with self.lock:
            self.ax.clear()
            self._style_axes()
            had_content = False
            if self.points is not None and len(self.points) > 0:
                try:
                    self.ax.scatter(
                        self.points[:, 0], self.points[:, 1], self.points[:, 2],
                        s=4, alpha=0.35, depthshade=True
                    )
                    had_content = True
                except Exception:
                    pass
            self._draw_tm_overlays()
            for o in self.objects:
                c = self._color(o["type"])
                size = 40 + min(160, int(o["speed"] * 3.0))
                self.ax.scatter(o["x"], o["y"], o["z"], c=c, s=size, depthshade=True)
                had_content = True
                # keep 3D view clean (no trails); just annotate speed/type/TID
                tid_tag = f" [{o['tid']}]" if o.get("tid") else ""
                txt = self.ax.text(o["x"], o["y"], o["z"],
                                   f"{o['type']}{tid_tag} • {o['speed']:.1f} km/h",
                                   fontsize=8, color=THEME["FG"])
                try:
                    txt.set_path_effects([pe.Stroke(linewidth=2, foreground="#000000"), pe.Normal()])
                except Exception:
                    pass
                if o["vx"] or o["vy"] or o["vz"]:
                    self.ax.quiver(o["x"], o["y"], o["z"], o["vx"], o["vy"], o["vz"], length=0.5, normalize=True, color=c, linewidth=1.4)
        now = time.time()
        if now - self._last_scatter_save >= self._scatter_save_every:
            tmp_path = None
            try:
                os.makedirs(self.static_dir, exist_ok=True)
                if had_content or not self._last_scatter_had_content:
                    # write standard view
                    self.fig.set_size_inches(*self.STD_FIGSIZE, forward=True)
                    fd, tmp_path = tempfile.mkstemp(prefix=".scatter_", suffix=".png", dir=self.static_dir)
                    os.close(fd)
                    self.fig.savefig(tmp_path, dpi=110, bbox_inches="tight", facecolor=self.fig.get_facecolor())
                    self._replace_atomic(tmp_path, os.path.join(self.static_dir, "scatter_3d.png"))
                    # write full view (fixed limits, hi-DPI)
                    prev_xlim, prev_ylim, prev_zlim = self.ax.get_xlim(), self.ax.get_ylim(), self.ax.get_zlim()
                    self.ax.set_xlim(*self.full_xlim); self.ax.set_ylim(*self.full_ylim); self.ax.set_zlim(*self.full_zlim)
                    self.fig.set_size_inches(*self.FULL_FIGSIZE, forward=True)
                    fd2, tmp_path2 = tempfile.mkstemp(prefix=".scatter_full_", suffix=".png", dir=self.static_dir)
                    os.close(fd2)
                    self.fig.savefig(tmp_path2, dpi=180, bbox_inches="tight", facecolor=self.fig.get_facecolor())
                    self._replace_atomic(tmp_path2, os.path.join(self.static_dir, self._scatter_full_name))
                    self.ax.set_xlim(*prev_xlim); self.ax.set_ylim(*prev_ylim); self.ax.set_zlim(*prev_zlim)
                    self.fig.set_size_inches(*self.STD_FIGSIZE, forward=True)
                    self._last_scatter_had_content = had_content
                    self._last_scatter_save = now
                    
                    tmp_full = None
                    try:
                        self.ax.set_xlim(*self.full_xlim)
                        self.ax.set_ylim(*self.full_ylim)
                        self.ax.set_zlim(*self.full_zlim)
                        try:
                            self.ax.set_xticks(np.arange(self.full_xlim[0], self.full_xlim[1] + 1, 5))
                            self.ax.set_yticks(np.arange(self.full_ylim[0], self.full_ylim[1] + 1, 5))
                        except Exception:
                            pass
                        self.fig.set_size_inches(*self.FULL_FIGSIZE, forward=True)
                        fd2, tmp_full = tempfile.mkstemp(prefix=".scatter_full_", suffix=".png", dir=self.static_dir)
                        os.close(fd2)
                        self.fig.savefig(tmp_full, dpi=180, bbox_inches="tight", facecolor=self.fig.get_facecolor())
                        self._replace_atomic(tmp_full, os.path.join(self.static_dir, self._scatter_full_name))

                    finally:
                        # restore original limits
                        self.ax.set_xlim(*prev_xlim)
                        self.ax.set_ylim(*prev_ylim)
                        self.ax.set_zlim(*prev_zlim)
                        self.fig.set_size_inches(*self.STD_FIGSIZE, forward=True)
                        try:
                            # let Matplotlib recalc default ticks for normal view
                            self.ax.set_xticks(self.ax.get_xticks())
                            self.ax.set_yticks(self.ax.get_yticks())
                        except Exception:
                            pass
                        if tmp_full and os.path.exists(tmp_full):
                            os.remove(tmp_full)
            except Exception as e:
                print(f"[SCATTER SAVE ERROR] {e}")
            finally:
                # ensure no temp files remain if replace() failed
                try:
                    if tmp_path and os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
            self._ws_maybe_emit("scatter3d", self.fig)

    def _color(self, t: str):
        cmap = {
            "CAR": "#60A5FA",
            "TRUCK": "#F59E0B",
            "BUS": "#A78BFA",
            "HUMAN": "#22C55E",
            "PERSON": "#22C55E",
            "BIKE": "#06B6D4",
            "BICYCLE": "#06B6D4",
            "UNKNOWN": "#94A3B8",
        }
        return cmap.get(str(t).upper(), "#94A3B8")

    def _draw_tm_overlays(self):
        # Zones (optional polygons)
        for Z in self.zones:
            P = Z["poly"]
            if P is None or len(P) < 3: 
                continue
            px = list(P[:,0]) + [P[0,0]]
            py = list(P[:,1]) + [P[0,1]]
            pz = [0]*len(px)
            self.ax.plot(px, py, pz, color="#34D399", lw=1.2, ls="--")

    # ───────── 2-D TM panes (headless PNG writer) ─────────
    def _render_loop_2d(self):
        while self.running:
            try:
                self._render_once_2d()
            except Exception:
                pass
            time.sleep(0.25)  # ~10 fps

    def _render_once_2d(self):
        had_content = False
        lane_stats_for_json = []
        with self.lock:
            # -------- Left pane: Gating & Association --------
            a = self.ax_gate
            a.cla()
            self._style_2d_axes(a, title="Gating & Association")
            # Zones
            for Z in self.zones:
                P = Z.get("poly")
                if isinstance(P, np.ndarray) and len(P) >= 3:
                    px = list(P[:, 0]) + [P[0, 0]]
                    py = list(P[:, 1]) + [P[0, 1]]
                    a.plot(px, py, color="#34D399", lw=1.0, ls="--")
                    had_content = True
            # 2D projected cloud (context)
            if isinstance(self.points2d, np.ndarray) and len(self.points2d) > 0:
                try:
                    a.scatter(self.points2d[:, 0], self.points2d[:, 1], s=4, alpha=0.18)
                    had_content = True
                except Exception:
                    pass
            # Heads + gating circles
            GATE_R = 2.0
            for o in self.tracks2d:
                a.scatter(o["x"], o["y"], s=18, alpha=0.95, color=self._color(o["type"]))
                label = (o.get("tid") or "").strip()
                if label:
                    a.text(o["x"], o["y"] + 0.8, f"{label}  {o['speed']:.1f}", fontsize=8, color=THEME["FG"])
                self._draw_gate_circle(a, o["x"], o["y"], GATE_R, color=self._color(o["type"]))
                had_content = True

            # -------- Right pane: Counting / Trails --------
            b = self.ax_count
            b.cla()
            self._style_2d_axes(b, title="Counting / Trails")
            # Trails (faded)
            try:
                for tid, dq in list(self.trails.items()):
                    if len(dq) < 2:
                        continue
                    color = "#94A3B8"
                    for _o in self.tracks2d:
                        if _o.get("tid") == tid:
                            color = self._color(_o.get("type", "UNKNOWN"))
                            break
                    n = len(dq)
                    for i in range(n - 1):
                        _, x1, y1 = dq[i]
                        _, x2, y2 = dq[i + 1]
                        alpha = 0.10 + 0.70 * (i + 1) / (n - 1)
                        b.plot([x1, x2], [y1, y2], linewidth=2.0, alpha=alpha, color=color)
                        had_content = True
            except Exception:
                pass
            # Lanes + live counts + avg speed per lane
            for L in (self.lanes or []):
                x1, y1, x2, y2 = L["rect"]
                xs = [x1, x2, x2, x1, x1]
                ys = [y1, y1, y2, y2, y1]
                b.plot(xs, ys, color=L.get("color", "#60A5FA"), lw=1.5)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                live_cnt = int(L.get("count", 0))
                cnt = int(self.tm_counts.get(str(L.get("id", "")), live_cnt) or live_cnt)
                speeds = [o["speed"] for o in self.tracks2d if self._point_in_rect(o["x"], o["y"], L["rect"])]
                avg_kmh = (sum(speeds) / len(speeds)) if speeds else 0.0
                b.text(cx, cy, f"{cnt}  |  {avg_kmh:.1f} km/h",
                       ha="center", va="center", color=THEME["FG"], fontsize=8,
                       path_effects=[pe.Stroke(linewidth=2, foreground="#000"), pe.Normal()])
                lane_stats_for_json.append({
                    "lane_id": str(L.get("id", "")),
                    "live_count": cnt,
                    "avg_kmh": round(avg_kmh, 1)
                })
                had_content = True
        now = time.time()
        if (now - getattr(self, "_last_tm2d_save", 0.0)) < getattr(self, "_tm2d_save_every", 0.08):
            return
        self._last_tm2d_save = now
        # --- Atomic save for standard and full-view (no-blip) ---
        try:
            if had_content or not self._last_tm2d_had_content:
                os.makedirs(self.static_dir, exist_ok=True)

                fd1, tmp1 = tempfile.mkstemp(prefix=".tm2d_", suffix=".png", dir=self.static_dir)
                os.close(fd1)
                try:
                    self.fig2.savefig(
                        tmp1, dpi=92, bbox_inches="tight",
                        facecolor=self.fig2.get_facecolor()
                    )
                    self._replace_atomic(tmp1, self._tm2d_path)
                finally:
                    if os.path.exists(tmp1):
                        try: os.remove(tmp1)
                        except Exception: pass

                # Write tm2d_full.png with wider limits + higher DPI, atomically
                a_xlim, a_ylim = self.ax_gate.get_xlim(), self.ax_gate.get_ylim()
                b_xlim, b_ylim = self.ax_count.get_xlim(), self.ax_count.get_ylim()
                fd2, tmp2 = tempfile.mkstemp(prefix=".tm2d_full_", suffix=".png", dir=self.static_dir)
                os.close(fd2)
                try:
                    self.ax_gate.set_xlim(*self.tm2d_full_xlim)
                    self.ax_gate.set_ylim(*self.tm2d_full_ylim)
                    self.ax_count.set_xlim(*self.tm2d_full_xlim)
                    self.ax_count.set_ylim(*self.tm2d_full_ylim)
                    xt = np.arange(self.tm2d_full_xlim[0], self.tm2d_full_xlim[1] + 1, 5)
                    yt = np.arange(self.tm2d_full_ylim[0], self.tm2d_full_ylim[1] + 1, 5)
                    for ax in (self.ax_gate, self.ax_count):
                        ax.set_xticks(xt)
                        ax.set_yticks(yt)
                        ax.grid(True, color=THEME["GRID"], alpha=0.5, linewidth=0.6)
                    self.fig2.savefig(
                        tmp2, dpi=180, bbox_inches="tight",
                        facecolor=self.fig2.get_facecolor()
                    )
                    self._replace_atomic(tmp2, self._tm2d_full_path)
                finally:
                    # restore limits and cleanup
                    self.ax_gate.set_xlim(*a_xlim); self.ax_gate.set_ylim(*a_ylim)
                    self.ax_count.set_xlim(*b_xlim); self.ax_count.set_ylim(*b_ylim)
                    for ax in (self.ax_gate, self.ax_count):
                        ax.set_xticks(ax.get_xticks())
                        ax.set_yticks(ax.get_yticks())
                    if os.path.exists(tmp2):
                        try: os.remove(tmp2)
                        except Exception: pass

                # remember whether this frame had content
                self._last_tm2d_had_content = had_content
                try:
                    fdj, tmpj = tempfile.mkstemp(prefix=".tm_stats_", suffix=".json", dir=self.static_dir)
                    os.close(fdj)
                    with open(tmpj, "w") as f:
                        json.dump({"lanes": lane_stats_for_json, "ts": time.time()}, f, separators=(",", ":"))
                    self._replace_atomic(tmpj, self._tm_stats_path)
                except Exception as e:
                    print(f"[TM STATS SAVE ERROR] {e}")             
            # else: no content and we've already written content before -> keep previous files intact
        except Exception as e:
            print(f"[TM2D SAVE ERROR] {e}")
        self._ws_maybe_emit("tm2d", self.fig2)

    def _style_2d_axes(self, ax, title=""):
        ax.set_facecolor(THEME["BG"])
        ax.grid(True, color=THEME["GRID"], alpha=0.4, linewidth=0.5)
        ax.set_xlabel("X (m)", color=THEME["FG"])
        ax.set_ylabel("Y (m)", color=THEME["FG"])
        ax.set_xlim(-20, 20)
        ax.set_ylim(0, 80)
        ax.tick_params(colors=THEME["AX"])
        if title:
            ax.set_title(title, color=THEME["FG"], fontsize=10)

    def _draw_gate_circle(self, ax, cx, cy, r, color="#94A3B8", steps=64):
        """Small dashed circle representing the gating radius around a track head."""
        t = np.linspace(0, 2*np.pi, steps)
        xs = cx + r * np.cos(t)
        ys = cy + r * np.sin(t)
        ax.plot(xs, ys, ls="--", lw=0.9, color=color, alpha=0.55)

    def _build_default_lanes(self):
        d = self._lane_defaults
        n = max(1, int(d["num_lanes"]))
        w = float(d["lane_width_m"])
        cx = float(d["center_x_m"])
        y1 = float(d["start_y_m"])
        y2 = float(d["end_y_m"])
        total_w = n * w
        left_edge = cx - total_w / 2.0
        lanes = []
        for i in range(n):
            x1 = left_edge + i * w
            x2 = x1 + w
            lanes.append({"id": i + 1, "rect": [x1, y1, x2, y2], "count": 0, "color": "#60A5FA"})
        return lanes

    def _point_in_rect(self, x, y, rect):
        x1, y1, x2, y2 = rect
        return (x1 <= x <= x2) and (y1 <= y <= y2)

    def _style_axes(self):
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(0, 80)
        self.ax.set_zlim(0, 6)
        self.ax.set_xlabel("X (m)", color=THEME["FG"])
        self.ax.set_ylabel("Y (m)", color=THEME["FG"])
        self.ax.set_zlabel("Z (m)", color=THEME["FG"])
        self.ax.set_title("IWR6843 • Live 3D Object Cloud", color=THEME["FG"])
        try:
            self.ax.xaxis._axinfo["grid"]["color"] = THEME["GRID"]
            self.ax.yaxis._axinfo["grid"]["color"] = THEME["GRID"]
            self.ax.zaxis._axinfo["grid"]["color"] = THEME["GRID"]
        except Exception:
            pass

    # ───────────────────────── WebSocket helpers ─────────────────────────
    def _ws_maybe_emit(self, kind: str, fig, *, force: bool=False):
        """
        Encode the current figure into a JPEG and pass to frame sink, throttled.
        kind ∈ {"scatter3d", "tm2d"}
        """
        if not self._frame_sink:
            return
        now = time.time()
        if not force and (now - self._last_ws_emit) < self._ws_emit_every:
            return
        self._last_ws_emit = now
        try:
            # draw canvas and grab RGB buffer
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
            ok, buf = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), int(self._ws_quality)])
            if ok:
                self._frame_sink(kind, buf.tobytes())
        except Exception:
            # best-effort: never let a UI sink stall the pipeline
            pass