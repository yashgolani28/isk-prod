from gui_parser import uartParser
from config_utils import load_config
import numpy as np
import time, os, sys, hashlib, re
from pathlib import Path
import serial

class IWR6843Interface:
    """
    Normalizes TLVs to: range_azimuth_heatmap, range_doppler_heatmap, azimuth_elevation_heatmap
    Keeps last good heatmap if a frame is partial.
    """
    def __init__(self):
        self.cfg = load_config()
        ports = self.cfg.get("iwr_ports", {}) or {}
        self.cli_port  = ports.get("cli",  "/dev/ttyUSB0")
        self.data_port = ports.get("data", "/dev/ttyUSB1")
        self.cfg_path  = ports.get("cfg_path", "isk_config.cfg")
        parser_type    = ports.get("parser_type", "3D People Counting")

        self._last_ra = None
        self._last_rd = None
        self._last_ae = None

        self.parser = uartParser(type=parser_type)
        self.parser.connectComPorts(self.cli_port, self.data_port)
        self.v_unamb_ms = None
        self._ensure_config_applied_once()

    def get_targets(self):
        try:
            frame = self.parser.readAndParseUart()
        except Exception as e:
            print(f"[ERROR] readAndParseUart failed: {e}", file=sys.stderr)
            return {}
        if not isinstance(frame, dict):
            return {}

        ra = self._extract_heatmap(frame, ["range_azimuth_heatmap","RANGE_AZIMUTH_HEATMAP","rangeAzimuthHeatMap","azimuthHeatMap"])
        rd = self._extract_heatmap(frame, ["range_doppler_heatmap","RANGE_DOPPLER_HEAT_MAP","rangeDopplerHeatMap","dopplerHeatMap"])
        ae = self._extract_heatmap(frame, ["azimuth_elevation_heatmap","AZIMUTH_ELEVATION_HEATMAP","azimuthElevationHeatMap"])

        if ra is None and self._last_ra is not None: frame["range_azimuth_heatmap"] = self._last_ra.copy()
        elif ra is not None: frame["range_azimuth_heatmap"] = ra; self._last_ra = ra

        if rd is None and self._last_rd is not None: frame["range_doppler_heatmap"] = self._last_rd.copy()
        elif rd is not None: frame["range_doppler_heatmap"] = rd; self._last_rd = rd

        if ae is None and self._last_ae is not None: frame["azimuth_elevation_heatmap"] = self._last_ae.copy()
        elif ae is not None: frame["azimuth_elevation_heatmap"] = ae; self._last_ae = ae

        return frame

    def _cfg_hash(self) -> str:
        """
        Stable hash over cfg contents + selected ports. If either changes,
        we will (re)apply the cfg.
        """
        h = hashlib.sha256()
        h.update(self.cli_port.encode("utf-8", "ignore"))
        h.update(self.data_port.encode("utf-8", "ignore"))
        try:
            with open(self.cfg_path, "rb") as f:
                h.update(f.read())
        except Exception:
            pass
        return h.hexdigest()[:16]

    def _reopen_cli(self):
        """
        gui_parser.sendCfg() closes the CLI port upon completion, so make
        sure it's open before any subsequent cfg sends.
        """
        try:
            # reconnect (safe even if already open)
            self.parser.connectComPorts(self.cli_port, self.data_port)
        except Exception as e:
            print(f"[CFG] Reopen CLI failed: {e}", file=sys.stderr)

    # ───────────── Robust CLI helpers ─────────────
    def _clean_cfg_line(self, s: str) -> str:
        # Drop inline comments and normalize whitespace
        s = s.split('%', 1)[0]
        return ' '.join(s.strip().split())

    def _read_until_prompt(self, ser: serial.Serial, timeout: float = 2.0) -> str:
        """Read until we see Done/Error or the mmwDemo prompt."""
        end = time.monotonic() + timeout
        buf = []
        saw_done = False
        while time.monotonic() < end:
            line = ser.readline().decode('latin1', errors='ignore')
            if not line:
                time.sleep(0.005)
                continue
            buf.append(line)
            if 'Error' in line:
                break
            if 'Done' in line:
                saw_done = True
            # different FW builds print slightly different prompts
            if ('mmwDemo:/' in line) or line.strip().endswith('mmwDemo:/>'):
                break
        # If we saw 'Done' but no prompt, still accept the response
        if not buf and saw_done:
            return "Done"
        return ''.join(buf)

    def _normalize_comp_line_for_3dpc(self, cmd: str) -> str:
        """
        3DPC FW expects compRangeBiasAndRxChanPhase: bias + 12 complex pairs (3Tx x 4Rx).
        Pad with identity pairs (1 0) or truncate extra values as needed.
        """
        toks = cmd.split()
        if not toks or toks[0].lower() != 'comprangebiasandrxchanphase':
            return cmd

        vals = toks[1:]
        bias = vals[0] if vals else '0'
        # keep only numeric tokens after bias
        nums = [n for n in vals[1:] if re.match(r'^-?\d+(?:\.\d+)?$', n)]
        need = 24  # 12 pairs
        nums = (nums + (['1', '0'] * 12))[:need]  # pad then truncate
        # normalize bias printing (avoid '0.0'→'0')
        try:
            bias = str(float(bias)).rstrip('0').rstrip('.')
        except Exception:
            bias = '0'
        return 'compRangeBiasAndRxChanPhase ' + bias + ' ' + ' '.join(nums)

    def _send_cfg_robust(self, lines):
        """Send cfg with strict pacing + prompt wait (bypasses gui_parser.sendCfg)."""
        self._reopen_cli()
        ser = self.parser.cliCom
        # normalize user lines
        lines = [self._clean_cfg_line(x) for x in lines]
        lines = [x for x in lines if x]
        # idempotent prelude
        if not lines or lines[0].lower() != 'sensorstop':
            lines.insert(0, 'sensorStop')
        if len(lines) < 2 or lines[1].lower() != 'flushcfg':
            lines.insert(1, 'flushCfg')
        # ensure exactly-one DFE output mode (Frame). Put it right after flushCfg.
        lines = [x for x in lines if not x.lower().startswith('dfedataoutputmode')]
        lines.insert(2, 'dfeDataOutputMode 1')
        # give flushCfg a breath
        time.sleep(0.05)

        for cmd in lines:
            cmd = self._normalize_comp_line_for_3dpc(cmd)
            ser.write((cmd + '\r\n').encode('ascii'))
            ser.flush()
            # tiny pacing helps 3DPC CLI keep up
            time.sleep(0.020)
            resp = self._read_until_prompt(ser, timeout=2.5)
            if 'Error' in resp:
                # Special-case: FW slipped into Chirp mode. Force Frame and retry once.
                if 'DFE Output Mode' in resp or 'DFE' in resp:
                    ser.write(('dfeDataOutputMode 1\r\n').encode('ascii'))
                    ser.flush()
                    time.sleep(0.040)
                    _ = self._read_until_prompt(ser, timeout=2.0)
                    ser.write((cmd + '\r\n').encode('ascii'))
                    ser.flush()
                    time.sleep(0.020)
                    resp2 = self._read_until_prompt(ser, timeout=2.5)
                    if 'Error' not in resp2:
                        continue
                    resp = resp2
                raise RuntimeError(f"Radar CLI error on '{cmd}': {resp.strip()}")

        # brief warm-up before we start reading TLVs
        time.sleep(0.15)
        try:
            ser.reset_input_buffer()
            ser.close()
        except Exception:
            pass
        print("[INFO] Config sent successfully.")

    def _has_live_stream(self, wait_s: float = 1.5) -> bool:
        """
        Quick, bounded probe for the 8-byte magic word on data UART.
        Does NOT block indefinitely. Safe to call at startup/restart.
        """
        magic = b"\x02\x01\x04\x03\x06\x05\x08\x07"
        dc = getattr(self.parser, "dataCom", None)
        if dc is None:
            return False
        try:
            # Clear stale bytes and give the UART a moment
            dc.reset_input_buffer()
        except Exception:
            pass
        start = time.time()
        buf = b""
        while (time.time() - start) < max(0.3, float(wait_s)):
            try:
                chunk = dc.read(1024)  # timeout set by gui_parser (≈0.3 s)
            except Exception:
                chunk = b""
            if chunk:
                buf += chunk
                if magic in buf:
                    return True
            else:
                # tiny nap to avoid tight spin on empty reads
                time.sleep(0.02)
        return False

    def _send_cfg_idempotent(self):
        """
        Prepend a safe prelude (sensorStop/flushCfg) so re-applying the cfg
        never wedges the device, then send the user cfg lines.
        """
        # Ensure CLI UART is open (sendCfg closes it afterwards)
        self._reopen_cli()
        # Read cfg (keep % comment lines out)
        with open(self.cfg_path, "r") as f:
            base = [ln for ln in f if ln.strip() and not ln.strip().startswith("%")]

        # Pre-compute unambiguous speed from the *intended* cfg lines
        try:
            self.v_unamb_ms = self._compute_unambiguous_speed_from_cfg(base)
            print(f"[INFO] v_unamb (predicted) = {self.v_unamb_ms:.2f} m/s")
        except Exception as e:
            print(f"[WARN] Could not compute v_unamb from cfg: {e}")
        print("[INFO] Applying radar cfg (idempotent)…")
        # Use robust sender (waits for Done/prompt; avoids line fragmentation)
        self._send_cfg_robust(base)
        print("[INFO] Radar cfg applied.")

    def get_unambiguous_speed(self):
        """Return unambiguous radial speed (m/s) derived from current cfg, or None."""
        return self.v_unamb_ms

    def _compute_unambiguous_speed_from_cfg(self, lines) -> float:
        """
        Parse channelCfg/profileCfg and compute Doppler unambiguous speed:
            v_unamb = λ / (4 * PRI_sameTX),  PRI_sameTX = (idle + rampEnd)*Ntx
        """
        def clean(s): return ' '.join(s.split('%',1)[0].strip().split())
        start_freq = 60.75  # GHz default if not present
        idle_us = ramp_us = None
        tx_mask = None
        for raw in lines:
            ln = clean(raw)
            if not ln: continue
            toks = ln.split()
            key = toks[0].lower()
            if key == 'channelcfg' and len(toks) >= 3:
                # channelCfg <rxMask> <txMask> <casc>
                try:
                    tx_mask = int(toks[2], 0)
                except: pass
            elif key == 'profilecfg' and len(toks) >= 6:
                # profileCfg id startFreq(GHz) idle(us) adcStart(us) rampEnd(us) ...
                try:
                    start_freq = float(toks[2])
                    idle_us = float(toks[3])
                    ramp_us = float(toks[5])
                except: pass
        if tx_mask is None or idle_us is None or ramp_us is None:
            raise ValueError("Missing channelCfg/profileCfg fields")
        # count TX from mask
        ntx = bin(tx_mask).count('1')
        if ntx <= 0:
            raise ValueError("No TX enabled")
        c = 299_792_458.0
        lam = c / (start_freq * 1e9)  # meters
        pri_same_tx = (idle_us + ramp_us) * 1e-6 * ntx
        if pri_same_tx <= 0:
            raise ValueError("Invalid PRI")
        return lam / (4.0 * pri_same_tx)

    def _ensure_config_applied_once(self):
        """
        Replacement for the old config_sent.flag logic. We keep a flag under
        /run that’s tied to a content hash AND we still probe the stream —
        if the stream is live we skip, otherwise we (re)apply the cfg.
        """
        flag_dir = Path("/run/iwr6843isk")
        flag_dir.mkdir(parents=True, exist_ok=True)
        fpath = flag_dir / f"config_{self._cfg_hash()}.flag"

        # If stream is already live, we're good regardless of flags.
        if self._has_live_stream(wait_s=1.2):
            print("[INFO] Radar stream detected → skipping cfg.")
            # Compute v_unamb from the existing cfg so main/tracker can unwrap Doppler.
            try:
                with open(self.cfg_path, "r") as f:
                    base = [ln for ln in f if ln.strip() and not ln.strip().startswith("%")]
                self.v_unamb_ms = self._compute_unambiguous_speed_from_cfg(base)
                print(f"[INFO] v_unamb (predicted) = {self.v_unamb_ms:.2f} m/s")
            except Exception as e:
                print(f"[WARN] Could not compute v_unamb from cfg: {e}")
            # Make sure the current hash flag exists (helps future restarts)
            try: fpath.touch(exist_ok=True)
            except Exception: pass
            return

        # No live stream → (re)apply config
        print("[INFO] No live stream detected → (re)applying cfg.")
        # Clean up any stale flags from old cfg hashes
        try:
            for p in flag_dir.glob("config_*.flag"):
                try: p.unlink()
                except Exception: pass
        except Exception:
            pass
        # Apply cfg with safe prelude
        self._send_cfg_idempotent()
        # Wait briefly for frames to appear (best-effort)
        if not self._has_live_stream(wait_s=2.0):
            print("[WARN] Stream probe after cfg failed (continuing anyway).")
        # Record the new hash flag
        try: fpath.touch(exist_ok=True)
        except Exception: pass

    def _extract_heatmap(self, frame, keys):
        for k in keys:
            if k in frame and frame[k] is not None:
                arr = self._to_flat_array(frame[k])
                if arr is None: continue
                n = arr.size
                if n in (4096, 2048, 1024) or (n % 32 == 0 and n >= 512):
                    return arr
        return None

    def _to_flat_array(self, raw):
        try:
            if raw is None: return None
            if isinstance(raw, np.ndarray): return raw.astype(np.float32, copy=False).ravel()
            if isinstance(raw, (list, tuple)): return np.asarray(raw, dtype=np.float32).ravel()
            if isinstance(raw, (bytes, bytearray, memoryview)):
                return np.frombuffer(raw, dtype=np.int16).astype(np.float32, copy=False).ravel()
        except Exception as e:
            print(f"[DEBUG] _to_flat_array failed: {e}", file=sys.stderr)
        return None

def check_radar_connection(port="/dev/ttyUSB*", baudrate=115200, timeout=2):
    try:
        import serial
        ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        if ser.is_open: return ser
        ser.close()
    except Exception:
        pass
    return None
