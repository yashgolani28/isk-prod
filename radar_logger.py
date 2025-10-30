import csv
import os
import time
from datetime import datetime

class IWR6843Logger:
    def __init__(self, log_dir="radar-logs"):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(log_dir, f"iwr_people_log_{ts}.csv")
        self.fields = [
            "timestamp", "object_id", "motion_state",
            "range_m", "azimuth_deg", "elevation_deg",
            "x", "y", "z",
            "vRel", "snr", "direction", "classification"
        ]
        with open(self.filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writeheader()

    def log_targets(self, targets):
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            for t in targets:
                writer.writerow({
                    "timestamp": datetime.fromtimestamp(t.get("timestamp", time.time())).isoformat(),
                    "object_id": t.get("object_id", t.get("id", "")),
                    "motion_state": t.get("motion_state", "unknown"),
                    "range_m": round(t.get("distance", 0.0), 2),
                    "azimuth_deg": t.get("azimuth", 0.0),
                    "elevation_deg": t.get("elevation", 0.0),
                    "x": round(t.get("posX", 0.0), 2),
                    "y": round(t.get("posY", 0.0), 2),
                    "z": round(t.get("posZ", 0.0), 2),
                    "vRel": round((t.get("velX", 0.0)**2 + t.get("velY", 0.0)**2)**0.5, 2),
                    "snr": round(t.get("confidence", 0.0) * 10, 1),
                    "direction": t.get("direction", "STATIC"),
                    "classification": t.get("type", "UNKNOWN")
                })

