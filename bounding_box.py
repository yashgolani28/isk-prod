from ultralytics import YOLO
import cv2
import os
from threading import Thread
import numpy as np
import math
import json
from logger import logger

PROJECTION_MATRIX_PATH = "calibration/camera_projection_matrix.npy"

try:
    model = YOLO("yolov8s.pt")
    print("[INFO] Custom model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    model = None

projection_matrix = None
if os.path.exists(PROJECTION_MATRIX_PATH):
    try:
        projection_matrix = np.load(PROJECTION_MATRIX_PATH)
        print(f"[INFO] Loaded camera projection matrix from {PROJECTION_MATRIX_PATH}")
    except Exception as e:
        print(f"[WARN] Failed to load projection matrix: {e}")
        projection_matrix = None


def project_radar_to_pixel(x, y, z, image_width):
    try:
        coeffs = np.load("calibration/camera_projection_matrix.npy")
        pixel_x = coeffs[0] * x + coeffs[1]
        return float(pixel_x), 0.0
    except:
        return image_width / 2, 0.0


def estimate_distance_from_box(bbox, frame_height):
    x1, y1, x2, y2 = bbox
    box_height = y2 - y1
    if box_height <= 0 or not math.isfinite(box_height):
        return float("nan")
    real_height_m = 1.7
    focal_length_px = 650
    distance = (real_height_m * focal_length_px) / box_height
    if not math.isfinite(distance) or distance > 100:
        return float("nan")
    return distance


def _canon_label(name: str):
    if not name:
        return None
    n = name.strip().lower()
    mapping = {
        "car": "CAR",
        "truck": "TRUCK",
        "bus": "BUS",
        "bicycle": "BICYCLE",
        "motorbike": "BIKE",
        "motorcycle": "BIKE",
        "person": "HUMAN",
    }
    return mapping.get(n)


def annotate_speeding_object(
    image_path,
    radar_distance,
    label=None,
    save_dir="snapshots",
    min_confidence=0.35,
    obj_x=0.0,
    obj_y=1.0,
    obj_z=0.0,
    class_hint=None,
):
    """
    Returns: (save_path, visual_distance, corrected_distance, bbox, detected_label)
    """
    print(f"[DEBUG] Annotating snapshot: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Failed to read image: {image_path}")
        return None, None, radar_distance, None, None
    if model is None:
        print("[ERROR] YOLO model not available.")
        return None, None, radar_distance, None, None

    results = model.predict(source=img, imgsz=640, verbose=False)[0]
    h, w = img.shape[:2]

    pixel_from_azimuth, _ = project_radar_to_pixel(obj_x, obj_y, obj_z, w)
    print(f"[DEBUG] Radar azimuth projected to pixel: {pixel_from_azimuth}")

    vehicle_classes = {"car", "truck", "bus", "bicycle", "motorbike", "motorcycle"}
    human_classes = {"person"}
    allowed_classes = vehicle_classes | human_classes

    prefer_vehicle = False
    if class_hint:
        ch = str(class_hint).upper()
        prefer_vehicle = any(k in ch for k in ("VEHICLE", "CAR", "TRUCK", "BUS", "BIKE", "BICYCLE"))

    best_box = None
    best_score = float("inf")
    best_class = None

    table_band_min = int(0.50 * h) if (obj_z is not None and float(obj_z) < 0.30) else 0

    for box in results.boxes:
        cls_id = int(box.cls)
        class_name = model.names[cls_id].lower()
        conf = float(box.conf[0])

        if class_name not in allowed_classes or conf < min_confidence:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bw, bh = (x2 - x1), (y2 - y1)
        if bw < 16 or bh < 16:
            continue

        if prefer_vehicle:
            aspect = bw / max(1, bh)
            if table_band_min and y2 < table_band_min:
                continue
            if aspect < 0.85:
                continue

        center_x = (x1 + x2) / 2.0
        angle_score = abs(center_x - pixel_from_azimuth)

        if class_name in human_classes:
            est_dist = estimate_distance_from_box((x1, y1, x2, y2), h)
            if math.isnan(est_dist):
                continue

        gate = 160 if prefer_vehicle else 120
        if angle_score < best_score and angle_score <= gate:
            best_score = angle_score
            best_box = (x1, y1, x2, y2)
            best_class = class_name

    if not best_box and prefer_vehicle:
        veh_candidates = [
            b
            for b in results.boxes
            if model.names[int(b.cls)].lower() in vehicle_classes
            and float(b.conf[0]) >= min_confidence
        ]
        if veh_candidates:
            print("[FALLBACK] Using top-confidence VEHICLE box")
            top = max(veh_candidates, key=lambda b: float(b.conf[0]))
            best_box = tuple(map(int, top.xyxy[0]))
            best_class = model.names[int(top.cls)].lower()

    if not best_box and not prefer_vehicle:
        filtered = [
            b
            for b in results.boxes
            if model.names[int(b.cls)].lower() in allowed_classes
            and float(b.conf[0]) >= min_confidence
        ]
        if filtered:
            print("[FALLBACK] Using top-confidence allowed box")
            top = max(filtered, key=lambda b: float(b.conf[0]))
            best_box = tuple(map(int, top.xyxy[0]))
            best_class = model.names[int(top.cls)].lower()

    if not best_box:
        print("[DEBUG] No valid bounding box matched.")
        return None, None, radar_distance, None, None

    x1, y1, x2, y2 = best_box
    print(f"[MATCH] picked={(x1, y1, x2, y2)} min_conf={min_confidence} prefer_vehicle={prefer_vehicle}")
    cropped = img[y1:y2, x1:x2]
    if cropped.size == 0:
        print("[ERROR] Cropped image is empty.")
        return None, None, radar_distance, None, None

    ch = cropped.shape[0]
    mtop = int(0.08 * ch)
    mbot = int(0.08 * ch)
    cropped = cropped[mtop : ch - mbot, :]

    resized = cv2.resize(cropped, (384, 448), interpolation=cv2.INTER_AREA)
    save_name = f"cropped_{os.path.basename(image_path)}"
    save_path = os.path.join(save_dir, save_name)
    cv2.imwrite(save_path, resized)
    print(f"[DEBUG] Saved clean cropped image: {save_path}")

    vis = estimate_distance_from_box((x1, y1, x2, y2), h)
    visual_distance = vis if (vis is not None and math.isfinite(vis)) else None
    corrected = float(radar_distance)  # keep radar as the authoritative distance
    detected_label = _canon_label(best_class)
    return save_path, visual_distance, corrected, (x1, y1, x2, y2), detected_label

def annotate_async(
    image_path,
    radar_distance,
    label=None,
    save_dir="snapshots",
    min_confidence=0.35,
    obj_x=0.0,
    obj_y=1.0,
    obj_z=0.0,
    class_hint=None,
):
    def _run():
        try:
            annotate_speeding_object(
                image_path=image_path,
                radar_distance=radar_distance,
                label=label,
                save_dir=save_dir,
                min_confidence=min_confidence,
                obj_x=obj_x,
                obj_y=obj_y,
                obj_z=obj_z,
                class_hint=class_hint,
            )
        except Exception as e:
            print(f"[ANNOTATION ASYNC FAIL] {e}")

    def low_priority():
        try:
            os.nice(10)
        except:
            pass
        _run()

    Thread(target=low_priority, daemon=True).start()
