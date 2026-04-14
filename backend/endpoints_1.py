import os

import main_timestamp
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import time
import json
import datetime
import asyncio
import threading
import numpy as np
from threading import Thread
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import config
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

from main_timestamp import (
    initialization,
    feeder,
    ACLAE_inference,
    stop_event
)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED STATE
# ══════════════════════════════════════════════════════════════════════════════

# latest_frame holds a JPEG-encoded frame with YOLO boxes drawn on it.
# Written by yolo_worker_annotating(), read by /video_feed.
latest_frame: bytes | None = None
frame_lock   = threading.Lock()

ws_clients: set = set()
ws_lock         = asyncio.Lock()

alert_settings: dict = {
    "Smoking":         True,
    "Group of People": True,
    "Weapons":         True,
    "Abuse":           True,
    "Arrest":          True,
    "Arson":           True,
    "Assault":         True,
    "Burglary":        True,
    "Explosion":       True,
    "Fighting":        True,
    "Normal Videos":   True,
    "RoadAccidents":   True,
    "Robbery":         True,
    "Shooting":        True,
    "Shoplifting":     True,
    "Stealing":        True,
    "Vandalism":       True,
}


# ══════════════════════════════════════════════════════════════════════════════
# YOLO WORKER  (replaces the bare yolo_worker from main_3)
# Draws boxes on each frame, writes JPEG to latest_frame, fires YOLO alerts.
# ══════════════════════════════════════════════════════════════════════════════

_LABEL_COLORS = {
    "person":    (255, 80,  80),
    "cigarette": (0,   0,  255),
}
_DEFAULT_COLOR = (0, 255, 0)


def yolo_worker_annotating():
    global latest_frame

    from queue import Empty

    if main_timestamp.Yolo is None:
        print("[YOLO] ERROR: Model not initialized. Thread exiting.")
        return

    print("[YOLO] Annotating worker started.")

    COOLDOWN      = getattr(config, "COOLDOWN", 5)
    _last_smoking = 0.0
    _last_crowd   = 0.0

    while True:
        try:
            frame = main_timestamp.yolo_queue.get(timeout=2)
        except Empty:
            if stop_event.is_set():
                break
            continue

        if frame is None:
            break

        try:
            results = main_timestamp.Yolo(frame, verbose=False)[0]
        except Exception as e:
            print(f"[YOLO] Inference error: {e}")
            main_timestamp.yolo_queue.task_done()
            continue

        person_centers    = []
        cigarette_centers = []
        annotated         = frame.copy()

        for box in results.boxes:
            cls             = int(box.cls[0])
            label           = main_timestamp.Yolo.names[cls].lower()
            conf            = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy          = (x1 + x2) // 2, (y1 + y2) // 2
            color           = _LABEL_COLORS.get(label, _DEFAULT_COLOR)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                f"{label} {conf:.2f}",
                (x1, max(y1 - 6, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
            )

            if label == "person":
                person_centers.append((cx, cy))
            elif label == "cigarette":
                cigarette_centers.append((cx, cy))

        now = time.time()

        # ── Smoking alert ──────────────────────────────────────────────────────
        smoking = any(
            np.linalg.norm(np.array(p) - np.array(c)) < config.SMOKING_DISTANCE
            for p in person_centers for c in cigarette_centers
        )
        if smoking and alert_settings.get("Smoking", True) and now - _last_smoking > COOLDOWN:
            _last_smoking = now
            print("[YOLO] ⚠ Smoking detected!")
            _fire_alert(
                label="Smoking",
                message="Smoking detected!",
                conf=None,
                start_str=datetime.datetime.now().strftime("%H:%M:%S"),
                end_str=None,
            )

        # ── Crowd alert ────────────────────────────────────────────────────────
        if (
            len(person_centers) >= config.GROUP_COUNT
            and alert_settings.get("Group of People", True)
            and now - _last_crowd > COOLDOWN
        ):
            arr         = np.array(person_centers)
            dist_matrix = np.linalg.norm(arr[:, None] - arr[None, :], axis=2)
            if np.any((dist_matrix < config.GROUP_DISTANCE).sum(axis=1) >= config.GROUP_COUNT):
                _last_crowd = now
                print("[YOLO] ⚠ Group detected!")
                _fire_alert(
                    label="Group of People",
                    message="Group of people detected!",
                    conf=None,
                    start_str=datetime.datetime.now().strftime("%H:%M:%S"),
                    end_str=None,
                )

        # ── Store annotated frame ──────────────────────────────────────────────
        ok, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with frame_lock:
                latest_frame = jpeg.tobytes()

        main_timestamp.yolo_queue.task_done()

    print("[YOLO] Annotating worker done.")


# ══════════════════════════════════════════════════════════════════════════════
# BROADCAST HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _fire_alert(label: str, message: str, conf, start_str: str, end_str: str | None):
    """Called from sync threads (YOLO worker). Runs the async broadcast in a new loop."""
    payload = {
        "event":      "alert",
        "type":       label,
        "message":    message,
        "confidence": round(conf, 4) if conf is not None else None,
        "clip_start": start_str,
        "clip_end":   end_str or start_str,
        "timestamp":  datetime.datetime.now().isoformat(),
    }
    asyncio.run(_async_broadcast(payload))


async def _async_broadcast(message: dict):
    msg = json.dumps(message)
    async with ws_lock:
        dead = set()
        for ws in ws_clients:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        ws_clients.difference_update(dead)
    _append_log(message)


async def broadcast_message(message: dict):
    """
    Called from main_3._final_output for ViViT detections.
    Must be awaited — runs inside the FastAPI event loop via asyncio.run()
    in _final_output.
    """
    await _async_broadcast(message)


def _append_log(message: dict):
    log_path = getattr(config, "LOG_FILE", None)
    if not log_path:
        return
    try:
        clip_start = message.get("clip_start", "")
        clip_end   = message.get("clip_end",   "")
        window_str = f" [{clip_start} → {clip_end}]" if clip_start else ""
        conf       = message.get("confidence")
        conf_str   = f" (conf: {conf})" if conf is not None else ""
        entry = (
            f"{message.get('timestamp', datetime.datetime.now().isoformat())} "
            f"- {message.get('type', 'Unknown')}{window_str}{conf_str}\n"
        )
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(entry)
    except OSError as e:
        print(f"[Log] WARNING: Could not write to log file — {e}")


# ══════════════════════════════════════════════════════════════════════════════
# LIFESPAN
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        initialization(
            config.ACLAE_path,
            config.MILNET_Path,
            config.ResNet_path,
            config.Yolo_path,
            config.ViViT_wt_path,
            config.Vivit_architecture_path,
        )
    except Exception as e:
        print(f"[Startup] FATAL: Model initialization failed — {e}")
        raise

    Thread(target=feeder,                 daemon=True, name="Feeder").start()
    Thread(target=yolo_worker_annotating, daemon=True, name="YOLO").start()
    Thread(target=ACLAE_inference,        daemon=True, name="ACLAE").start()
    print("[Startup] All worker threads started.")

    yield

    stop_event.set()
    print("[Shutdown] stop_event set — workers will exit cleanly.")


# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="CCTV Alert System", lifespan=lifespan)

_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")
else:
    print("[Startup] WARNING: 'static' directory not found — /static not mounted.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/video_feed")
def video_feed():
    """
    MJPEG stream of YOLO-annotated frames.
    Frontend points an <img> src at this URL.
    Delay is natural — YOLO inference takes time.
    """
    def generate():
        while not stop_event.is_set():
            with frame_lock:
                frame = latest_frame
            if frame is None:
                time.sleep(0.01)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.033)   # ~30 fps ceiling

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/logs")
def get_logs():
    """
    Returns the last 50 log entries.
    Frontend polls this ~800 ms after each WebSocket alert.
    Each line format: <timestamp> - <type> [<start> → <end>] (conf: <n>)
    """
    log_path = getattr(config, "LOG_FILE", None)
    if not log_path or not os.path.exists(log_path):
        return {"logs": []}
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-50:]
        return {"logs": [l.strip() for l in lines if l.strip()]}
    except OSError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.websocket("/ws/alerts")
async def websocket_endpoint(ws: WebSocket):
    """
    Persistent WebSocket. Frontend receives alert objects:
    {
      event:      "alert",
      type:       "Burglary",          // matches frontend labels list
      message:    "Burglary detected!",
      confidence: 0.7395,              // null for YOLO-only alerts
      clip_start: "23:44:07",
      clip_end:   "23:44:22",
      timestamp:  "2026-04-13T23:44:22.123456"
    }
    """
    await ws.accept()
    async with ws_lock:
        ws_clients.add(ws)
    try:
        while True:
            await ws.receive_text()     # keeps the connection alive
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS] Unexpected error: {e}")
    finally:
        async with ws_lock:
            ws_clients.discard(ws)


@app.post("/update-settings")
async def update_settings(request: Request):
    """
    Receives { label: bool, ... } from the frontend Settings dialog.
    Updates alert_settings immediately — no restart needed.
    Both YOLO and ViViT alerts check this before broadcasting.
    """
    global alert_settings
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON payload."})
    if not isinstance(data, dict):
        return JSONResponse(status_code=400, content={"error": "Expected a JSON object."})

    alert_settings = data
    print("[Settings] Updated:", alert_settings)
    return {"status": "ok"}