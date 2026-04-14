import os
import sys
import cv2
import time
import json
import datetime
import numpy as np
import asyncio
import threading
import winsound
from ultralytics import YOLO
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import config

from fastapi.middleware.cors import CORSMiddleware

models_path = r'D:\Essentials\Projects\Major Project\Intelligent Monitoring System\Final_file\Integration'
sys.path.append(models_path)

import Integration_threaded_v2




# ---thread variables---
Yolo_thread_running= False
Sec2_thread_running = False


# --- Setup ---
os.makedirs(config.SNAPSHOT_DIR, exist_ok=True)
# model = YOLO(config.MODEL_PATH)
app = FastAPI(title="CCTV Alert System")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],  # frontend ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

alert_settings = {
    "Smoking": True,
    "Group of People": True,
"Weapons": True,
"Abuse": True,
"Arrest": True,
"Arson": True,
"Assault": True,
"Burglary": True,
"Explosion": True,
"Fighting": True,
"Normal Videos": True,
"RoadAccidents": True,
"Robbery": True,
"Shooting": True,
"Shoplifting": True,
"Stealing": True,
"Vandalism": True
}

# Shared state
latest_frame = None
frame_lock = threading.Lock()
ws_clients = set()
ws_lock = asyncio.Lock()
alert_queue = asyncio.Queue()

# Helper: Log event
def log_event(event_type):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(config.LOG_FILE, "a") as f:
        f.write(f"{ts} - {event_type}\n")
    print(ts, "-", event_type)

# Run async coroutine in thread
def run_in_thread(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(coro)
    loop.close()

# Helper: Center of a bounding box
def get_center(box):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    return ((x1 + x2)//2, (y1 + y2)//2)

# --- Broadcast helper ---
async def broadcast_message(message: dict):
    msg = json.dumps(message)
    async with ws_lock:
        for ws in list(ws_clients):
            try:
                await ws.send_text(msg)
            except:
                ws_clients.remove(ws)


def YOLO_inference(frame):
        current_time = time.time()
        results = YOLO_model(frame)[0]
        person_centers, cigarette_centers = [], []

        for box in results.boxes:
            cls = int(box.cls[0])
            label = YOLO_model.names[cls].lower()
            cx, cy = get_center(box)

            if label == "person":
                person_centers.append((cx, cy))
                color = (255, 0, 0)
            elif label == "cigarette":
                cigarette_centers.append((cx, cy))
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # --- Smoking Detection ---
        smoking_detected = any(
            np.linalg.norm(np.array(p) - np.array(c)) < config.SMOKING_DISTANCE
            for p in person_centers for c in cigarette_centers
        )

        if smoking_detected and (current_time - smoking_last > config.COOLDOWN):
            if alert_settings.get("Smoking", True):
                log_event("Smoking detected")
                winsound.Beep(config.ALERT_SOUND_FREQ, config.ALERT_SOUND_DUR)
                snap_path = os.path.join(config.SNAPSHOT_DIR, f"smoking_{int(time.time())}.jpg")
                cv2.imwrite(snap_path, frame)

                run_in_thread(broadcast_message({
                    "event": "alert",
                    "type": "smoking",
                    "message": "🚬 Smoking detected!",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "snapshot": f"/static/snapshots/{os.path.basename(snap_path)}"
                }))
            smoking_last = current_time

        # --- Group Detection ---
        crowd_detected = False
        if len(person_centers) >= config.GROUP_COUNT:
            dist_matrix = np.linalg.norm(
                np.array(person_centers)[:, None] - np.array(person_centers)[None, :], axis=2)
            close_counts = (dist_matrix < config.GROUP_DISTANCE).sum(axis=1)
            if np.any(close_counts >= config.GROUP_COUNT):
                crowd_detected = True

        if crowd_detected and (current_time - group_last > config.COOLDOWN):
            if alert_settings.get("Group of People", True):
                log_event("Group detected")
                winsound.Beep(config.ALERT_SOUND_FREQ, config.ALERT_SOUND_DUR)
                snap_path = os.path.join(config.SNAPSHOT_DIR, f"group_{int(time.time())}.jpg")
                cv2.imwrite(snap_path, frame)

                run_in_thread(broadcast_message({
                    "event": "alert",
                    "type": "group",
                    "message": "👥 Group detected!",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "snapshot": f"/static/snapshots/{os.path.basename(snap_path)}"
                }))
            group_last = current_time



def detector_loop():
    global latest_frame
    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    global smoking_last, group_last
    smoking_last = 0 , group_last = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        if not Yolo_thread_running:
            threading.Thread(target= YOLO_inference(frame))
        
        if not Sec2_thread_running:
            threading.Thread()

        # Encode frame for frontend
        _, jpeg = cv2.imencode(".jpg", frame)
        with frame_lock:
            latest_frame = jpeg.tobytes()





def initialize():
    global YOLO_model
    YOLO_model = YOLO(config.MODEL_PATH)
    Integration_threaded_v2.initialize_ConvLSTM()
    Integration_threaded_v2.initialize_thread2_models()
    run_video_stream()






# --- Startup ---
@app.on_event("startup")
async def startup_event():
    threading.Thread(target=detector_loop, daemon=True).start()

# --- Video Stream ---
@app.get("/video_feed")
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if latest_frame is None:
                    continue
                frame = latest_frame
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.04)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

# --- Logs Endpoint ---
@app.get("/api/logs")
def get_logs():
    if not os.path.exists(config.LOG_FILE):
        return {"logs": []}
    with open(config.LOG_FILE, "r") as f:
        lines = f.readlines()[-50:]
    return {"logs": [l.strip() for l in lines]}

# --- WebSocket Alerts ---
@app.websocket("/ws/alerts")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    async with ws_lock:
        ws_clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        async with ws_lock:
            ws_clients.remove(ws)

from fastapi import Request

@app.post("/update-settings")
async def update_settings(request: Request):
    global alert_settings
    alert_settings = await request.json()
    print("Updated settings:", alert_settings)
    return {"status": "ok"}
