import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import cv2
import time
import json
import datetime
import numpy as np
import asyncio
import threading
import winsound
from threading import Thread
from ultralytics import YOLO


models_path = r'D:\Essentials\Projects\Major Project\Intelligent Monitoring System\Final_file\Integration'
sys.path.append(models_path)
from ConvLSTM_Autoencoder.HybridConvLSTM import HybridConvLSTM
from Clip_MIL.MILNet import MILNet


from transformers import VivitForVideoClassification
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import config
import torch
from torch.nn import MSELoss
from collections import deque
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
import torchvision.models as models
import config
from fastapi.middleware.cors import CORSMiddleware
from main_final import run_video_stream
from contextlib import asynccontextmanager


# --- Shared state ---
latest_frame = None
frame_lock = threading.Lock()
ws_clients = set()
ws_lock = asyncio.Lock()
alert_queue = asyncio.Queue()


# --- Lifespan (startup) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=run_video_stream, daemon=True).start()
    yield


# --- App creation (with lifespan and routes) ---
app = FastAPI(
    title="CCTV Alert System",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Video stream endpoint ---
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


# --- Logs endpoint ---
@app.get("/api/logs")
def get_logs():
    if not os.path.exists(config.LOG_FILE):
        return {"logs": []}
    with open(config.LOG_FILE, "r") as f:
        lines = f.readlines()[-50:]
    return {"logs": [l.strip() for l in lines]}


# --- WebSocket alerts ---
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


# --- Settings update ---
from fastapi import Request

@app.post("/update-settings")
async def update_settings(request: Request):
    global alert_settings
    alert_settings = await request.json()
    print("Updated settings:", alert_settings)
    return {"status": "ok"}