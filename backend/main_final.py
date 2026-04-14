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



mse = MSELoss()






DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---thread variables---
Yolo_thread_running= False
Sec2_thread_running = False
stage2_running = False

# ----Variables-----
high_level_buffer= deque(maxlen=60)
main_buffer= []
temp_frame_buffer=[] 
rgb_buffer = [] 
flow_buffer = [] 
results = [] 
score_history = [] 
s = False 
use_dynamic = False
prev_frame = None

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


def initialize_ConvLSTM():
    global ConvLSTM_model
    ConvLSTM_model = HybridConvLSTM().to(DEVICE)
    ConvLSTM_model.load_state_dict(torch.load(r"D:\Essentials\Projects\Major Project\Intelligent Monitoring System\Final_file\Integration\ConvLSTM_Autoencoder\ConvLSTM_Autoencoder.pth", map_location=DEVICE))



def initialize_MiL():
    global MIL_model
    MIL_model = MILNet(512).to(DEVICE)
    MIL_model.load_state_dict(torch.load(r"D:\Essentials\Projects\Major Project\Intelligent Monitoring System\Final_file\Integration\Clip_MIL\MIL_model.pth", map_location=DEVICE))

def initialize_ResNet18():
    global resnet18
    resnet18 = models.resnet18(pretrained=False)
    state_dict = torch.load(r"D:\Essentials\Projects\Major Project\Intelligent Monitoring System\Final_file\Integration\ResNet-18\resnet18-f37072fd.pth", map_location=DEVICE)
    resnet18.load_state_dict(state_dict)
    resnet18 = torch.nn.Sequential(*list(resnet18.children())[:-1])
    resnet18 = resnet18.to(DEVICE)
    resnet18.eval()

def initialize_ViViT():
    global ViViT_model

    ViViT_model = VivitForVideoClassification.from_pretrained(
        r"D:\Essentials\Projects\Major Project\Intelligent Monitoring System\Final_file\Integration\ViViT\ViViT Architecture"
    )

    checkpoint = torch.load(
        r"D:\Essentials\Projects\Major Project\Intelligent Monitoring System\Final_file\Integration\ViViT\vivit_weights.pth",
        map_location=DEVICE
    )

    ViViT_model.load_state_dict(checkpoint["model_state"], strict=True)

    ViViT_model = ViViT_model.to(DEVICE)
    ViViT_model.eval()

    if isinstance(ViViT_model.config.image_size, int):
        ViViT_model.config.image_size = (
            ViViT_model.config.image_size,
            ViViT_model.config.image_size
        )


def initialize_thread2_models():
    initialize_MiL()
    initialize_ResNet18()
    initialize_ViViT()


def stage2_pipeline():
    global stage2_running
    
    print("High-level buffer processing started.")

    while len(high_level_buffer) >0:
        print("Processing high-level buffer...")
        getting_the_most_relevant_frames()

    stage2_running = False
    print("Stage-2 thread finished.")


def compute_optical_flow(prev, next):
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    flow = cv2.resize(flow, (config.Conv_IMG_SIZE, config.Conv_IMG_SIZE))
    flow = np.clip(flow / 20.0, -1, 1)
    return flow


def final_output(vivit_results): 
    final_scores = torch.zeros_like(vivit_results[0])
    n_clips = len(vivit_results)
    
    for weighted_probs in vivit_results:
        final_scores += weighted_probs 
    
    avg_scores = final_scores / n_clips

    print(f"======================================\nMost likely class: {config.LABELS[final_scores.argmax().item()]} with avg confidence {avg_scores.max().item():.4f}\n======================================")

    run_in_thread(broadcast_message({
                    "event": "alert",
                    "type": f"{config.LABELS[final_scores.argmax().item()]}",
                    "message": f"{config.LABELS[final_scores.argmax().item()]} detected!",
                    "timestamp": datetime.datetime.now().isoformat()
                }))

    # Close the ViViT thread after processing


def preprocess(frame):
    frame = cv2.resize(frame, (config.ViViT_input_size, config.ViViT_input_size))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255.0
    frame = np.transpose(frame, (2, 0, 1))  # (C, H, W)
    return frame

def ViViT_in(selected_clips):
    print("ViViT Started processing")
    vivit_results = []
    

    for idx, (prob, clip) in enumerate(selected_clips):

        if len(clip) != 32:
            continue

        frames = np.stack([preprocess(frame) for frame in clip], axis=0)

        clip_tensor = (
            torch.from_numpy(frames)
            .float()
            .unsqueeze(0)
            .to(DEVICE)
        )

        with torch.no_grad():
            outputs = ViViT_model(pixel_values=clip_tensor)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

        # probs = probs*prob

                 
        vivit_results.append(
                probs
            )
    # print("length of ViViT",len(vivit_results))
    final_output(vivit_results)


def anomly_score(err, prob):
    return 0.1 * err + 0.9 * prob

def getting_the_most_relevant_frames():
    prob_ordered_frames = []
    raw_scores = []

    print("Entered MIL processing function, processing clips in high-level buffer...")

    TEMP = 10  # temperature for softmax weighting

    # Drain current buffer safely
    clips_to_process = []

    while len(high_level_buffer) > 0:
        clip = high_level_buffer.popleft()
        if len(clip) == 32:
            clips_to_process.append(clip)

    if len(clips_to_process) == 0:
        return

    with torch.no_grad():
        for clip in clips_to_process:
            features = []

            for frame in clip:
                frame_tensor = (
                    torch.tensor(frame, dtype=torch.float32)
                    .permute(2, 0, 1)     # (C, H, W)
                    .unsqueeze(0)         # (1, C, H, W)
                    .to(DEVICE)
                )

                feat = resnet18(frame_tensor)   # (1, 512, 1, 1)
                feat = feat.view(-1)            # (512,)
                features.append(feat)

            features = torch.stack(features).mean(dim=0).unsqueeze(0)  # (1, 512)

            score = MIL_model(features).squeeze()  # raw MIL score (logit)
            raw_scores.append(score)

            prob_ordered_frames.append((score, clip))

    # Convert raw scores → softmax normalized weights
    scores_tensor = torch.stack(raw_scores)  # shape: (N,)
    weights = torch.softmax(scores_tensor / TEMP, dim=0)

    # Replace raw score with normalized weight
    for i in range(len(prob_ordered_frames)):
        prob_ordered_frames[i] = (
            weights[i].item(),
            prob_ordered_frames[i][1]
        )

    # print("First normalized MIL weight:", prob_ordered_frames[0][0])

    TOP_K = min(10, len(prob_ordered_frames))

    selected_clips = prob_ordered_frames[:TOP_K]

    ViViT_in(selected_clips)



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



def run_video_stream():
    global main_buffer, stage2_running, temp_frame_buffer, rgb_buffer, flow_buffer, results, score_history, s, use_dynamic, high_level_buffer, prev_frame
    temp_frame_buffer = []
    rgb_buffer = []
    flow_buffer = []
    results = []
    s = False
    score_history = [] 
    use_dynamic = False
    
    prev_frame = None
    
    global latest_frame
    cap = cv2.VideoCapture(r"D:\Essentials\Projects\Major Project\Intelligent Monitoring System\testing_videos\Burglary_7_org_00.mp4")
    print("Starting")
    global smoking_last, group_last
    smoking_last = 0
    group_last = 0

    while True:
        print("Reading video stream...")
        ret, frame = cap.read()
        if not ret:
            continue
        
        if not Yolo_thread_running:
            threading.Thread(target = YOLO_inference(frame), args=(frame.copy(),)).start()
        threading.Thread(target = YOLO_inference(frame)).run()
        # Encode frame for frontend
        _, jpeg = cv2.imencode(".jpg", frame)
        with frame_lock:
            latest_frame = jpeg.tobytes()
        
        if not Sec2_thread_running:
            threading.Thread(target = section2_inference(frame), args=(frame.copy(),)).start()
        threading.Thread(target = section2_inference(frame)).run()

        if KeyboardInterrupt:
             break

    if len(high_level_buffer) > 0 and not stage2_running:
        final_scraping()
    
def final_scraping():
            print("final scraping")
            try:
                stage2_running = True
                stage2_thread = Thread(target=stage2_pipeline)
                stage2_thread.start()
            except Exception as e:
                print(f"Error starting stage2_thread: {e}")
            print ("Video stream processing completed.")






def initialize():
    global YOLO_model
    YOLO_model = YOLO(config.MODEL_PATH)
    initialize_ConvLSTM()
    initialize_thread2_models()
    run_video_stream()



def section2_inference(frame):
    global s, main_buffer, high_level_buffer, temp_frame_buffer, rgb_buffer, flow_buffer, prev_frame, use_dynamic, score_history, results
    print("Section-2 processing started.")
    if s:
            main_buffer.append(frame)
            if len(main_buffer) == 32:
                # s = False
                high_level_buffer.append(main_buffer.copy())
                # main_buffer.clear()
                main_buffer= main_buffer[-5:]
                # print("High level buffer size in thread 1:", len(high_level_buffer))
                if len(high_level_buffer) == 15:
                    if not stage2_running:
                                try:
                                    stage2_running = True
                                    stage2_thread = Thread(target=stage2_pipeline, )
                                    stage2_thread.start()
                                except Exception as e:
                                    print(f"Error starting stage2_thread: {e}")
                    # print("High level buffer reached 15 clips, starting stage-2 thread...")
                    main_buffer.clear()
                    s = False
                

    res_frame = cv2.resize(frame, (config.Conv_IMG_SIZE, config.Conv_IMG_SIZE))
    rgb = res_frame.astype(np.float32) / 255.0

    if prev_frame is None:
                prev_frame = res_frame
                return 

    flow = compute_optical_flow(prev_frame, res_frame)

    temp_frame_buffer.append(frame)
    rgb_buffer.append(rgb)
    flow_buffer.append(flow)
    prev_frame = res_frame

    if len(rgb_buffer) < config.SEQ_LEN:
                return

    if len(rgb_buffer) > config.SEQ_LEN:
                rgb_buffer.pop(0)
                flow_buffer.pop(0)
                temp_frame_buffer.pop(0)

    rgb_seq = np.array(rgb_buffer)
    flow_seq = np.array(flow_buffer)
    combined = np.concatenate([rgb_seq, flow_seq], axis=-1)
    combined = combined.transpose(0, 3, 1, 2)

    input_tensor = torch.tensor(combined, dtype=torch.float32).unsqueeze(0).to(DEVICE)


    with torch.no_grad():
                recon, prob = ConvLSTM_model(input_tensor)

    err = mse(recon, input_tensor[:, -1]).item()
    anomaly_score = anomly_score(err, prob.item())


    if not use_dynamic:
                current_threshold = config.STATIC_THRESHOLD
                score_history.append(anomaly_score)
                if len(score_history) >= config.WARM_UP_FRAMES:
                    use_dynamic = True
    else:
                # print("Thread 1 running....")
                if len(score_history) >= config.WARM_UP_FRAMES:
                    score_history.pop(0)  # Maintain window size
                score_history.append(anomaly_score)
                mean_score = np.mean(score_history)
                std_score = np.std(score_history)
                if std_score == 0:
                    std_score = 1e-6
                current_threshold = mean_score + config.Z * std_score
    if not s:    
            if anomaly_score > current_threshold:
                    print(f"Anomaly Detected! Score: {anomaly_score:.4f} | Threshold: {current_threshold:.4f}")
                    s = True
                    main_buffer.extend(temp_frame_buffer)

    results.append(anomaly_score)




if __name__ == "__main__":
     initialize()

