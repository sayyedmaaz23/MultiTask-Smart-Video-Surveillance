import sys
import os

# Add the root of your project (Intelligent Monitoring System/) to the path
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
import time
import config
import cv2
import numpy as np
from torch.nn import MSELoss
import threading
from threading import Thread, Event, active_count
from ultralytics import YOLO
from collections import deque
from queue import Queue, Empty
import warnings
import torchvision.models as models
from ConvLSTM_Autoencoder.HybridConvLSTM import HybridConvLSTM
from Clip_MIL.MILNet import MILNet
from transformers import VivitForVideoClassification
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
# Global variables and configurations
# shared queue
stage2_3_queue= Queue(maxsize=30)
Aclae_queue = Queue(maxsize=10)
yolo_queue = Queue(maxsize=10)

Yolo = None
Aclae = None
MILnet = None
RESnet = None
ViViT = None


stop_event      = Event()   # signals all threads to shut down cleanly
stage2_running  = False     # guarded — only Thread 3 writes this

mse = MSELoss()




# feeder thread functions================================================
def feeder():
    # cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    cap = cv2.VideoCapture(r"C:\Users\sayye\Downloads\test_video.mp4")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("[Feeder] End of stream.")
            break

        # Non-blocking puts with a small timeout so we don't stall the feeder
        # if a downstream queue is temporarily full.
        try:
            yolo_queue.put(frame.copy(), timeout=1)
        except Exception:
            pass  # drop frame rather than block — YOLO can miss frames

        try:
            Aclae_queue.put(frame.copy(), timeout=1)
        except Exception:
            pass
        # print(f"[Feeder] Frame put in YOLO queue. Queue size: {yolo_queue.qsize()}")
        # print(f"[Feeder] Frame put in ACLAE queue. Queue size: {Aclae_queue.qsize()}")

    
    # Send stop sentinels
    yolo_queue.put(None)
    Aclae_queue.put(None)
    cap.release()
    print("[Feeder] Done.")






# ACLAE thread fucntions==============================================
def ACLAE_inference():
    global stage2_running

    print("[ConvLSTM] Started.")

    # All state is local — no shared mutation needed
    temp_frame_buffer = []
    rgb_buffer        = []
    flow_buffer       = []
    score_history     = []
    use_dynamic       = False
    prev_frame        = None
    s                 = False      # are we currently collecting an anomaly clip?
    main_buffer       = []
    clips_count       = 0

    while True:
        if KeyboardInterrupt:
            break
        try:
            frame = Aclae_queue.get(timeout=2)
            # print(f"[ConvLSTM] Frame received for ACLAE inference. Queue size: {Aclae_queue.qsize()}====")
        except Empty:
            if stop_event.is_set():
                break
            continue

        if frame is None:           # sentinel
            break

        # ── Clip collection (active anomaly window) ──────────────────────────
        if s:
            main_buffer.append(frame)
            if len(main_buffer) == 32:
                print(f"Clips count= {clips_count+1} and buffer length = {stage2_3_queue.qsize()}")
                stage2_3_queue.put(main_buffer.copy())
                clips_count+=1
                main_buffer = main_buffer[-5:]          # 5-frame overlap
                if stage2_3_queue.qsize() >= 5 :
                    print("Attempting....")
                    # _launch_stage2()
                    # print("Sec2 thread should get started and ViViT result should get printed")
                    # print(f"threads running are {active_count()}")
                    # for t in threading.enumerate():
                    #     print(t.name, t.is_alive())
                    #     print(t, t._target)
                    # Sec2_thread = Thread(target=stage2_pipeline, daemon=True)                        
                    if Sec2_thread.is_alive():
                        pass
                    else:
                        print("Starting the thread..")
                        Sec2_thread.start()
                        stage2_running = True
                    if clips_count>=15:
                        print("Clearing the status")
                        main_buffer.clear()
                        clips_count=0
                        s = False
                    if not stage2_running:
                        new_sec2 = Thread(target=stage2_pipeline, daemon=True)
                        new_sec2.start()
                        stage2_running = True

        # ── Sliding-window feature prep ───────────────────────────────────────
        res_frame = cv2.resize(frame, (config.Conv_IMG_SIZE, config.Conv_IMG_SIZE))
        rgb       = res_frame.astype(np.float32) / 255.0

        if prev_frame is None:
            prev_frame = res_frame
            Aclae_queue.task_done()
            continue

        flow = compute_optical_flow(prev_frame, res_frame)
        temp_frame_buffer.append(frame)
        rgb_buffer.append(rgb)
        flow_buffer.append(flow)
        prev_frame = res_frame

        if len(rgb_buffer) < config.SEQ_LEN:
            Aclae_queue.task_done()
            continue

        # Keep window at exactly SEQ_LEN
        if len(rgb_buffer) > config.SEQ_LEN:
            rgb_buffer.pop(0)
            flow_buffer.pop(0)
            temp_frame_buffer.pop(0)

        # print(f"[ConvLSTM] Prepared clip for inference. RGB buffer size: {len(rgb_buffer)}")
        # ── ConvLSTM inference ────────────────────────────────────────────────
        combined = np.concatenate(
            [np.array(rgb_buffer), np.array(flow_buffer)], axis=-1
        ).transpose(0, 3, 1, 2)

        input_tensor = (torch.tensor(combined, dtype=torch.float32)
                        .unsqueeze(0).to(DEVICE))

        with torch.no_grad():
            recon, prob = Aclae(input_tensor)

        err           = mse(recon, input_tensor[:, -1]).item()
        anomaly_score = 0.1 * err + 0.9 * prob.item()

        # ── Adaptive threshold ────────────────────────────────────────────────
        if not use_dynamic:
            current_threshold = config.STATIC_THRESHOLD
            score_history.append(anomaly_score)
            if len(score_history) >= config.WARM_UP_FRAMES:
                use_dynamic = True
        else:
            if len(score_history) >= config.WARM_UP_FRAMES:
                score_history.pop(0)
            score_history.append(anomaly_score)
            mean_s = np.mean(score_history)
            std_s  = max(np.std(score_history), 1e-6)
            current_threshold = mean_s + config.Z * std_s

        # ── Trigger anomaly collection ────────────────────────────────────────
        if not s and anomaly_score > current_threshold:
            print(f"[ConvLSTM] ⚠ Anomaly! score={anomaly_score:.4f} "
                  f"threshold={current_threshold:.4f}")
            s = True
            main_buffer.extend(temp_frame_buffer)   # seed buffer with recent frames

        Aclae_queue.task_done()

    # ── End of stream: flush whatever is left ─────────────────────────────────
    if stage2_3_queue.qsize() > 0 and not stage2_running:
        print("[ConvLSTM] End of stream reached. Flushing remaining clips to Stage-2...")
        # Sec2_thread.start()

    print("[ConvLSTM] Done.")




# def _launch_stage2():
#     global stage2_running
#     stage2_running = True
#     print("[ConvLSTM] Launching Stage-2 thread...")
#     Thread(target=stage2_pipeline, daemon=True).start()

# ==== Stage 2 model thread functions =====================================
def stage2_pipeline():
    global stage2_running
    stage2_running = True
    while not stop_event.is_set():
        if stage2_3_queue.qsize() >= 5:
            _run_mil_and_vivit()
        else:
            time.sleep(0.1)   # idle wait
    stage2_running = False

def _run_mil_and_vivit():
    print("MILnet is starting")
    prob_ordered_frames, raw_scores = [], []
    idx=0
    # Drain buffer in one pass
    clips = []
    while stage2_3_queue.qsize() > 0:
        print("Extracting the clips from the common queue")
        clip = stage2_3_queue.get()
        if len(clip) == 32:
            clips.append(clip)

    if not clips:
        return

    print("MIL and Resnet now working")
    # ── MIL + ResNet-18 scoring ───────────────────────────────────────────────
    with torch.no_grad():
        for clip in clips:
            idx+=1
            print(f"Processing clip {idx+1}/{len(clips)} with MIL and ResNet...")
        # Stack all 32 frames into a single batch tensor
            batch = torch.stack([
                torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)
                for frame in clip
            ]).to(DEVICE)                          # (32, C, H, W)

            feats = RESnet(batch)                  # single forward pass → (32, 512, 1, 1)
            feat_mean = feats.view(32, -1).mean(dim=0).unsqueeze(0)  # (1, 512)

            score = MILnet(feat_mean).squeeze()
            raw_scores.append(score)
            prob_ordered_frames.append((score, clip))
    weights = torch.softmax(torch.stack(raw_scores) / config.TEMP, dim=0)
    for i in range(len(prob_ordered_frames)):
        prob_ordered_frames[i] = (weights[i].item(), prob_ordered_frames[i][1])

    selected = prob_ordered_frames[:min(10, len(prob_ordered_frames))]

    # ── ViViT classification ──────────────────────────────────────────────────
    vivit_results = []
    print("Vivit about to start...")
    for _, clip in selected:
        print("ViVit speed")
        if len(clip) != 32:
            continue
        frames = np.stack([_preprocess_vivit(f) for f in clip], axis=0)
        clip_t = (torch.from_numpy(frames).float().unsqueeze(0).to(DEVICE))

        with torch.no_grad():
            probs = torch.softmax(ViViT(pixel_values=clip_t).logits, dim=1)
        vivit_results.append(probs)

    if vivit_results:
        _final_output(vivit_results)


def _preprocess_vivit(frame):
    frame = cv2.resize(frame, (config.ViViT_input_size, config.ViViT_input_size))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(frame, (2, 0, 1))


def _final_output(vivit_results):
    total = torch.zeros_like(vivit_results[0])
    for p in vivit_results:
        total += p
    avg = total / len(vivit_results)
    label = config.LABELS[avg.argmax().item()]
    conf  = avg.max().item()
    print(f"[ViViT] ══ Detected: {label}  (confidence: {conf:.4f}) ══")
    # TODO: route to your FastAPI broadcast here


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def compute_optical_flow(prev, nxt):
    flow = cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(nxt,  cv2.COLOR_BGR2GRAY),
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    flow = cv2.resize(flow, (config.Conv_IMG_SIZE, config.Conv_IMG_SIZE))
    return np.clip(flow / 20.0, -1, 1)


# Yolo thread functions=================================================
def yolo_worker():
    print("[YOLO] Started.")
    while True:
        try:
            frame = yolo_queue.get(timeout=2)
        except Empty:
            if stop_event.is_set():
                break
            continue

        if frame is None:           # sentinel → shut down
            break

        results = Yolo(frame, verbose=False)[0]
        person_centers, cigarette_centers = [], []

        for box in results.boxes:
            cls   = int(box.cls[0])
            label = Yolo.names[cls].lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if label == "person":
                person_centers.append((cx, cy))
            elif label == "cigarette":
                cigarette_centers.append((cx, cy))

        # ── Smoking check ──
        smoking = any(
            np.linalg.norm(np.array(p) - np.array(c)) < 80   # replace with config value
            for p in person_centers for c in cigarette_centers
        )
        if smoking:
            print("[YOLO] ⚠ Smoking detected!")
            pass
            # TODO: hook your alert/broadcast here

        # ── Crowd check ──
        if len(person_centers) >= 3:                          # replace with config value
            arr = np.array(person_centers)
            dist_matrix = np.linalg.norm(arr[:, None] - arr[None, :], axis=2)
            if np.any((dist_matrix < 150).sum(axis=1) >= 3):  # replace with config value
                print("[YOLO] ⚠ Group detected!")
                pass
                # TODO: hook your alert/broadcast here

        yolo_queue.task_done()

    print("[YOLO] Done.")



# thread initiailization and start funcstion=====================================================
def initialization(ACLAE_path, MILNET_Path, ResNet_path, Yolo_path, ViViT_wt_path, Vivit_architecture_path):
    global Aclae, MILnet, Yolo, RESnet, ViViT
    print("Initializing models...")
    Yolo =  YOLO(Yolo_path)
    print("YOLO Initialized")
    Aclae = HybridConvLSTM().to(DEVICE)
    Aclae.load_state_dict(torch.load(ACLAE_path, map_location=DEVICE) )
    Aclae.eval()
    print("ConvLSTMAutoencoder Initialized")
    MILnet = MILNet(512).to(DEVICE)
    MILnet.load_state_dict(torch.load(MILNET_Path, map_location=DEVICE))
    MILnet.eval()
    print("MILnet Initialized")
    RESnet = models.resnet18(pretrained = False).to(DEVICE)
    RESnet.load_state_dict(torch.load(ResNet_path, map_location=DEVICE))
    RESnet = torch.nn.Sequential(*list(RESnet.children())[:-1])
    RESnet.eval()
    print("Resnet Initialized")
    ViViT = VivitForVideoClassification.from_pretrained(Vivit_architecture_path)
    checkpoint = torch.load(ViViT_wt_path, map_location=DEVICE)
    ViViT.load_state_dict(checkpoint["model_state"], strict=True)
    ViViT = ViViT.to(DEVICE)   # ← this line was missing
    ViViT.eval()
    print("ViViT Initialized")
    print("Models initialized successfully!")



    
# main function includes initialization of models along with threads 
if __name__ == "__main__":
    import torch
    print(torch.cuda.device_count())
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    print("Starting Intelligent Monitoring System...")
    # Initialize models
    initialization(
        config.ACLAE_path,
        config.MILNET_Path,
        config.ResNet_path,
        config.Yolo_path,
        config.ViViT_wt_path,
        config.Vivit_architecture_path
    )

    # spawn threads
    print("initializing threads...")
    feeder_thread = Thread(target=feeder, daemon=True)
    Yolo_thread = Thread(target=yolo_worker, daemon=True)
    Aclae_thread = Thread(target=ACLAE_inference, daemon=True)
    Sec2_thread = Thread(target=stage2_pipeline, daemon=True)
    print("Threads initialized successfully! Starting video stream...")

    # if KeyboardInterrupt():
    #     print("Keyboard interrupt received. Shutting down...")
    #     stop_event.set()
    #     sys.exit(0)

    feeder_thread.start()
    Yolo_thread.start()
    Aclae_thread.start()

    feeder_thread.join()
    Yolo_thread.join()
    Aclae_thread.join()
    stop_event.set() 