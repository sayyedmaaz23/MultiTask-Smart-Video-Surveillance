import sys
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
import time
import datetime
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

# ── Shared queues ──────────────────────────────────────────────────────────────
stage2_3_queue = Queue(maxsize=30)
Aclae_queue    = Queue(maxsize=10)
yolo_queue     = Queue(maxsize=10)

# ── Model handles (set during initialization) ──────────────────────────────────
Yolo   = None
Aclae  = None
MILnet = None
RESnet = None
ViViT  = None

# ── Coordination ───────────────────────────────────────────────────────────────
stop_event     = Event()
stage2_running = False
# Lock so only one thread can spawn / inspect Sec2_thread at a time
_sec2_lock     = threading.Lock()
_sec2_thread   = None          # always access via _sec2_lock

mse = MSELoss()


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


def _ensure_stage2_running():
    """
    Spawn a fresh Stage-2 thread if one is not already alive.
    Protected by _sec2_lock so concurrent callers can't double-spawn.
    """
    global stage2_running, _sec2_thread
    with _sec2_lock:
        if _sec2_thread is not None and _sec2_thread.is_alive():
            return                          # already running — nothing to do
        _sec2_thread = Thread(target=stage2_pipeline, daemon=True, name="Stage2")
        _sec2_thread.start()
        stage2_running = True
        print("[ACLAE] Stage-2 thread spawned.")


# ══════════════════════════════════════════════════════════════════════════════
# THREAD 1 — FEEDER
# ══════════════════════════════════════════════════════════════════════════════

def feeder(video_source=r"D:\Essentials\Projects\Major Project\Intelligent Monitoring System\testing_videos\Normal_Videos_11_org_00.mp4"):
    src = video_source or config.VIDEO_SOURCE
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"[Feeder] ERROR: Cannot open video source '{src}'. Sending sentinels.")
        yolo_queue.put(None)
        Aclae_queue.put(None)
        return

    print(f"[Feeder] Started — source: {src}")

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("[Feeder] End of stream.")
            break

        try:
            yolo_queue.put(frame.copy(), timeout=1)
        except Exception:
            pass  # drop frame — YOLO can miss frames

        try:
            Aclae_queue.put(frame.copy(), timeout=1)
        except Exception:
            pass

    # Sentinels shut down consumer threads cleanly
    yolo_queue.put(None)
    Aclae_queue.put(None)
    cap.release()
    print("[Feeder] Done.")


# ══════════════════════════════════════════════════════════════════════════════
# THREAD 2 — YOLO
# ══════════════════════════════════════════════════════════════════════════════

def yolo_worker():
    if Yolo is None:
        print("[YOLO] ERROR: Model not initialized. Thread exiting.")
        return

    print("[YOLO] Started.")

    while True:
        try:
            frame = yolo_queue.get(timeout=2)
        except Empty:
            if stop_event.is_set():
                break
            continue

        if frame is None:       # sentinel
            break

        try:
            results = Yolo(frame, verbose=False)[0]
        except Exception as e:
            print(f"[YOLO] Inference error: {e}")
            yolo_queue.task_done()
            continue

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

        # ── Smoking check ──────────────────────────────────────────────────────
        smoking = any(
            np.linalg.norm(np.array(p) - np.array(c)) < config.SMOKING_DISTANCE
            for p in person_centers for c in cigarette_centers
        )
        if smoking:
            print("[YOLO] ⚠ Smoking detected!")
            # pass
            # TODO: hook your alert/broadcast here

        # ── Crowd check ────────────────────────────────────────────────────────
        if len(person_centers) >= config.GROUP_COUNT:
            arr = np.array(person_centers)
            dist_matrix = np.linalg.norm(arr[:, None] - arr[None, :], axis=2)
            if np.any((dist_matrix < config.GROUP_DISTANCE).sum(axis=1) >= config.GROUP_COUNT):
                print("[YOLO] ⚠ Group detected!")
                # TODO: hook your alert/broadcast here

        yolo_queue.task_done()

    print("[YOLO] Done.")


# ══════════════════════════════════════════════════════════════════════════════
# THREAD 3 — ACLAE (ConvLSTM anomaly detector)
# ══════════════════════════════════════════════════════════════════════════════

def ACLAE_inference():
    if Aclae is None:
        print("[ACLAE] ERROR: Model not initialized. Thread exiting.")
        return

    print("[ConvLSTM] Started.")

    temp_frame_buffer = []
    rgb_buffer        = []
    flow_buffer       = []
    score_history     = []
    use_dynamic       = False
    prev_frame        = None
    s                 = False
    main_buffer       = []
    clips_count       = 0
    anomaly_start_ts  = None   # wall-clock time when anomaly was first detected
    anomaly_end_ts    = None   # wall-clock time when the collection window closes

    while True:
        try:
            frame = Aclae_queue.get(timeout=2)
        except Empty:
            if stop_event.is_set():
                break
            continue

        if frame is None:       # sentinel
            break

        # ── Clip collection (active anomaly window) ────────────────────────────
        if s:
            main_buffer.append(frame)
            if len(main_buffer) == 32:
                # print(f"[ACLAE] Clips count={clips_count + 1}  queue={stage2_3_queue.qsize()}")
                try:
                    stage2_3_queue.put(
                        (main_buffer.copy(), anomaly_start_ts, datetime.datetime.now()),
                        timeout=2,
                    )
                except Exception:
                    print("[ACLAE] WARNING: stage2 queue full — clip dropped.")
                clips_count += 1
                main_buffer = main_buffer[-5:]      # 5-frame overlap

                if stage2_3_queue.qsize() >= 15:
                    print("[ACLAE] Clip limit reached — resetting collection.")
                    anomaly_end_ts = datetime.datetime.now()
                    print(f"[ACLAE] Anomaly window: "
                          f"{anomaly_start_ts.strftime('%H:%M:%S')} → "
                          f"{anomaly_end_ts.strftime('%H:%M:%S')}")
                    _ensure_stage2_running()
                    main_buffer.clear()
                    clips_count      = 0
                    s                = False
                    anomaly_start_ts = None
                    anomaly_end_ts   = None

        # ── Sliding-window feature prep ────────────────────────────────────────
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

        if len(rgb_buffer) > config.SEQ_LEN:
            rgb_buffer.pop(0)
            flow_buffer.pop(0)
            temp_frame_buffer.pop(0)

        # ── ConvLSTM inference ─────────────────────────────────────────────────
        try:
            combined = np.concatenate(
                [np.array(rgb_buffer), np.array(flow_buffer)], axis=-1
            ).transpose(0, 3, 1, 2)

            input_tensor = (torch.tensor(combined, dtype=torch.float32)
                            .unsqueeze(0).to(DEVICE))

            with torch.no_grad():
                recon, prob = Aclae(input_tensor)

            err           = mse(recon, input_tensor[:, -1]).item()
            anomaly_score = 0.1 * err + 0.9 * prob.item()

        except Exception as e:
            print(f"[ACLAE] Inference error: {e}")
            Aclae_queue.task_done()
            continue

        # ── Adaptive threshold ─────────────────────────────────────────────────
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

        # ── Trigger anomaly collection ─────────────────────────────────────────
        if not s and anomaly_score > current_threshold:
            print(f"[ConvLSTM] ⚠ Anomaly! score={anomaly_score:.4f} "
                  f"threshold={current_threshold:.4f}")
            s                = True
            anomaly_start_ts = datetime.datetime.now()
            main_buffer.extend(temp_frame_buffer)

        Aclae_queue.task_done()

    # ── End of stream flush ────────────────────────────────────────────────────
    if stage2_3_queue.qsize() > 0 and not stage2_running:
        print("[ACLAE] End of stream — flushing remaining clips to Stage-2.")
        _ensure_stage2_running()

    print("[ConvLSTM] Done.")


# ══════════════════════════════════════════════════════════════════════════════
# THREAD 4 — STAGE-2  (MIL + ResNet-18 + ViViT)
# ══════════════════════════════════════════════════════════════════════════════

def stage2_pipeline():
    global stage2_running
    stage2_running = True
    print("[Stage-2] Started.")

    while not stop_event.is_set():
        if stage2_3_queue.qsize() >= 15:
            try:
                _run_mil_and_vivit()
            except Exception as e:
                print(f"[Stage-2] ERROR in _run_mil_and_vivit: {e}")
        else:
            time.sleep(0.1)

    stage2_running = False
    print("[Stage-2] Done.")


def _run_mil_and_vivit():
    # Drain queue
    entries = []
    while stage2_3_queue.qsize() > 0:
        try:
            item = stage2_3_queue.get(timeout=1)
            clip, start_ts, end_ts = item
            if len(clip) == 32:
                entries.append((clip, start_ts, end_ts))
        except Empty:
            break
    
    if stage2_3_queue.qsize() ==0:
            print("[Stage-2] Queue drained. Current window is empty")

    if not entries:
        return

    # ── Group clips by anomaly window (keyed on start_ts) ─────────────────────
    windows: dict = {}
    for clip, start_ts, end_ts in entries:
        key = start_ts  # datetime object — unique per anomaly trigger
        if key not in windows:
            windows[key] = {"clips": [], "start_ts": start_ts, "end_ts": end_ts}
        windows[key]["clips"].append(clip)
        # Keep the latest end_ts seen for this window
        if end_ts is not None:
            windows[key]["end_ts"] = end_ts

    # ── Process each window independently ─────────────────────────────────────
    for window in windows.values():
        _process_window(
            window["clips"],
            window["start_ts"],
            window["end_ts"],
        )


def _process_window(clips, start_ts, end_ts):
    """Run MIL + ResNet + ViViT for one anomaly window."""
    prob_ordered_clips, raw_scores = [], []

    with torch.no_grad():
        for idx, clip in enumerate(clips):
            try:
                batch = torch.stack([
                    torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)
                    for frame in clip
                ]).to(DEVICE)
                feats     = RESnet(batch)
                feat_mean = feats.view(32, -1).mean(dim=0).unsqueeze(0)
                score     = MILnet(feat_mean).squeeze()
                raw_scores.append(score)
                prob_ordered_clips.append((score, clip))
            except Exception as e:
                print(f"[Stage-2] ResNet/MIL error on clip {idx + 1}: {e}")
                continue

    if not raw_scores:
        return

    weights = torch.softmax(torch.stack(raw_scores) / config.TEMP, dim=0)
    for i in range(len(prob_ordered_clips)):
        prob_ordered_clips[i] = (weights[i].item(), prob_ordered_clips[i][1])

    selected     = prob_ordered_clips[:min(10, len(prob_ordered_clips))]
    prob_ordered_clips.clear()  # free memory
    print("Old window cleared")
    vivit_results = []

    for _, clip in selected:
        if len(clip) != 32:
            continue
        try:
            frames = np.stack([_preprocess_vivit(f) for f in clip], axis=0)
            clip_t = torch.from_numpy(frames).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(ViViT(pixel_values=clip_t).logits, dim=1)
            vivit_results.append(probs)
        except Exception as e:
            print(f"[Stage-2] ViViT error: {e}")
            continue
    
    selected.clear()
    print("Sent for final results and Selected clips cleared")

    if vivit_results:
        _final_output(vivit_results, start_ts, end_ts)

def _preprocess_vivit(frame):
    frame = cv2.resize(frame, (config.ViViT_input_size, config.ViViT_input_size))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(frame, (2, 0, 1))


def _final_output(vivit_results, start_ts: datetime.datetime, end_ts: datetime.datetime):
    total = torch.zeros_like(vivit_results[0])
    for p in vivit_results:
        total += p
    avg   = total / len(vivit_results)
    label = config.LABELS[avg.argmax().item()]
    conf  = avg.max().item()
    if conf < config.VIVIT_CONF_THRESHOLD:
        label = "Unknown"
        conf = 0.0
    start_str = start_ts.strftime("%H:%M:%S") if start_ts else "unknown"
    end_str   = end_ts.strftime("%H:%M:%S")   if end_ts   else "unknown"

    print(f"[ViViT] ══ Detected: {label}  (confidence: {conf:.4f})")
    print(f"[ViViT]    Clip window: {start_str} → {end_str}")

    # Broadcast to FastAPI WebSocket clients
    from endpoints_1 import broadcast_message, alert_settings
    import asyncio
    if alert_settings.get(label, True):
        asyncio.run(broadcast_message({
            "event":      "alert",
            "type":       label,
            "message":    f"{label} detected!",
            "confidence": round(conf, 4),
            "clip_start": start_str,
            "clip_end":   end_str,
            "timestamp":  datetime.datetime.now().isoformat(),
        }))


# ══════════════════════════════════════════════════════════════════════════════
# INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def initialization(ACLAE_path, MILNET_Path, ResNet_path, Yolo_path,
                   ViViT_wt_path, Vivit_architecture_path):
    global Aclae, MILnet, Yolo, RESnet, ViViT
    print(DEVICE)
    print("Initializing models...")

    Yolo = YOLO(Yolo_path)
    print("YOLO initialized.")

    Aclae = HybridConvLSTM().to(DEVICE)
    Aclae.load_state_dict(torch.load(ACLAE_path, map_location=DEVICE,
                                     weights_only=False))
    Aclae.eval()
    print("ConvLSTM Autoencoder initialized.")

    MILnet = MILNet(512).to(DEVICE)
    MILnet.load_state_dict(torch.load(MILNET_Path, map_location=DEVICE,
                                      weights_only=False))
    MILnet.eval()
    print("MILnet initialized.")

    RESnet = models.resnet18(pretrained=False).to(DEVICE)
    RESnet.load_state_dict(torch.load(ResNet_path, map_location=DEVICE,
                                      weights_only=False))
    RESnet = torch.nn.Sequential(*list(RESnet.children())[:-1])
    RESnet.eval()
    print("ResNet-18 initialized.")

    ViViT = VivitForVideoClassification.from_pretrained(Vivit_architecture_path)
    checkpoint = torch.load(ViViT_wt_path, map_location=DEVICE, weights_only=False)
    ViViT.load_state_dict(checkpoint["model_state"], strict=True)
    ViViT = ViViT.to(DEVICE)
    ViViT.eval()
    print("ViViT initialized.")

    print("All models initialized successfully.")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"CUDA devices: {torch.cuda.device_count()}")
    print(f"Using device: {DEVICE}")
    print("Starting Intelligent Monitoring System...")

    initialization(
        config.ACLAE_path,
        config.MILNET_Path,
        config.ResNet_path,
        config.Yolo_path,
        config.ViViT_wt_path,
        config.Vivit_architecture_path,
    )

    print("Spawning threads...")
    feeder_thread = Thread(target=feeder,          daemon=True, name="Feeder")
    yolo_thread   = Thread(target=yolo_worker,     daemon=True, name="YOLO")
    aclae_thread  = Thread(target=ACLAE_inference, daemon=True, name="ACLAE")

    feeder_thread.start()
    yolo_thread.start()
    aclae_thread.start()

    try:
        feeder_thread.join()
        yolo_thread.join()
        aclae_thread.join()
    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt — shutting down.")
    finally:
        stop_event.set()
        # Give Stage-2 a moment to finish its current batch
        with _sec2_lock:
            if _sec2_thread is not None:
                _sec2_thread.join(timeout=30)
        print("[Main] Shutdown complete.")