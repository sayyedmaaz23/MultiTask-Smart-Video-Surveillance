MODEL_PATH = "models/best_v11.pt"
VIDEO_SOURCE = 0
GROUP_DISTANCE = 150
GROUP_COUNT = 5
SMOKING_DISTANCE = 100
ALERT_SOUND_FREQ = 1000
ALERT_SOUND_DUR = 500
COOLDOWN = 5
LOG_FILE = "logs.txt"
SNAPSHOT_DIR = "static/snapshots"
STATIC_THRESHOLD = 0.50
TEMP = 10
Z = 3.5
WARM_UP_FRAMES = 30
Conv_IMG_SIZE=  128
ViViT_input_size = 224
SEQ_LEN = 16
CHANNELS = 5
VIVIT_CONF_THRESHOLD = 0.3
CONF_THRESHOLD = 0.45
SMOKING_FRAMES_REQUIRED = 5
HAND_MOUTH_DISTANCE = 50
SHOW_DEBUG_POINTS = False
STREAK_REQUIRED = 1
CLIP_FPS = 24
stage2_thread = None
stage2_running = False
LABELS = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "Normal", "RoadAccidents",
    "Robbery", "Shooting", "Normal", "Stealing", "Vandalism"
]
ACLAE_path = r"ConvLSTM_Autoencoder\ConvLSTM_Autoencoder.pth"
MILNET_Path = r"Clip_MIL\MIL_model.pth"
ResNet_path = r"ResNet-18\resnet18-f37072fd.pth"
Yolo_path = r"Yolo\best_v11.pt"
ViViT_wt_path = r"ViViT\vivit_weights.pth"
Vivit_architecture_path = r"ViViT\ViViT Architecture"