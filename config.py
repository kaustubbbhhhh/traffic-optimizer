"""
Configuration settings for the Traffic Flow Optimizer
"""
import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "weights"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if not exist
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# YOLOv8 Configuration
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8s.pt")  # small model - better accuracy
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.35"))  # lower threshold for more detections
YOLO_IOU_THRESHOLD = float(os.getenv("YOLO_IOU_THRESHOLD", "0.5"))

# Vehicle classes from COCO dataset
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle", 
    5: "bus",
    7: "truck",
    1: "bicycle"
}

# Emergency vehicle detection (color-based)
EMERGENCY_COLORS = {
    "red": [(0, 100, 100), (10, 255, 255)],      # Fire truck, ambulance lights
    "blue": [(100, 100, 100), (130, 255, 255)],  # Police lights
    "white": [(0, 0, 200), (180, 30, 255)]       # Ambulance body
}

# Traffic density thresholds
DENSITY_THRESHOLDS = {
    "low": 10,       # vehicles < 10
    "medium": 25,    # 10 <= vehicles < 25
    "high": 40,      # 25 <= vehicles < 40
    "critical": 100  # vehicles >= 40
}

# Signal timing (seconds)
SIGNAL_TIMING = {
    "low": {"green": 15, "yellow": 3, "red": 45},
    "medium": {"green": 30, "yellow": 3, "red": 30},
    "high": {"green": 45, "yellow": 3, "red": 15},
    "critical": {"green": 60, "yellow": 3, "red": 10}
}

# LSTM Model Configuration
LSTM_SEQUENCE_LENGTH = 24  # Hours of historical data
LSTM_PREDICTION_HORIZON = 6  # Hours to predict ahead
LSTM_HIDDEN_UNITS = 64
LSTM_EPOCHS = 100
LSTM_BATCH_SIZE = 32

# Video Processing
VIDEO_FPS = 30
FRAME_SKIP = 5  # Process every 5th frame for efficiency
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 480

# Dashboard refresh rate (seconds)
DASHBOARD_REFRESH_RATE = 2

# Camera feeds (example URLs - replace with actual CCTV feeds)
CAMERA_FEEDS = {
    "intersection_1": {
        "name": "Main St & 1st Ave",
        "url": "rtsp://camera1.local/stream",
        "type": "rtsp"
    },
    "intersection_2": {
        "name": "Highway Junction",
        "url": "rtsp://camera2.local/stream",
        "type": "rtsp"
    },
    "intersection_3": {
        "name": "Downtown Square",
        "url": "rtsp://camera3.local/stream",
        "type": "rtsp"
    },
    "intersection_4": {
        "name": "Industrial Zone",
        "url": "rtsp://camera4.local/stream",
        "type": "rtsp"
    }
}
