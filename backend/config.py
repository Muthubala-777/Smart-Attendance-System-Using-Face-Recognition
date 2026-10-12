import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

SAMPLES_DIR = DATA_DIR / "face_samples"
SAMPLES_DIR.mkdir(exist_ok=True, parents=True)

REPORTS_DIR = DATA_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True, parents=True)

DB_PATH = DATA_DIR / "attendance_system.db"
DATABASE_URL = f"sqlite:///{DB_PATH.as_posix()}"

# Security
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "smart-attendance-system-arcface-deepmind-secret-key-2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Vision & Recognition Config Defaults
DEFAULT_CONFIGS = {
    "recognition_threshold": 0.65,        # Cosine similarity threshold for ArcFace
    "temporal_frames_required": 3,       # Consecutive frames required for identity confirmation
    "temporal_window_size": 5,           # Sliding window size
    "min_face_size": 70,                 # Minimum bounding box size in pixels
    "blur_threshold": 65.0,              # Laplacian variance threshold
    "min_brightness": 35.0,              # Minimum acceptable mean luminance
    "max_brightness": 235.0,             # Maximum acceptable mean luminance
    "liveness_ear_threshold": 0.22,      # Eye aspect ratio threshold for blink detection
    "liveness_motion_threshold": 1.2,    # Micro-motion / pose change threshold
    "scheduler_poll_seconds": 15,        # Timetable check interval
}
