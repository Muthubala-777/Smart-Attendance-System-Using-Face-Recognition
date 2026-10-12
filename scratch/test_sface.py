import cv2
import numpy as np
from pathlib import Path

models_dir = Path("backend/vision/models")
sface_path = models_dir / "face_recognition_sface.onnx"
print("SFace path exists:", sface_path.exists())

recognizer = cv2.FaceRecognizerSF.create(
    model=str(sface_path),
    config="",
    backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
    target_id=cv2.dnn.DNN_TARGET_CPU
)

test_face = np.full((112, 112, 3), 120, dtype=np.uint8)
feat = recognizer.feature(test_face)
print("SFace feature shape:", feat.shape, "dtype:", feat.dtype)
print("SFace feature norm:", np.linalg.norm(feat))
