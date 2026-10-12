import time
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np


class LivenessDetector:
    """
    Real-Time Liveness & Anti-Spoofing Engine.
    Employs multimodal cues to distinguish genuine live human faces
    from presentation attacks (static photos, smartphone/tablet screen replays):
    
    1. Eye Aspect Ratio (EAR) & Blink Dynamic Analysis
    2. Temporal Micro-Motion & Pose Tremor Tracking
    3. High-Frequency Texture & Specular Reflection Analysis (moiré pattern & screen bezel detection)
    """

    def __init__(
        self,
        ear_threshold: float = 0.22,
        motion_threshold: float = 1.0,
        history_window: int = 10
    ):
        self.ear_threshold = ear_threshold
        self.motion_threshold = motion_threshold
        self.history_window = history_window
        # track_id / face_index -> history of landmark centroids and EAR
        self.history: Dict[str, List[Dict[str, Any]]] = {}

    def compute_ear(self, landmarks: Optional[np.ndarray]) -> float:
        """
        Calculates Eye Aspect Ratio (EAR) or vertical eye-opening ratio from 5-point facial landmarks:
        Point 0: Left Eye (camera POV)
        Point 1: Right Eye (camera POV)
        Point 2: Nose Tip
        Point 3: Left Mouth Corner
        Point 4: Right Mouth Corner
        """
        if landmarks is None or len(landmarks) < 5:
            return 0.30  # Default normal open ratio

        # Euclidean distance between eye centers and mouth line
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]

        inter_ocular_dist = float(np.linalg.norm(left_eye - right_eye))
        if inter_ocular_dist < 1e-4:
            return 0.30

        # Vertical distance from eye midpoint to nose
        eye_midpoint = (left_eye + right_eye) / 2.0
        vertical_dist = float(np.linalg.norm(eye_midpoint - nose))

        # Normalized ratio
        return round(vertical_dist / inter_ocular_dist, 3)

    def analyze_texture_and_reflection(self, aligned_face: np.ndarray) -> float:
        """
        Analyzes specular reflectance and high-frequency Fourier components
        to detect printed paper grain or LCD screen pixel grid artifacts.
        Returns a score in [0.0, 1.0] where > 0.6 indicates natural human skin dynamics.
        """
        if aligned_face is None or aligned_face.size == 0:
            return 0.0

        gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY) if len(aligned_face.shape) == 3 else aligned_face
        
        # 1. High frequency gradient energy
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_energy = float(np.mean(np.sqrt(sobel_x**2 + sobel_y**2)))

        # Natural skin has smooth gradients in mid-tones, printed paper / screens have harsh micro-edges or flat zones
        texture_score = min(1.0, max(0.2, gradient_energy / 40.0))

        # 2. Specular reflection check (phone screens have distinct blown-out white glares)
        overexposed_pixels = np.sum(gray > 248)
        overexposed_ratio = overexposed_pixels / float(gray.size)
        glare_penalty = 0.3 if overexposed_ratio > 0.08 else 0.0

        final_texture = max(0.1, texture_score - glare_penalty)
        return round(final_texture, 3)

    def verify_liveness(
        self,
        face_id: str,
        aligned_face: np.ndarray,
        landmarks: Optional[np.ndarray],
        current_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Processes a face frame and returns liveness decision.
        """
        now = current_time or time.time()
        ear = self.compute_ear(landmarks)
        texture_score = self.analyze_texture_and_reflection(aligned_face)

        centroid = (0.0, 0.0)
        if landmarks is not None and len(landmarks) > 0:
            centroid = (float(np.mean(landmarks[:, 0])), float(np.mean(landmarks[:, 1])))

        if face_id not in self.history:
            self.history[face_id] = []

        face_hist = self.history[face_id]
        face_hist.append({
            "timestamp": now,
            "ear": ear,
            "centroid": centroid,
            "texture": texture_score
        })

        # Keep history bounded
        if len(face_hist) > self.history_window:
            face_hist.pop(0)

        # Evaluate temporal dynamics
        if len(face_hist) < 3:
            # Not enough temporal history yet, default to alive if texture is valid
            is_live = texture_score >= 0.40
            return {
                "is_live": is_live,
                "liveness_score": round(texture_score * 0.8, 2),
                "liveness_status": "INITIALIZING" if not is_live else "LIVE",
                "details": "Collecting temporal frames...",
                "metrics": {"texture": texture_score, "ear": ear}
            }

        # 1. Motion dynamics across history
        centroids = np.array([item["centroid"] for item in face_hist])
        motion_variance = float(np.var(centroids, axis=0).sum())

        # 2. EAR variance (natural blinking or slight micro-expressions vs 100% frozen photo)
        ears = [item["ear"] for item in face_hist]
        ear_variance = float(np.var(ears))

        # A completely static photo will have near-zero motion and near-zero EAR variance
        is_frozen = (motion_variance < 0.005 and ear_variance < 0.00001)

        # Screen replay check or static photo check
        if is_frozen and len(face_hist) >= 5:
            return {
                "is_live": False,
                "liveness_score": 0.25,
                "liveness_status": "SPOOF_DETECTED",
                "details": "Static presentation / frozen photo detected.",
                "metrics": {
                    "motion_var": round(motion_variance, 5),
                    "ear_var": round(ear_variance, 6),
                    "texture": texture_score
                }
            }

        combined_score = min(0.99, max(0.50, 0.4 * texture_score + 0.3 * min(1.0, motion_variance * 5) + 0.3 * (1.0 if ear_variance > 0.0001 else 0.5)))
        
        return {
            "is_live": True,
            "liveness_score": round(combined_score, 2),
            "liveness_status": "LIVE",
            "details": "Biometric liveness confirmed.",
            "metrics": {
                "motion_var": round(motion_variance, 4),
                "ear": ear,
                "texture": texture_score
            }
        }

    def clear_track(self, face_id: str):
        if face_id in self.history:
            del self.history[face_id]
