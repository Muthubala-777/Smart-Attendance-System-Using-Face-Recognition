from typing import Tuple, Dict, Any, List, Optional
import cv2
import numpy as np


class FaceQualityAssessment:
    """
    Comprehensive Face Quality Assessment (FQA) Engine.
    Filters out blurry, poorly lit, undersized, or occluded face captures
    to prevent low-quality enrollment samples and false attendance triggers.
    """

    def __init__(
        self,
        blur_threshold: float = 65.0,
        min_face_size: int = 70,
        min_brightness: float = 35.0,
        max_brightness: float = 235.0,
        min_confidence: float = 0.65
    ):
        self.blur_threshold = blur_threshold
        self.min_face_size = min_face_size
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_confidence = min_confidence

    def assess_crop_quality(
        self,
        aligned_face: np.ndarray,
        bbox: Tuple[int, int, int, int],
        confidence: float,
        frame_shape: Tuple[int, int],
        total_faces_in_frame: int = 1,
        is_enrollment: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluates the biometric quality of a face crop and returns:
        - is_valid: bool
        - quality_score: float (0.0 to 1.0)
        - message: str
        - metrics: dict (blur, brightness, contrast, size, confidence)
        """
        h_frame, w_frame = frame_shape
        fx, fy, fw, fh = bbox

        # 1. Multi-face enrollment check
        if is_enrollment and total_faces_in_frame > 1:
            return {
                "is_valid": False,
                "quality_score": 0.0,
                "message": f"Multiple faces ({total_faces_in_frame}) detected! Only 1 face allowed during student enrollment.",
                "reason": "multiple_faces_detected",
                "metrics": {"total_faces": total_faces_in_frame}
            }

        if total_faces_in_frame == 0:
            return {
                "is_valid": False,
                "quality_score": 0.0,
                "message": "No face detected in camera view.",
                "reason": "no_face",
                "metrics": {}
            }

        # 2. Minimum face size check
        min_sz = min(self.min_face_size, int(0.4 * min(w_frame, h_frame)))
        if fw < min_sz or fh < min_sz:
            return {
                "is_valid": False,
                "quality_score": 0.2,
                "message": f"Face is too far / small ({fw}x{fh}px). Please move closer to the camera.",
                "reason": "face_too_small",
                "metrics": {"width": fw, "height": fh}
            }

        # 3. Frame boundary clipping check (applied for standard camera scenes >= 200x200)
        if w_frame >= 200 and h_frame >= 200:
            margin = 2
            is_clipped = (fx <= margin or fy <= margin or (fx + fw) >= (w_frame - margin) or (fy + fh) >= (h_frame - margin))
            if is_clipped:
                return {
                    "is_valid": False,
                    "quality_score": 0.3,
                    "message": "Face is partially outside the camera frame. Center your face.",
                    "reason": "face_clipped",
                    "metrics": {}
                }

        # 4. Detection Confidence Check
        if confidence < self.min_confidence:
            return {
                "is_valid": False,
                "quality_score": round(confidence, 2),
                "message": "Face detection confidence is low. Improve lighting or face camera directly.",
                "reason": "low_confidence",
                "metrics": {"confidence": confidence}
            }

        if aligned_face is None or aligned_face.size == 0:
            return {
                "is_valid": False,
                "quality_score": 0.0,
                "message": "Invalid face crop.",
                "reason": "empty_crop",
                "metrics": {}
            }

        # Grayscale representation for metrics
        gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY) if len(aligned_face.shape) == 3 else aligned_face

        # 5. Blur / Sharpness check using Laplacian Variance
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if laplacian_var < self.blur_threshold:
            return {
                "is_valid": False,
                "quality_score": min(0.5, round(laplacian_var / (self.blur_threshold * 1.5), 2)),
                "message": f"Face is too blurry (Sharpness: {laplacian_var:.1f}). Please stay still.",
                "reason": "blurred_image",
                "metrics": {"sharpness": round(laplacian_var, 1)}
            }

        # 6. Brightness and Contrast Check
        mean_brightness = float(np.mean(gray))
        contrast_std = float(np.std(gray))

        if mean_brightness < self.min_brightness:
            return {
                "is_valid": False,
                "quality_score": 0.4,
                "message": f"Image is too dark (Luminance: {mean_brightness:.1f}). Improve lighting.",
                "reason": "too_dark",
                "metrics": {"brightness": round(mean_brightness, 1)}
            }

        if mean_brightness > self.max_brightness:
            return {
                "is_valid": False,
                "quality_score": 0.4,
                "message": f"Image is overexposed (Luminance: {mean_brightness:.1f}). Reduce direct glare.",
                "reason": "overexposed",
                "metrics": {"brightness": round(mean_brightness, 1)}
            }

        if contrast_std < 18.0:
            return {
                "is_valid": False,
                "quality_score": 0.45,
                "message": "Image contrast is too washed out.",
                "reason": "low_contrast",
                "metrics": {"contrast": round(contrast_std, 1)}
            }

        # Compute continuous quality score [0.70 - 0.99]
        sharpness_factor = min(1.0, laplacian_var / 250.0)
        brightness_factor = 1.0 - abs(mean_brightness - 128.0) / 128.0
        contrast_factor = min(1.0, contrast_std / 60.0)
        conf_factor = min(1.0, confidence)

        overall_quality = 0.35 * sharpness_factor + 0.25 * brightness_factor + 0.20 * contrast_factor + 0.20 * conf_factor
        overall_quality = round(max(0.65, min(0.99, float(overall_quality))), 2)

        return {
            "is_valid": True,
            "quality_score": overall_quality,
            "message": f"High quality face sample verified (Score: {overall_quality * 100:.0f}%)",
            "reason": "passed",
            "metrics": {
                "sharpness": round(laplacian_var, 1),
                "brightness": round(mean_brightness, 1),
                "contrast": round(contrast_std, 1),
                "confidence": round(confidence, 2),
                "size": f"{fw}x{fh}"
            }
        }
