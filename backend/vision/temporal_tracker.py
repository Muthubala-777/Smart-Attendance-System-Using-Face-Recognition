import time
from typing import Dict, Any, List, Optional, Tuple


class TemporalTracker:
    """
    Multi-Frame Temporal Verification & Identity Stabilization Engine.
    Ensures that a student is only confirmed as PRESENT if their facial embedding
    consistently matches above the recognition threshold across consecutive video frames.
    
    Prevents false attendance marks caused by transient sensor noise, motion blur,
    temporary occlusions, or camera angle shifts.
    """

    def __init__(
        self,
        consecutive_frames_required: int = 3,
        window_size: int = 5,
        track_timeout_seconds: float = 3.0
    ):
        self.consecutive_frames_required = consecutive_frames_required
        self.window_size = window_size
        self.track_timeout_seconds = track_timeout_seconds
        
        # spatial_track_key -> list of recognition event dicts
        self.tracks: Dict[str, List[Dict[str, Any]]] = {}
        self.last_seen_track: Dict[str, float] = {}

    def _get_spatial_key(self, bbox: Tuple[int, int, int, int]) -> str:
        """
        Quantizes face coordinates into spatial grid cells to correlate
        face detections across consecutive high-speed video frames without complex Hungarian matching.
        """
        fx, fy, fw, fh = bbox
        cx = fx + fw // 2
        cy = fy + fh // 2
        # Grid quantization to nearest 40 pixels
        grid_x = cx // 40
        grid_y = cy // 40
        return f"track_{grid_x}_{grid_y}"

    def update_and_verify(
        self,
        bbox: Tuple[int, int, int, int],
        recognition_result: Dict[str, Any],
        liveness_result: Dict[str, Any],
        current_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Updates temporal track for this spatial face region and evaluates consistency.
        Returns:
        - verified: bool (True if consecutive identity threshold met)
        - student_id: str
        - consecutive_count: int
        - is_liveness_passed: bool
        - status: str ("PRESENT", "VERIFYING", "INVALID")
        """
        now = current_time or time.time()
        track_key = self._get_spatial_key(bbox)
        
        # Cleanup expired tracks
        for k in list(self.last_seen_track.keys()):
            if (now - self.last_seen_track[k]) > self.track_timeout_seconds:
                self.tracks.pop(k, None)
                self.last_seen_track.pop(k, None)

        self.last_seen_track[track_key] = now

        if track_key not in self.tracks:
            self.tracks[track_key] = []

        history = self.tracks[track_key]
        
        student_id = recognition_result.get("student_id", "UNKNOWN")
        is_matched = recognition_result.get("matched", False)
        similarity = recognition_result.get("similarity", 0.0)
        is_live = liveness_result.get("is_live", False)

        history.append({
            "timestamp": now,
            "student_id": student_id,
            "matched": is_matched,
            "similarity": similarity,
            "is_live": is_live
        })

        if len(history) > self.window_size:
            history.pop(0)

        # Count consecutive matching occurrences of this specific student ID from the end
        consecutive_matches = 0
        for event in reversed(history):
            if event["matched"] and event["student_id"] == student_id:
                consecutive_matches += 1
            else:
                break

        # Verification decision
        if is_matched and consecutive_matches >= self.consecutive_frames_required and is_live:
            return {
                "verified": True,
                "student_id": student_id,
                "student_name": recognition_result.get("student_name", student_id),
                "consecutive_count": consecutive_matches,
                "required_frames": self.consecutive_frames_required,
                "similarity": similarity,
                "is_live": is_live,
                "status": "PRESENT",
                "message": f"Confirmed identity: {consecutive_matches}/{self.consecutive_frames_required} frames verified."
            }
        elif is_matched:
            return {
                "verified": False,
                "student_id": student_id,
                "student_name": recognition_result.get("student_name", student_id),
                "consecutive_count": consecutive_matches,
                "required_frames": self.consecutive_frames_required,
                "similarity": similarity,
                "is_live": is_live,
                "status": "VERIFYING",
                "message": f"Verifying identity ({consecutive_matches}/{self.consecutive_frames_required} frames)..."
            }
        else:
            return {
                "verified": False,
                "student_id": "UNKNOWN",
                "student_name": "Unknown / Invalid Student",
                "consecutive_count": 0,
                "required_frames": self.consecutive_frames_required,
                "similarity": similarity,
                "is_live": is_live,
                "status": "INVALID",
                "message": "Identity not registered or below similarity threshold."
            }

    def reset(self):
        self.tracks.clear()
        self.last_seen_track.clear()
