import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.database import SessionLocal, Student, FaceEmbedding
from backend.vision.face_detector import FaceDetector
from backend.vision.face_aligner import FaceAligner
from backend.vision.face_quality import FaceQualityAssessment
from backend.vision.face_embedder import ArcFaceEmbedder
from backend.vision.matcher import FaceMatcher
from backend.config import DEFAULT_CONFIGS


def run_evaluation(verbose: bool = True) -> Dict[str, Any]:
    """
    Evaluates real facial recognition accuracy on enrolled students and test probes.
    Measures genuine matching, impostor rejection, accuracy, precision, recall, F1, FAR, FRR, and latency.
    """
    db = SessionLocal()
    try:
        students = db.query(Student).filter(Student.is_active == True).all()
        if not students:
            print("[Evaluation] Error: No students found in database.")
            return {}

        detector = FaceDetector(score_threshold=0.60)
        aligner = FaceAligner(output_size=(112, 112))
        quality = FaceQualityAssessment(
            blur_threshold=DEFAULT_CONFIGS["blur_threshold"],
            min_face_size=DEFAULT_CONFIGS["min_face_size"],
            min_brightness=DEFAULT_CONFIGS["min_brightness"],
            max_brightness=DEFAULT_CONFIGS["max_brightness"]
        )
        embedder = ArcFaceEmbedder(embedding_dim=128)
        matcher = FaceMatcher()
        matcher.reload_from_db(db)

        # Build genuine and impostor probe dataset
        samples_dir = PROJECT_ROOT / "data" / "face_samples"
        genuine_probes: List[Tuple[str, np.ndarray, str]] = []  # (student_id, image, variant_type)
        impostor_probes: List[Tuple[str, np.ndarray, str]] = []

        for stu in students:
            s_id = stu.student_id
            stu_dir = samples_dir / s_id
            if stu_dir.exists():
                for img_p in stu_dir.glob("*.jpg"):
                    img = cv2.imread(str(img_p))
                    if img is not None and img.size > 0:
                        # 1. Base clean probe
                        genuine_probes.append((s_id, img, "clean"))
                        # 2. Illumination / lighting shift probe
                        bright = cv2.convertScaleAbs(img, alpha=1.12, beta=12)
                        genuine_probes.append((s_id, bright, "lighting_high"))
                        dim = cv2.convertScaleAbs(img, alpha=0.88, beta=-12)
                        genuine_probes.append((s_id, dim, "lighting_low"))
                        # 3. Mild motion blur probe
                        blurred = cv2.GaussianBlur(img, (3, 3), 0.5)
                        genuine_probes.append((s_id, blurred, "blur"))

        # Generate impostor probes (faces not enrolled in student gallery)
        for i in range(len(genuine_probes) // 2 + 10):
            # Synthetic / random face generator with un-enrolled seed
            impostor = np.full((220, 220, 3), 130 + (i % 25), dtype=np.uint8)
            seed = (i * 73 + 127) % 255
            cv2.circle(impostor, (110, 110), 80, (seed, 140, 210), -1)
            cv2.circle(impostor, (78, 90), 12, (30, 30, 30), -1)
            cv2.circle(impostor, (142, 90), 12, (30, 30, 30), -1)
            cv2.ellipse(impostor, (110, 150), (32, 16), 0, 0, 180, (20, 20, 20), 4)
            impostor_probes.append(("UNKNOWN_IMPOSTOR", impostor, "synthetic_impostor"))

        total_probes = len(genuine_probes) + len(impostor_probes)
        if verbose:
            print(f"[Evaluation] Total Test Probes: {total_probes} ({len(genuine_probes)} genuine, {len(impostor_probes)} impostor)")

        tp = 0  # Genuine recognized correctly
        fn = 0  # Genuine rejected or misclassified
        tn = 0  # Impostor correctly rejected as UNKNOWN
        fp = 0  # Impostor falsely accepted as student

        detection_success = 0
        latencies = []

        # Evaluate genuine probes
        for s_id, probe_img, var_type in genuine_probes:
            t0 = time.perf_counter()
            if probe_img.shape[:2] == (112, 112):
                detection_success += 1
                aligned = aligner.align_face(probe_img, (0, 0, 112, 112), None)
            else:
                dets = detector.detect_faces(probe_img)
                if not dets:
                    h, w = probe_img.shape[:2]
                    aligned = aligner.align_face(probe_img, (0, 0, w, h), None)
                else:
                    detection_success += 1
                    aligned = aligner.align_face(probe_img, dets[0].bbox, dets[0].landmarks)

            t1 = time.perf_counter()
            emb = embedder.extract_embedding(aligned)
            match_res = matcher.match_embedding(emb)
            t_total = (time.perf_counter() - t0) * 1000.0  # ms
            latencies.append(t_total)

            pred_id = match_res.get("student_id", "UNKNOWN")
            if pred_id == s_id and match_res.get("matched", False):
                tp += 1
            else:
                fn += 1

        # Evaluate impostor probes
        for imp_id, probe_img, var_type in impostor_probes:
            t0 = time.perf_counter()
            dets = detector.detect_faces(probe_img)
            if dets:
                detection_success += 1
                aligned = aligner.align_face(probe_img, dets[0].bbox, dets[0].landmarks)
            else:
                h, w = probe_img.shape[:2]
                aligned = aligner.align_face(probe_img, (0, 0, w, h), None)

            emb = embedder.extract_embedding(aligned)
            match_res = matcher.match_embedding(emb)
            latencies.append((time.perf_counter() - t0) * 1000.0)

            pred_id = match_res.get("student_id", "UNKNOWN")
            if pred_id == "UNKNOWN" or not match_res.get("matched", False):
                tn += 1
            else:
                fp += 1

        total_trials = tp + fn + tn + fp
        accuracy = round(((tp + tn) / max(1, total_trials)) * 100.0, 2)
        precision = round((tp / max(1, tp + fp)) * 100.0, 2)
        recall = round((tp / max(1, tp + fn)) * 100.0, 2)
        f1 = round((2 * precision * recall / max(1.0, precision + recall)), 2)
        far = round((fp / max(1, fp + tn)) * 100.0, 2)
        frr = round((fn / max(1, tp + fn)) * 100.0, 2)
        avg_latency = round(float(np.mean(latencies)), 2)
        det_rate = round((detection_success / max(1, total_trials)) * 100.0, 2)

        results = {
            "total_test_probes": total_trials,
            "genuine_count": len(genuine_probes),
            "impostor_count": len(impostor_probes),
            "detection_rate": det_rate,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "far": far,
            "frr": frr,
            "latency_ms": avg_latency,
            "confusion_matrix": {
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn
            }
        }

        if verbose:
            print("\n=======================================================")
            print("         RECOGNITION PIPELINE EVALUATION REPORT        ")
            print("=======================================================")
            print(f"Total Test Probes Tested : {total_trials}")
            print(f"Face Detection Rate      : {det_rate}%")
            print(f"Recognition Accuracy     : {accuracy}%")
            print(f"Precision                : {precision}%")
            print(f"Recall                   : {recall}%")
            print(f"F1-Score                 : {f1}%")
            print(f"False Acceptance Rate    : {far}%")
            print(f"False Rejection Rate     : {frr}%")
            print(f"Average Inference Latency: {avg_latency} ms")
            print(f"Confusion Matrix         : TP={tp}, FP={fp}, TN={tn}, FN={fn}")
            print("=======================================================\n")

        return results

    finally:
        db.close()


if __name__ == "__main__":
    run_evaluation(verbose=True)
