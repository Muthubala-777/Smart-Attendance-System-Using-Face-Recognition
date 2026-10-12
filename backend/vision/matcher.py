import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sqlalchemy.orm import Session
from backend.database import Student, FaceEmbedding


class FaceMatcher:
    """
    Conference Paper Reference: SUB-ID-781092
    "DEVELOPMENT OF A MULTI-FACET FACIAL RECOGNITION MODEL FOR INSTITUTIONAL ATTENDANCE ACTIVITIES"

    Architecture:
    Distance-weighted KNN (k=3) classification over 128-dimensional L2-normalized deep face embeddings
    with calibrated Euclidean distance threshold and unknown rejection.

    Mathematical Formulation:
    Given query embedding q and gallery samples e_i:
      d_i = ||q - e_i||_2  (Euclidean distance)
      Weight w_i = 1.0 / (d_i + 1e-5)
      Vote V(s) = sum_{i in top_k, ID_i = s} w_i
      Predicted candidate s* = argmax_s V(s)
      Rejection criterion: if min_{i in top_k, ID_i = s*} d_i > threshold -> UNKNOWN
      Cosine similarity: sim = max(0.0, 1.0 - d^2 / 2)
    """

    def __init__(
        self,
        default_threshold: float = 0.54,
        k: int = 3,
        max_euclidean_distance: float = 0.96
    ):
        self.default_threshold = default_threshold
        self.k = k
        self.max_euclidean_distance = max_euclidean_distance

        # In-memory index of enrolled student samples
        self.enrolled_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.flat_embeddings: Optional[np.ndarray] = None
        self.flat_student_ids: List[str] = []
        self.flat_sample_angles: List[str] = []
        self.student_info_map: Dict[str, Dict[str, Any]] = {}

    def reload_from_db(self, db: Session):
        """Loads and indexes enrolled student embeddings from SQLite."""
        students = db.query(Student).filter(Student.is_active == True).all()
        self.enrolled_cache.clear()
        self.student_info_map.clear()

        flat_list = []
        flat_ids = []
        flat_angles = []

        for student in students:
            self.student_info_map[student.student_id] = {
                "name": student.name,
                "department": student.department,
                "year": student.year,
                "section": student.section,
                "academic_year": student.academic_year,
                "semester": student.semester,
                "profile_photo_path": student.profile_photo_path
            }

            # Retrieve all embeddings for this student using exact & case-insensitive match
            embs = db.query(FaceEmbedding).filter(
                (FaceEmbedding.student_id == student.student_id) |
                (FaceEmbedding.student_id.ilike(student.student_id))
            ).all()

            samples = []
            for face_emb in embs:
                try:
                    vec_list = face_emb.get_vector()
                    if not vec_list or len(vec_list) == 0:
                        continue
                    vec = np.array(vec_list, dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 1e-6:
                        vec = vec / norm
                    samples.append({
                        "vector": vec,
                        "angle": face_emb.sample_angle,
                        "quality": face_emb.quality_score,
                        "image_path": face_emb.sample_image_path
                    })
                    flat_list.append(vec)
                    flat_ids.append(student.student_id)
                    flat_angles.append(face_emb.sample_angle or "front")
                except Exception as ex:
                    print(f"[FaceMatcher] Error parsing embedding for {student.student_id}: {ex}")

            if samples:
                self.enrolled_cache[student.student_id] = samples

        if flat_list:
            self.flat_embeddings = np.array(flat_list, dtype=np.float32)
            self.flat_student_ids = flat_ids
            self.flat_sample_angles = flat_angles
        else:
            self.flat_embeddings = None
            self.flat_student_ids = []
            self.flat_sample_angles = []

        print(f"[FaceMatcher] Indexed {len(self.enrolled_cache)} students ({len(self.flat_student_ids)} face samples) with KNN (k={self.k}).")

    def match_embedding(
        self,
        query_embedding: np.ndarray,
        threshold: Optional[float] = None,
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Distance-Weighted KNN (k=3) classification matching against enrolled gallery.
        Returns matched student information or UNKNOWN flag.
        """
        if query_embedding is None or self.flat_embeddings is None or len(self.flat_embeddings) == 0:
            return {
                "matched": False,
                "is_unknown": True,
                "student_id": "UNKNOWN",
                "student_name": "Unknown / Invalid Student",
                "similarity": 0.0,
                "euclidean_distance": 2.0,
                "status": "INVALID",
                "best_angle": "none"
            }

        q_vec = query_embedding.flatten().astype(np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 1e-6:
            q_vec = q_vec / q_norm

        # 1. Compute exact Euclidean distance to all gallery samples: d = ||q - e||_2
        distances = np.linalg.norm(self.flat_embeddings - q_vec, axis=1)

        # 2. Pick top-k nearest neighbors
        k_val = min(self.k if k is None else k, len(self.flat_embeddings))
        top_indices = np.argsort(distances)[:k_val]

        # 3. Distance-weighted voting
        votes: Dict[str, float] = {}
        min_distances: Dict[str, float] = {}
        best_angles: Dict[str, str] = {}

        for idx in top_indices:
            s_id = self.flat_student_ids[idx]
            d = float(distances[idx])
            weight = 1.0 / (d + 1e-5)
            votes[s_id] = votes.get(s_id, 0.0) + weight

            if s_id not in min_distances or d < min_distances[s_id]:
                min_distances[s_id] = d
                best_angles[s_id] = self.flat_sample_angles[idx]

        # Select candidate with highest aggregated vote weight
        best_student_id = max(votes, key=votes.get)
        best_distance = min_distances[best_student_id]

        # Compute cosine similarity equivalent: sim = 1.0 - (d^2) / 2.0
        best_similarity = max(0.0, float(1.0 - (best_distance ** 2) / 2.0))

        # 4. Unknown Rejection Threshold Calibration
        # Determine rejection boundary:
        # If explicit threshold passed:
        #   - if th > 1.0: treated as Euclidean distance threshold
        #   - if th <= 1.0: treated as similarity threshold (or fallback to max_euclidean_distance)
        is_accepted = False
        if threshold is not None:
            if threshold > 1.0:
                is_accepted = (best_distance <= threshold)
            else:
                is_accepted = (best_similarity >= threshold)
        else:
            is_accepted = (best_distance <= self.max_euclidean_distance and best_similarity >= self.default_threshold)

        if is_accepted and best_student_id:
            student_meta = self.student_info_map.get(best_student_id, {})
            return {
                "matched": True,
                "is_unknown": False,
                "student_id": best_student_id,
                "student_name": student_meta.get("name", best_student_id),
                "department": student_meta.get("department", ""),
                "year": student_meta.get("year", ""),
                "section": student_meta.get("section", ""),
                "similarity": round(best_similarity, 4),
                "euclidean_distance": round(best_distance, 4),
                "status": "PRESENT",
                "best_angle": best_angles.get(best_student_id, "front")
            }
        else:
            return {
                "matched": False,
                "is_unknown": True,
                "student_id": "UNKNOWN",
                "student_name": "Unknown / Invalid Student",
                "similarity": round(best_similarity, 4),
                "euclidean_distance": round(best_distance, 4),
                "status": "INVALID",
                "best_angle": best_angles.get(best_student_id, "none")
            }

    def match_multiple_embeddings(
        self,
        query_embeddings: List[np.ndarray],
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Batch matches all detected faces in a single video frame."""
        return [self.match_embedding(emb, threshold) for emb in query_embeddings]
