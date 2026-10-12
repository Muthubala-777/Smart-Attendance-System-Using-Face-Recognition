import numpy as np
import cv2
from pathlib import Path
from backend.database import SessionLocal, Student
from backend.vision.face_detector import FaceDetector
from backend.vision.face_aligner import FaceAligner
from backend.vision.face_embedder import ArcFaceEmbedder

db = SessionLocal()
students = db.query(Student).filter(Student.is_active == True).all()

# Load gallery from database
gallery_vecs = []
gallery_ids = []
for s in students:
    # Only use students that have embeddings
    for emb in s.face_embeddings:
        v = np.array(emb.get_vector(), dtype=np.float32)
        v = v / np.linalg.norm(v)
        gallery_vecs.append(v)
        gallery_ids.append(s.student_id)

gallery_mat = np.array(gallery_vecs, dtype=np.float32)
db.close()

def match_knn(query_vec, k=3, threshold=0.88):
    q = query_vec / np.linalg.norm(query_vec)
    # Euclidean distances to all gallery samples
    dists = np.linalg.norm(gallery_mat - q, axis=1)
    
    # Top-k nearest indices
    top_k_idx = np.argsort(dists)[:k]
    
    votes = {}
    min_dists = {}
    for idx in top_k_idx:
        s_id = gallery_ids[idx]
        d = float(dists[idx])
        w = 1.0 / (d + 1e-5)
        votes[s_id] = votes.get(s_id, 0.0) + w
        if s_id not in min_dists or d < min_dists[s_id]:
            min_dists[s_id] = d
            
    best_id = max(votes, key=votes.get)
    best_dist = min_dists[best_id]
    
    # Cosine similarity equivalent
    sim = max(0.0, 1.0 - (best_dist**2) / 2.0)
    
    if best_dist <= threshold:
        return {"matched": True, "student_id": best_id, "distance": best_dist, "similarity": sim}
    else:
        return {"matched": False, "student_id": "UNKNOWN", "distance": best_dist, "similarity": sim}

# Test with samples
samples_dir = Path("data/face_samples")
aligner = FaceAligner()
embedder = ArcFaceEmbedder(embedding_dim=128)
det = FaceDetector()

correct = 0
total = 0
for s in ['23AD015', '23AD026', '23AD043', '23AD044', '25TAD01', 'TEST_ENROLL_001']:
    for p in (samples_dir / s).glob('*.jpg'):
        img = cv2.imread(str(p))
        faces = det.detect_faces(img)
        if faces:
            al = aligner.align_face(img, faces[0].bbox, faces[0].landmarks)
        else:
            al = aligner.align_face(img, (0,0,img.shape[1],img.shape[0]), None)
        emb = embedder.extract_embedding(al)
        res = match_knn(emb, k=3, threshold=0.88)
        is_ok = (res['student_id'] == s)
        if is_ok: correct += 1
        total += 1
        print(f"Probe {s} ({p.name}): predicted={res['student_id']}, dist={res['distance']:.4f}, sim={res['similarity']:.4f}, match={is_ok}")

print(f"\nKNN (k=3) Accuracy on real samples: {correct}/{total} = {correct/total*100:.2f}%")
