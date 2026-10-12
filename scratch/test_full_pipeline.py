import cv2
import numpy as np
from pathlib import Path
from backend.database import SessionLocal, Student, FaceEmbedding
from backend.vision.face_detector import FaceDetector
from backend.vision.face_aligner import FaceAligner
from backend.vision.face_embedder import ArcFaceEmbedder
from backend.vision.matcher import FaceMatcher

db = SessionLocal()
matcher = FaceMatcher(default_threshold=0.60, k=3, max_euclidean_distance=0.96)
matcher.reload_from_db(db)
db.close()

det = FaceDetector(score_threshold=0.50)
aligner = FaceAligner()
embedder = ArcFaceEmbedder(embedding_dim=128)

samples = list(Path('data/face_samples').glob('*/*.jpg'))
tp, fn = 0, 0
det_count = 0
total_trials = 0

for p in samples:
    s_id = p.parent.name
    face = cv2.imread(str(p))
    variants = [
        face,
        cv2.convertScaleAbs(face, alpha=1.1, beta=10),
        cv2.convertScaleAbs(face, alpha=0.9, beta=-10),
        cv2.GaussianBlur(face, (3, 3), 0.5)
    ]
    for var in variants:
        total_trials += 1
        canvas = np.full((480, 640, 3), 195, dtype=np.uint8)
        canvas[140:140+112, 240:240+112] = var
        faces = det.detect_faces(canvas)
        if faces:
            det_count += 1
            al = aligner.align_face(canvas, faces[0].bbox, faces[0].landmarks)
        else:
            al = aligner.align_face(canvas, (240, 140, 112, 112), None)
        emb = embedder.extract_embedding(al)
        res = matcher.match_embedding(emb)
        if res['student_id'] == s_id and res['matched']:
            tp += 1
        else:
            fn += 1
            print(f"Mismatch for {s_id}: pred={res['student_id']}, dist={res['euclidean_distance']}, sim={res['similarity']}")

tn, fp = 0, 0
for i in range(60):
    total_trials += 1
    canvas = np.full((480, 640, 3), 195, dtype=np.uint8)
    imp = np.full((112, 112, 3), 130 + (i%20), dtype=np.uint8)
    cv2.circle(imp, (56, 56), 40, (i*13%255, 140, 200), -1)
    canvas[140:140+112, 240:240+112] = imp
    faces = det.detect_faces(canvas)
    if faces:
        det_count += 1
        al = aligner.align_face(canvas, faces[0].bbox, faces[0].landmarks)
    else:
        al = aligner.align_face(canvas, (240, 140, 112, 112), None)
    emb = embedder.extract_embedding(al)
    res = matcher.match_embedding(emb)
    if res['is_unknown'] or not res['matched']:
        tn += 1
    else:
        fp += 1

print(f"\n=======================================================")
print(f"Full Scene Detection Rate: {det_count}/{total_trials} = {det_count/total_trials*100:.2f}%")
print(f"Genuine Probes: TP={tp}, FN={fn}, Total={tp+fn}")
print(f"Impostor Probes: TN={tn}, FP={fp}, Total={tn+fp}")
acc = (tp + tn) / total_trials * 100
prec = tp / max(1, tp + fp) * 100
rec = tp / max(1, tp + fn) * 100
f1 = 2 * prec * rec / max(1, prec + rec)
far = fp / max(1, fp + tn) * 100
frr = fn / max(1, tp + fn) * 100
print(f"Accuracy: {acc:.2f}%")
print(f"Precision: {prec:.2f}%")
print(f"Recall: {rec:.2f}%")
print(f"F1-Score: {f1:.2f}%")
print(f"FAR: {far:.2f}%")
print(f"FRR: {frr:.2f}%")
print(f"=======================================================\n")
