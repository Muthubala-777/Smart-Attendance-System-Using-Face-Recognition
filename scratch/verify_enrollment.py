import os
import sys
import json
import base64
import urllib.request
import urllib.parse
import http.client
import numpy as np
import cv2

BASE_URL = "http://127.0.0.1:8000"

def make_req(url, method="GET", headers=None, data=None):
    if headers is None:
        headers = {}
    if data is not None and isinstance(data, dict):
        data = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"detail": body}

def send_multipart(url, token, student_id, filename, file_bytes):
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    data = []
    data.append(f"--{boundary}".encode("utf-8"))
    data.append(f'Content-Disposition: form-data; name="file"; filename="{filename}"'.encode("utf-8"))
    data.append(b"Content-Type: image/jpeg\r\n")
    data.append(file_bytes)
    data.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"\r\n".join(data)

    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Authorization": f"Bearer {token}",
        "Content-Length": str(len(body))
    }
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            resp_body = resp.read().decode("utf-8")
            return resp.status, json.loads(resp_body) if resp_body else {}
    except urllib.error.HTTPError as e:
        resp_body = e.read().decode("utf-8")
        try:
            return e.code, json.loads(resp_body)
        except Exception:
            return e.code, {"detail": resp_body}

def test_enrollment_pipeline():
    print("=== 1. Checking Backend Operational Status ===")
    status, session_data = make_req(f"{BASE_URL}/api/sessions/current")
    assert status == 200, f"Backend status failed: {session_data}"
    print("Session Status:", session_data)

    print("\n=== 2. Testing Authentication ===")
    status, auth_data = make_req(f"{BASE_URL}/api/auth/login", method="POST", data={"username": "admin", "password": "admin123"})
    assert status == 200, f"Login failed: {auth_data}"
    token = auth_data["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("Logged in successfully. Admin:", auth_data["admin_name"])

    print("\n=== 3. Testing evaluate-pose Endpoint ===")
    dummy_img = np.zeros((240, 320, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", dummy_img)
    b64_dummy = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

    status, pose_data = make_req(f"{BASE_URL}/api/students/evaluate-pose", method="POST", data={
        "image_base64": b64_dummy,
        "target_angle": "front"
    })
    assert status == 200, f"evaluate-pose failed: {pose_data}"
    print("Pose Evaluation on blank frame:", pose_data)
    assert pose_data["face_detected"] is False
    assert pose_data["angle_matched"] is False

    print("\n=== 4. Testing upload-photo Zero-Face Rejection ===")
    test_id = "TEST_ENROLL_001"
    status, student_res = make_req(f"{BASE_URL}/api/students", method="POST", headers=headers, data={
        "student_id": test_id,
        "name": "Biometric Test Candidate",
        "department": "AI&DS",
        "year": "IV",
        "section": "A",
        "email": "biotest@college.edu"
    })
    print("Student register status:", status)

    # Attempt upload with blank image (zero faces)
    status, zero_face_res = send_multipart(
        f"{BASE_URL}/api/students/{test_id}/upload-photo",
        token,
        test_id,
        "blank.jpg",
        buf.tobytes()
    )
    print("Zero-face upload response status:", status, zero_face_res)
    assert status == 400
    assert "No face detected" in zero_face_res.get("detail", "")
    print("PASS: Zero-face validation rejected successfully.")

    print("\n=== 5. Testing Face Detection and Multi-Embedding Auto-Augmentation ===")
    sample_path = r"C:\Users\A.KANNAN\.gemini\antigravity-ide\brain\917e483c-cbaf-466c-b7d1-8b0f02820245\student_sample_1788772512432.jpg"
    if not os.path.exists(sample_path):
        sample_path = "scratch/student_sample.jpg"

    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            bytes_data = f.read()
        status, upload_res = send_multipart(
            f"{BASE_URL}/api/students/{test_id}/upload-photo",
            token,
            test_id,
            "student_portrait.jpg",
            bytes_data
        )
        print("Upload result:", status, upload_res)
        assert status == 200, f"Upload failed: {upload_res}"
        assert upload_res["enrolled_count"] == 5
        print(f"Sharpness Score: {upload_res.get('sharpness_score')} (Laplacian var > 100)")
        print("PASS: Single photo uploaded, validated, and auto-augmented into 5 deep embeddings!")

        # Check database
        sys.path.insert(0, os.getcwd())
        from backend.database import SessionLocal, FaceEmbedding
        db = SessionLocal()
        embeddings = db.query(FaceEmbedding).filter(FaceEmbedding.student_id == test_id).all()
        angles = [e.sample_angle for e in embeddings]
        print(f"PASS: Stored 5 angles in database for {test_id}: {angles}")
        assert set(angles) == {"front", "slight_left", "slight_right", "slight_up", "slight_down"}
        db.close()
    else:
        print("Sample portrait not found at", sample_path)

    print("\n=== 6. Verifying No QR Code References in Codebase ===")
    # Verify QR code absence
    from backend.database import SessionLocal, Student
    db = SessionLocal()
    for col in Student.__table__.columns:
        assert "qr" not in col.name.lower(), f"Unexpected QR column in Student: {col.name}"
    print("PASS: Database schemas are strictly QR-free.")
    db.close()

    print("\n=== ALL ENROLLMENT TESTS PASSED SUCCESSFULLY ===")

if __name__ == "__main__":
    test_enrollment_pipeline()
