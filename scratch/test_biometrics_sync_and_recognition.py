import sys
import os
import json
import base64
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.database import SessionLocal, Student, FaceEmbedding
from backend.config import SAMPLES_DIR

BASE_URL = "http://127.0.0.1:8000"

def request_json(url, method="GET", data=None, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as response:
            res_body = response.read().decode("utf-8")
            return response.status, json.loads(res_body) if res_body else {}
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        try:
            return e.code, json.loads(err_body)
        except Exception:
            return e.code, {"detail": err_body}

def run_tests():
    print("=== Authenticating as Admin ===")
    status, login_res = request_json(f"{BASE_URL}/api/auth/login", method="POST", data={"username": "admin", "password": "admin123"})
    assert status == 200, f"Login failed: {login_res}"
    token = login_res["access_token"]
    print("Admin logged in successfully.")

    print("\n================================================================")
    print("  TEST 1: Verify Student Directory Sample Counts")
    print("================================================================")
    status, students = request_json(f"{BASE_URL}/api/students", token=token)
    assert status == 200, f"Failed to get students: {status} - {students}"
    
    print(f"Total students retrieved: {len(students)}")
    for s in students:
        sid = s["student_id"]
        name = s["name"]
        cnt = s["face_sample_count"]
        samples_len = len(s["face_samples"])
        print(f"-> {sid:15} | {name:25} | {cnt} Samples (sample list len: {samples_len})")

    # Verify specific enrolled students do not have 0 samples
    kannan = next((s for s in students if s["student_id"] == "23AD015"), None)
    assert kannan is not None, "Student 23AD015 not found"
    print(f"\nStudent 23AD015 ({kannan['name']}) sample count: {kannan['face_sample_count']}")
    assert kannan["face_sample_count"] >= 3, f"Expected at least 3 samples for 23AD015, got {kannan['face_sample_count']}"

    student_24 = next((s for s in students if s["student_id"] == "23AD024"), None)
    if student_24:
        assert student_24["face_sample_count"] == 5, f"Expected 5 samples for 23AD024, got {student_24['face_sample_count']}"

    student_26 = next((s for s in students if s["student_id"] == "23AD026"), None)
    if student_26:
        assert student_26["face_sample_count"] == 5, f"Expected 5 samples for 23AD026, got {student_26['face_sample_count']}"

    test_enroll = next((s for s in students if s["student_id"] == "TEST_ENROLL_001"), None)
    if test_enroll:
        assert test_enroll["face_sample_count"] == 5, f"Expected 5 samples for TEST_ENROLL_001, got {test_enroll['face_sample_count']}"

    print("\n[PASSED] Student Directory correctly displays actual enrolled sample counts (NOT 0 Samples).")

    print("\n================================================================")
    print("  TEST 2: Live Attendance Recognition with Enrolled Face")
    print("================================================================")
    # Load genuine enrolled image of 23AD015
    img_path = SAMPLES_DIR / "23AD015" / "23AD015_front_1788769780.jpg"
    assert img_path.exists(), f"Sample image {img_path} not found"
    
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    process_payload = {
        "image_base64": f"data:image/jpeg;base64,{img_b64}",
        "session_id": None
    }
    status, frame_res = request_json(f"{BASE_URL}/api/attendance/process-frame", method="POST", data=process_payload)
    assert status == 200, f"Frame process failed: {status} - {frame_res}"
    
    print(f"Total faces detected in frame: {frame_res.get('total_faces', 0)}")
    faces = frame_res.get("detected_faces", [])
    assert len(faces) > 0, "No faces detected in frame"
    
    face0 = faces[0]
    print(f"Detected Student ID: {face0.get('student_id')}")
    print(f"Detected Student Name: {face0.get('student_name')}")
    print(f"Match Status: {face0.get('status')}")
    print(f"Similarity Confidence: {face0.get('confidence')}")
    print(f"Matched Flag: {face0.get('matched')}")

    assert face0.get("student_id") == "23AD015", f"Expected student 23AD015, got {face0.get('student_id')}"
    assert face0.get("student_name") == "A.Kannan", f"Expected name A.Kannan, got {face0.get('student_name')}"
    assert face0.get("matched") is True, "Expected matched to be True"
    assert face0.get("student_name") != "Unknown Student", "Should NOT be Unknown Student!"
    print("\n[PASSED] Enrolled student accurately recognized as 'A.Kannan' (NOT Unknown Student).")

    print("\n================================================================")
    print("  TEST 3: Live Attendance with Non-Enrolled Unknown Face")
    print("================================================================")
    # Generate synthetic/blank image of an unknown face or non-enrolled person
    import cv2
    import numpy as np
    unknown_img = np.full((300, 300, 3), 180, dtype=np.uint8)
    cv2.circle(unknown_img, (150, 150), 70, (210, 180, 150), -1)
    cv2.circle(unknown_img, (120, 130), 10, (50, 50, 50), -1)
    cv2.circle(unknown_img, (180, 130), 10, (50, 50, 50), -1)
    cv2.ellipse(unknown_img, (150, 185), (25, 10), 0, 0, 180, (40, 40, 40), 3)

    _, enc = cv2.imencode(".jpg", unknown_img)
    unk_b64 = base64.b64encode(enc).decode("utf-8")

    unk_payload = {
        "image_base64": f"data:image/jpeg;base64,{unk_b64}",
        "session_id": None
    }
    status, unk_res = request_json(f"{BASE_URL}/api/attendance/process-frame", method="POST", data=unk_payload)
    print(f"Non-enrolled face response status: {status}")
    if unk_res.get("detected_faces"):
        det_unk = unk_res["detected_faces"][0]
        print(f"Non-enrolled match result: student_id={det_unk.get('student_id')}, name={det_unk.get('student_name')}, matched={det_unk.get('matched')}")
        assert det_unk.get("matched") is False or det_unk.get("student_id") in ["UNKNOWN", "POOR_QUALITY"], "Unknown face should not match enrolled students"

    print("\n[PASSED] Non-enrolled face is correctly rejected / designated as Unknown Student.")

    print("\n================================================================")
    print("  ALL BIOMETRIC SYNCHRONIZATION AND RECOGNITION TESTS PASSED!")
    print("================================================================")

if __name__ == "__main__":
    run_tests()
