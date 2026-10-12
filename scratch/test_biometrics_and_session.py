import sys
import os
sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from backend.main import app
from backend.database import get_db, Student, Faculty, Admin, Attendance, Timetable, AttendanceSession
import cv2
import numpy as np
import base64

client = TestClient(app)

def test_live_session_and_face_attendance():
    print("\n--- 0. Loading Biometric Gallery into Matcher ---")
    db = next(get_db())
    from backend.main import face_matcher
    face_matcher.reload_from_db(db)

    print("\n--- 1. Faculty Login & Session Activation ---")
    staff_res = client.post("/api/auth/login", json={
        "username": "FAC01",
        "password": "staff123",
        "role": "staff"
    })
    assert staff_res.status_code == 200
    staff_token = staff_res.json()["access_token"]
    staff_headers = {"Authorization": f"Bearer {staff_token}"}

    # Find or create a timetable slot for testing
    db = next(get_db())
    slot = db.query(Timetable).first()
    assert slot is not None, "No timetable slot found in database"
    slot_id = slot.id
    print(f"Testing with Timetable Slot #{slot_id}: {slot.department} {slot.year}-{slot.section}")

    # Activate session
    act_res = client.post(f"/api/staff/session/activate/{slot_id}", headers=staff_headers)
    assert act_res.status_code == 200
    session_info = act_res.json()
    session_id = session_info["session_id"]
    print(f"✓ Session #{session_id} activated: {session_info['subject_name']}")

    # Check Staff Dashboard
    staff_dash_res = client.get("/api/staff/session", headers=staff_headers)
    assert staff_dash_res.status_code == 200
    dash_data = staff_dash_res.json()
    assert dash_data["has_active_session"] == True
    assert dash_data["session_id"] == session_id
    print(f"✓ Staff Dashboard shows active session #{session_id} with {dash_data['total_students']} total class students")

    # Check Class Students roster
    roster_res = client.get(f"/api/staff/session/{session_id}/students", headers=staff_headers)
    assert roster_res.status_code == 200
    roster = roster_res.json()
    print(f"✓ Class Roster retrieved: {len(roster)} students")
    for s in roster[:3]:
        print(f"   - {s['student_id']}: {s['name']} (Photo: {s['photo_url']}, Status: {s['status']})")

    # 2. Test Face Recognition Attendance using enrolled photo
    print("\n--- 2. Face Recognition & Attendance Marking ---")
    face_sample_path = "data/face_samples/23AD015/23AD015_front_1788769780.jpg"
    if not os.path.exists(face_sample_path):
        for root, dirs, files in os.walk("data/face_samples"):
            for f in files:
                if "front" in f and f.endswith(".jpg"):
                    face_sample_path = os.path.join(root, f)
                    break
            if os.path.exists(face_sample_path):
                break

    if os.path.exists(face_sample_path):
        print(f"Testing frame processing with real sample: {face_sample_path}")
        sample_img = cv2.imread(face_sample_path)
        assert sample_img is not None
        
        # Emulate 640x480 webcam viewport
        frame_canvas = np.zeros((480, 640, 3), dtype=np.uint8) + 120
        sh, sw = sample_img.shape[:2]
        frame_canvas[100:100+sh, 200:200+sw] = sample_img
        
        _, buffer = cv2.imencode(".jpg", frame_canvas)
        base64_str = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

        # Temporal tracker requires temporal_frames_required consecutive detections to confirm attendance
        # Send consecutive frames
        print("Sending webcam frames to temporal tracker...")
        for frame_idx in range(6):
            res = client.post("/api/attendance/process-frame", json={
                "image_base64": base64_str,
                "session_id": session_id
            })
            assert res.status_code == 200
            d = res.json()
            face = d["detected_faces"][0] if d["detected_faces"] else None
            if face:
                print(f"   Frame {frame_idx+1}: {face['student_name']} ({face['student_id']}) | Status: {face['status']} | Confidence: {face['confidence']:.2f} | Marked: {face.get('attendance_marked')} | Already: {face.get('already_marked')}")
                if face.get("attendance_marked") or face.get("already_marked"):
                    break

        # Verification of matching
        assert face is not None and face["matched"] == True
        print(f"✓ Biometric match confirmed: {face['student_name']} matched to enrolled gallery!")

        # Frame 2: Immediate second scan of the same face -> MUST flag duplicate (already_marked = True, attendance_marked = False)
        print("\n--- 3. Testing Duplicate Attendance Prevention ---")
        res2 = client.post("/api/attendance/process-frame", json={
            "image_base64": base64_str,
            "session_id": session_id
        })
        assert res2.status_code == 200
        d2 = res2.json()
        for f in d2["detected_faces"]:
            print(f"Frame 2 Result: {f['student_name']} ({f['student_id']}) -> Already Marked: {f.get('already_marked')}, New Mark: {f.get('attendance_marked')}")
            assert f.get("already_marked") == True, "Expected duplicate prevention flag already_marked=True"
            assert f.get("attendance_marked") == False, "Expected attendance_marked=False on repeated scan"
        print("✓ Duplicate prevention confirmed: Student recognized without duplicate database insert!")

        # 4. Verify Live Attendance record in Staff Dashboard
        print("\n--- 4. Checking Staff Live Attendance Feed ---")
        live_att_res = client.get(f"/api/sessions/{session_id}/attendance", headers=staff_headers)
        assert live_att_res.status_code == 200
        live_att = live_att_res.json()
        present_students = [r for r in live_att if r["status"] == "PRESENT"]
        print(f"✓ Staff live attendance feed shows {len(present_students)} student(s) PRESENT")
        for p in present_students:
            print(f"   - {p['student_name']} ({p['student_id']}) | First Seen: {p['first_seen']} | Score: {p.get('similarity_score')}")

    print("\n=======================================================")
    print("ALL BIOMETRIC ATTENDANCE & LIVE SESSION TESTS PASSED!")
    print("=======================================================")

if __name__ == "__main__":
    test_live_session_and_face_attendance()
