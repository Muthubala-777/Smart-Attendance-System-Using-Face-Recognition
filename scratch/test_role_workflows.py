import sys
import os
sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from backend.main import app
from backend.database import get_db, Student, Faculty, Admin, Attendance, Timetable
import numpy as np
import cv2
import base64

client = TestClient(app)

def test_role_authentication_and_dashboards():
    print("\n--- 1. Testing Student Authentication ---")
    res = client.post("/api/auth/login", json={
        "username": "23AD015",
        "password": "student123",
        "role": "student"
    })
    assert res.status_code == 200, f"Student login failed: {res.text}"
    student_data = res.json()
    assert student_data["role"] == "student"
    assert student_data["user_id"] == "23AD015"
    student_token = student_data["access_token"]
    print(f"✓ Student authenticated: {student_data['admin_name']} ({student_data['user_id']})")

    print("\n--- 2. Testing Student Dashboard Endpoints ---")
    headers_student = {"Authorization": f"Bearer {student_token}"}
    
    # Profile
    p_res = client.get("/api/student/profile", headers=headers_student)
    assert p_res.status_code == 200, f"Student profile failed: {p_res.text}"
    p_data = p_res.json()
    assert p_data["student_id"] == "23AD015"
    print(f"✓ Student profile retrieved: {p_data['name']}, {p_data['department']} {p_data['year']}-{p_data['section']}, Attendance: {p_data['attendance_percentage']}%")
    
    # Photo endpoint
    photo_res = client.get("/api/students/23AD015/photo")
    assert photo_res.status_code == 200, f"Photo failed: {photo_res.status_code}"
    print(f"✓ Student photo served with content-type: {photo_res.headers.get('content-type')}")

    # Attendance ledger
    att_res = client.get("/api/student/attendance", headers=headers_student)
    assert att_res.status_code == 200, f"Student attendance history failed: {att_res.text}"
    att_list = att_res.json()
    print(f"✓ Student attendance history returned: {len(att_list)} records")

    print("\n--- 3. Testing Student Role Restrictions ---")
    # Student trying to access Staff session
    forbidden_staff = client.get("/api/staff/session", headers=headers_student)
    assert forbidden_staff.status_code == 403, f"Expected 403 for student accessing staff endpoint, got {forbidden_staff.status_code}"
    print(f"✓ Role Guard: Student correctly denied access to Staff endpoint (HTTP {forbidden_staff.status_code})")

    # Student trying to access Admin audit logs
    forbidden_admin = client.get("/api/audit-logs", headers=headers_student)
    assert forbidden_admin.status_code == 403, f"Expected 403 for student accessing admin endpoint, got {forbidden_admin.status_code}"
    print(f"✓ Role Guard: Student correctly denied access to Admin audit logs (HTTP {forbidden_admin.status_code})")

    print("\n--- 4. Testing Faculty / Staff Authentication ---")
    staff_res = client.post("/api/auth/login", json={
        "username": "FAC01",
        "password": "staff123",
        "role": "staff"
    })
    assert staff_res.status_code == 200, f"Staff login failed: {staff_res.text}"
    staff_data = staff_res.json()
    assert staff_data["role"] == "staff"
    assert staff_data["user_id"] == "FAC01"
    staff_token = staff_data["access_token"]
    print(f"✓ Faculty authenticated: {staff_data['admin_name']} ({staff_data['user_id']})")

    print("\n--- 5. Testing Staff Dashboard Endpoints ---")
    headers_staff = {"Authorization": f"Bearer {staff_token}"}
    s_res = client.get("/api/staff/session", headers=headers_staff)
    assert s_res.status_code == 200, f"Staff session failed: {s_res.text}"
    s_data = s_res.json()
    print(f"✓ Staff session retrieved: Active: {s_data['has_active_session']}, Subject: {s_data['subject_name']}, Total Students: {s_data['total_students']}")

    print("\n--- 6. Testing Admin Authentication & Backward Compatibility ---")
    admin_res = client.post("/api/auth/login", json={
        "username": "admin",
        "password": "admin123",
        "role": "admin"
    })
    assert admin_res.status_code == 200, f"Admin login failed: {admin_res.text}"
    admin_data = admin_res.json()
    assert admin_data["role"] == "admin"
    admin_token = admin_data["access_token"]
    headers_admin = {"Authorization": f"Bearer {admin_token}"}
    print(f"✓ Admin authenticated: {admin_data['admin_name']}")

    # Admin access to audit logs
    audit_res = client.get("/api/audit-logs", headers=headers_admin)
    assert audit_res.status_code == 200
    print(f"✓ Admin access verified: audit logs reachable ({len(audit_res.json())} entries)")

    print("\n--- 7. Testing Multi-Face Attendance Frame Processor ---")
    # Generate blank test frame (640x480)
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", dummy_img)
    base64_str = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

    frame_res = client.post("/api/attendance/process-frame", json={
        "image_base64": base64_str
    })
    assert frame_res.status_code == 200, f"Frame process failed: {frame_res.text}"
    frame_data = frame_res.json()
    assert "detected_faces" in frame_data
    assert "total_faces" in frame_data
    print(f"✓ Frame processor executed cleanly: total faces detected: {frame_data['total_faces']}")

    print("\n==========================================")
    print("ALL ROLE-BASED DASHBOARD WORKFLOWS PASSED!")
    print("==========================================")

if __name__ == "__main__":
    test_role_authentication_and_dashboards()
