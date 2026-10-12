"""
Integration test for Student Details, Edit Student, Staff Attendance History, and Manual Corrections.
"""
import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from backend.main import app

def test_student_and_history_endpoints():
    client = TestClient(app)

    # 1. Staff Login
    res = client.post("/api/auth/login", json={
        "username": "FAC01",
        "password": "staff123",
        "role": "staff"
    })
    assert res.status_code == 200, res.text
    staff_token = res.json()["access_token"]
    staff_headers = {"Authorization": f"Bearer {staff_token}"}
    print("[OK] Staff authenticated successfully")

    # 2. Test GET /api/students/{student_id}
    res = client.get("/api/students/23AD015", headers=staff_headers)
    assert res.status_code == 200, res.text
    student_data = res.json()
    assert student_data["student_id"] == "23AD015"
    assert "name" in student_data
    assert "face_sample_count" in student_data
    print(f"[OK] GET /api/students/23AD015 succeeded: {student_data['name']} ({student_data['face_sample_count']} samples)")

    # 3. Test PUT /api/students/{student_id}
    orig_email = student_data.get("email")
    new_email = "kannan.updated@college.edu"
    res = client.put("/api/students/23AD015", json={"email": new_email}, headers=staff_headers)
    assert res.status_code == 200, res.text
    assert res.json()["email"] == new_email
    print(f"[OK] PUT /api/students/23AD015 succeeded: email updated to {new_email}")

    # Revert back
    client.put("/api/students/23AD015", json={"email": orig_email}, headers=staff_headers)

    # 4. Test GET /api/staff/attendance-history
    res = client.get("/api/staff/attendance-history", headers=staff_headers)
    assert res.status_code == 200, res.text
    history_items = res.json()
    assert isinstance(history_items, list)
    print(f"[OK] GET /api/staff/attendance-history returned {len(history_items)} records")

    if len(history_items) > 0:
        first_record = history_items[0]
        rec_id = first_record["id"]
        orig_status = first_record["status"]

        # 5. Test PUT /api/staff/attendance/{attendance_id}
        # First test validation: reason required
        res_fail = client.put(f"/api/staff/attendance/{rec_id}", json={
            "attendance_id": rec_id,
            "new_status": "PRESENT",
            "reason": ""
        }, headers=staff_headers)
        assert res_fail.status_code == 400
        print("[OK] Validation passed: empty correction reason correctly rejected (HTTP 400)")

        # Valid modification
        target_status = "LATE" if orig_status == "PRESENT" else "PRESENT"
        res_edit = client.put(f"/api/staff/attendance/{rec_id}", json={
            "attendance_id": rec_id,
            "new_status": target_status,
            "new_time": "09:12:00 AM",
            "reason": "Official lab duty approved by HOD"
        }, headers=staff_headers)
        assert res_edit.status_code == 200, res_edit.text
        edit_result = res_edit.json()
        assert edit_result["new_status"] == target_status
        assert "modified_by" in edit_result
        print(f"[OK] Attendance correction succeeded: Record #{rec_id} changed to {target_status} with reason logged")

        # Verify in history query
        res_verify = client.get("/api/staff/attendance-history", headers=staff_headers)
        updated_records = {r["id"]: r for r in res_verify.json()}
        assert updated_records[rec_id]["status"] == target_status
        assert updated_records[rec_id]["attendance_source"] == "STAFF_CORRECTION"
        assert updated_records[rec_id]["correction_reason"] == "Official lab duty approved by HOD"
        print(f"[OK] History record verified: source='STAFF_CORRECTION', reason='{updated_records[rec_id]['correction_reason']}'")

    print("\n=======================================================")
    print("ALL STUDENT EDIT & ATTENDANCE AUDIT TESTS PASSED!")
    print("=======================================================")

if __name__ == "__main__":
    test_student_and_history_endpoints()
