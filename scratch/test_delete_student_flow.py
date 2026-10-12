import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import urllib.request
import urllib.error
from backend.database import SessionLocal, Student, FaceEmbedding, Attendance, AuditLog
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

def test_delete_flow():
    print("=== 1. Login as Admin ===")
    status, login_res = request_json(f"{BASE_URL}/api/auth/login", method="POST", data={"username": "admin", "password": "admin123"})
    assert status == 200, f"Login failed: {login_res}"
    token = login_res["access_token"]
    print("Admin logged in successfully.")

    TARGET_ID = "23AD044"

    # Ensure target student exists for testing
    db = SessionLocal()
    st = db.query(Student).filter(Student.student_id == TARGET_ID).first()
    if not st:
        print(f"Creating student {TARGET_ID} to test deletion...")
        st = Student(
            student_id=TARGET_ID,
            name="M.Santhosh",
            department="AI&DS",
            year="IV",
            section="A",
            email="santhosh@example.com",
            phone="9876543210",
            is_active=True
        )
        db.add(st)
        db.commit()

        # Create sample folder and file
        folder = SAMPLES_DIR / TARGET_ID
        folder.mkdir(parents=True, exist_ok=True)
        sample_file = folder / f"{TARGET_ID}_front_test.jpg"
        sample_file.write_text("dummy_image_data")

        emb = FaceEmbedding(
            student_id=TARGET_ID,
            sample_angle="front",
            quality_score=0.95,
            sample_image_path=str(sample_file)
        )
        emb.set_vector([0.1] * 512)
        db.add(emb)
        db.commit()
    db.close()

    print("\n=== 2. Check current students in API ===")
    status, students = request_json(f"{BASE_URL}/api/students", token=token)
    assert status == 200
    student_ids = [s["student_id"] for s in students]
    print(f"Total students before deletion: {len(student_ids)}")
    print(f"Students: {student_ids}")
    assert TARGET_ID in student_ids, f"Student {TARGET_ID} should exist before test."

    # Record baseline counts of another student
    db = SessionLocal()
    other_student = db.query(Student).filter(Student.student_id == "23AD015").first()
    other_student_name = other_student.name
    other_emb_count = len(other_student.face_embeddings)
    other_att_count = len(other_student.attendance_records)
    db.close()

    print(f"\n=== 3. Execute DELETE for student {TARGET_ID} ===")
    del_status, del_json = request_json(f"{BASE_URL}/api/students/{TARGET_ID}", method="DELETE", token=token)
    print("DELETE Status Code:", del_status)
    print("DELETE Response JSON:", del_json)
    assert del_status == 200, f"Expected 200, got {del_status}"
    assert del_json.get("status") == "success"
    assert "deleted successfully" in del_json.get("message", "").lower()
    assert "deactivated" not in del_json.get("message", "").lower()

    print("\n=== 4. Verify SQLite Database Record is Permanently Removed ===")
    db = SessionLocal()
    st_in_db = db.query(Student).filter(Student.student_id == TARGET_ID).first()
    assert st_in_db is None, f"ERROR: Student {TARGET_ID} still exists in database!"
    
    emb_in_db = db.query(FaceEmbedding).filter(FaceEmbedding.student_id == TARGET_ID).all()
    assert len(emb_in_db) == 0, f"ERROR: {len(emb_in_db)} face embeddings for {TARGET_ID} still exist in database!"

    att_in_db = db.query(Attendance).filter(Attendance.student_id == TARGET_ID).all()
    assert len(att_in_db) == 0, f"ERROR: {len(att_in_db)} attendance records for {TARGET_ID} still exist in database!"

    # Verify audit log recorded DELETE_STUDENT
    audit = db.query(AuditLog).filter(AuditLog.target_id == TARGET_ID, AuditLog.action == "DELETE_STUDENT").order_by(AuditLog.id.desc()).first()
    assert audit is not None, "ERROR: Audit log for DELETE_STUDENT not created!"
    print("Audit log details:", audit.details)

    # Verify other students remain completely untouched
    other_st_after = db.query(Student).filter(Student.student_id == "23AD015").first()
    assert other_st_after is not None
    assert other_st_after.name == other_student_name
    assert len(other_st_after.face_embeddings) == other_emb_count
    assert len(other_st_after.attendance_records) == other_att_count
    print(f"Verified other student {other_st_after.student_id} ({other_st_after.name}) is unchanged with {other_emb_count} embeddings & {other_att_count} attendances.")

    db.close()

    print("\n=== 5. Verify Disk Face Samples Removed ===")
    sample_dir = SAMPLES_DIR / TARGET_ID
    assert not sample_dir.exists(), f"ERROR: Sample dir {sample_dir} still exists on disk!"
    print(f"Face sample directory {sample_dir} confirmed deleted.")

    print("\n=== 6. Verify Student Directory API List (Page reload simulation) ===")
    status_after, students_after = request_json(f"{BASE_URL}/api/students", token=token)
    assert status_after == 200
    student_ids_after = [s["student_id"] for s in students_after]
    print(f"Total students after deletion: {len(student_ids_after)}")
    assert TARGET_ID not in student_ids_after, f"ERROR: Student {TARGET_ID} still returned in student list!"

    print(f"\n=== 7. Verify DELETE on Non-Existent Student returns 404 ===")
    del_404_status, del_404_json = request_json(f"{BASE_URL}/api/students/{TARGET_ID}", method="DELETE", token=token)
    print("404 Test Status:", del_404_status, del_404_json)
    assert del_404_status == 404

    print("\n[SUCCESS] ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_delete_flow()
