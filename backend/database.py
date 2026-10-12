import json
from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Text,
    DateTime, ForeignKey, Boolean, UniqueConstraint, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from backend.config import DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Admin(Base):
    __tablename__ = "admins"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    password_hash = Column(String(256), nullable=False)
    name = Column(String(128), nullable=False)
    email = Column(String(128), unique=True, nullable=False)
    role = Column(String(32), default="admin")  # "superadmin", "admin"
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    audit_logs = relationship("AuditLog", back_populates="admin")


class Department(Base):
    __tablename__ = "departments"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(16), unique=True, nullable=False, index=True)  # e.g., "AI&DS", "CSE", "ECE"
    name = Column(String(128), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    subjects = relationship("Subject", back_populates="department_rel", cascade="all, delete-orphan")
    faculty_members = relationship("Faculty", back_populates="department_rel", cascade="all, delete-orphan")


class Subject(Base):
    __tablename__ = "subjects"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(32), unique=True, nullable=False, index=True)  # e.g., "CS8501"
    name = Column(String(128), nullable=False)                          # e.g., "Machine Learning"
    department_id = Column(Integer, ForeignKey("departments.id"), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    department_rel = relationship("Department", back_populates="subjects")
    timetable_slots = relationship("Timetable", back_populates="subject_rel")
    sessions = relationship("AttendanceSession", back_populates="subject_rel")


class Faculty(Base):
    __tablename__ = "faculty"

    id = Column(Integer, primary_key=True, index=True)
    faculty_id = Column(String(32), unique=True, nullable=False, index=True)
    name = Column(String(128), nullable=False)
    email = Column(String(128), nullable=False)
    phone = Column(String(32), nullable=True)
    department_id = Column(Integer, ForeignKey("departments.id"), nullable=False)
    password_hash = Column(String(256), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    department_rel = relationship("Department", back_populates="faculty_members")
    timetable_slots = relationship("Timetable", back_populates="faculty_rel")


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String(64), unique=True, nullable=False, index=True)  # Roll number / Reg ID
    name = Column(String(128), nullable=False, index=True)
    department = Column(String(64), nullable=False, index=True)               # e.g., "AI&DS"
    year = Column(String(16), nullable=False, index=True)                     # e.g., "IV", "III"
    section = Column(String(16), nullable=False, index=True)                  # e.g., "A", "B"
    email = Column(String(128), nullable=True)
    phone = Column(String(32), nullable=True)
    academic_year = Column(String(32), nullable=False, default="2025-2026")
    semester = Column(String(16), nullable=False, default="VII")
    profile_photo_path = Column(String(256), nullable=True)
    password_hash = Column(String(256), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    face_embeddings = relationship("FaceEmbedding", back_populates="student_rel", cascade="all, delete-orphan")
    attendance_records = relationship("Attendance", back_populates="student_rel", cascade="all, delete-orphan")


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String(64), ForeignKey("students.student_id"), nullable=False, index=True)
    # Stored as JSON string or float list
    embedding_json = Column(Text, nullable=False)
    sample_angle = Column(String(32), nullable=False, default="front")  # front, left, right, up, down
    quality_score = Column(Float, default=1.0)
    sample_image_path = Column(String(256), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    student_rel = relationship("Student", back_populates="face_embeddings")

    def get_vector(self) -> List[float]:
        return json.loads(self.embedding_json)

    def set_vector(self, vector: List[float]):
        self.embedding_json = json.dumps([round(float(v), 6) for v in vector])


class Timetable(Base):
    __tablename__ = "timetable"

    id = Column(Integer, primary_key=True, index=True)
    department = Column(String(64), nullable=False, index=True)
    year = Column(String(16), nullable=False, index=True)
    section = Column(String(16), nullable=False, index=True)
    day_of_week = Column(String(16), nullable=False, index=True)  # Monday, Tuesday, ...
    start_time = Column(String(8), nullable=False)                # "10:00"
    end_time = Column(String(8), nullable=False)                  # "11:00"
    subject_id = Column(Integer, ForeignKey("subjects.id"), nullable=False)
    faculty_id = Column(Integer, ForeignKey("faculty.id"), nullable=True)
    classroom = Column(String(64), nullable=False, default="Lab-1")
    academic_year = Column(String(32), nullable=False, default="2025-2026")
    semester = Column(String(16), nullable=False, default="VII")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    subject_rel = relationship("Subject", back_populates="timetable_slots")
    faculty_rel = relationship("Faculty", back_populates="timetable_slots")
    sessions = relationship("AttendanceSession", back_populates="timetable_rel")


class AttendanceSession(Base):
    __tablename__ = "attendance_sessions"

    id = Column(Integer, primary_key=True, index=True)
    timetable_id = Column(Integer, ForeignKey("timetable.id"), nullable=True)
    subject_id = Column(Integer, ForeignKey("subjects.id"), nullable=False)
    department = Column(String(64), nullable=False)
    year = Column(String(16), nullable=False)
    section = Column(String(16), nullable=False)
    session_date = Column(String(16), nullable=False, index=True)  # "YYYY-MM-DD"
    start_time = Column(String(8), nullable=False)                # "10:00"
    end_time = Column(String(8), nullable=False)                  # "11:00"
    status = Column(String(32), default="upcoming", index=True)   # "upcoming", "active", "completed", "cancelled"
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    timetable_rel = relationship("Timetable", back_populates="sessions")
    subject_rel = relationship("Subject", back_populates="sessions")
    attendance_records = relationship("Attendance", back_populates="session_rel", cascade="all, delete-orphan")


class Attendance(Base):
    __tablename__ = "attendance"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("attendance_sessions.id"), nullable=False, index=True)
    student_id = Column(String(64), ForeignKey("students.student_id"), nullable=False, index=True)
    status = Column(String(16), default="ABSENT", nullable=False, index=True)  # "PRESENT", "LATE", "ABSENT"
    first_seen = Column(DateTime, nullable=True)
    last_seen = Column(DateTime, nullable=True)
    similarity_score = Column(Float, nullable=True)
    liveness_status = Column(String(32), nullable=True)  # "VERIFIED", "SKIPPED", "MANUAL"
    attendance_source = Column(String(32), default="FACE_RECOGNITION", nullable=False)  # "FACE_RECOGNITION", "STAFF_CORRECTION"
    correction_reason = Column(Text, nullable=True)
    modified_by = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint("session_id", "student_id", name="uq_session_student"),
    )

    session_rel = relationship("AttendanceSession", back_populates="attendance_records")
    student_rel = relationship("Student", back_populates="attendance_records")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    admin_id = Column(Integer, ForeignKey("admins.id"), nullable=True)
    staff_id = Column(String(64), nullable=True)
    action = Column(String(64), nullable=False, index=True)        # "MANUAL_ATTENDANCE_EDIT", "STUDENT_ENROLLED", etc.
    target_type = Column(String(64), nullable=False)               # "attendance", "student", "timetable", "system"
    target_id = Column(String(64), nullable=True)
    old_value = Column(Text, nullable=True)
    new_value = Column(Text, nullable=True)
    details = Column(Text, nullable=True)
    ip_address = Column(String(64), nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    admin = relationship("Admin", back_populates="audit_logs")


class SystemConfig(Base):
    __tablename__ = "system_configs"

    id = Column(Integer, primary_key=True, index=True)
    config_key = Column(String(64), unique=True, nullable=False, index=True)
    config_value = Column(Text, nullable=False)
    description = Column(String(256), nullable=True)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


def init_db():
    Base.metadata.create_all(bind=engine)
    migrate_db()


def migrate_db():
    """
    Backward-compatible migration runner for existing SQLite database.
    Safely adds missing columns to students, faculty, attendance, and audit_logs tables without dropping data,
    and backfills student profile photos and role passwords.
    """
    import sqlite3
    from backend.config import DB_PATH
    if not DB_PATH.exists():
        return

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # 1. Check students table columns
    cursor.execute("PRAGMA table_info(students)")
    student_cols = [row[1] for row in cursor.fetchall()]

    if "profile_photo_path" not in student_cols:
        cursor.execute("ALTER TABLE students ADD COLUMN profile_photo_path VARCHAR(256)")
        print("[DB Migration] Added column 'profile_photo_path' to students table.")

    if "password_hash" not in student_cols:
        cursor.execute("ALTER TABLE students ADD COLUMN password_hash VARCHAR(256)")
        print("[DB Migration] Added column 'password_hash' to students table.")

    # 2. Check faculty table columns
    cursor.execute("PRAGMA table_info(faculty)")
    faculty_cols = [row[1] for row in cursor.fetchall()]

    if "password_hash" not in faculty_cols:
        cursor.execute("ALTER TABLE faculty ADD COLUMN password_hash VARCHAR(256)")
        print("[DB Migration] Added column 'password_hash' to faculty table.")

    if "is_active" not in faculty_cols:
        cursor.execute("ALTER TABLE faculty ADD COLUMN is_active BOOLEAN DEFAULT 1")
        print("[DB Migration] Added column 'is_active' to faculty table.")

    # 3. Check attendance table columns
    cursor.execute("PRAGMA table_info(attendance)")
    attendance_cols = [row[1] for row in cursor.fetchall()]

    if "attendance_source" not in attendance_cols:
        cursor.execute("ALTER TABLE attendance ADD COLUMN attendance_source VARCHAR(32) DEFAULT 'FACE_RECOGNITION'")
        print("[DB Migration] Added column 'attendance_source' to attendance table.")

    if "correction_reason" not in attendance_cols:
        cursor.execute("ALTER TABLE attendance ADD COLUMN correction_reason TEXT")
        print("[DB Migration] Added column 'correction_reason' to attendance table.")

    if "modified_by" not in attendance_cols:
        cursor.execute("ALTER TABLE attendance ADD COLUMN modified_by VARCHAR(64)")
        print("[DB Migration] Added column 'modified_by' to attendance table.")

    # 4. Check audit_logs table columns
    cursor.execute("PRAGMA table_info(audit_logs)")
    audit_cols = [row[1] for row in cursor.fetchall()]

    if "staff_id" not in audit_cols:
        cursor.execute("ALTER TABLE audit_logs ADD COLUMN staff_id VARCHAR(64)")
        print("[DB Migration] Added column 'staff_id' to audit_logs table.")

    conn.commit()

    # 3. Backfill passwords and photo paths using ORM session
    from backend.auth import hash_password
    db = SessionLocal()
    try:
        # Backfill student passwords & profile photos
        students = db.query(Student).all()
        for s in students:
            if s.is_active is None or s.student_id == "23AD015":
                s.is_active = True
            if not s.password_hash:
                s.password_hash = hash_password("student123")
            if not s.profile_photo_path:
                front_emb = db.query(FaceEmbedding).filter(
                    FaceEmbedding.student_id == s.student_id,
                    FaceEmbedding.sample_angle == "front"
                ).first()
                if front_emb and front_emb.sample_image_path:
                    s.profile_photo_path = front_emb.sample_image_path
        
        # Backfill faculty passwords
        faculty_members = db.query(Faculty).all()
        for f in faculty_members:
            if not f.password_hash:
                f.password_hash = hash_password("staff123")

        db.commit()
    except Exception as ex:
        print(f"[DB Migration Error] {ex}")
        db.rollback()
    finally:
        db.close()
        conn.close()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
