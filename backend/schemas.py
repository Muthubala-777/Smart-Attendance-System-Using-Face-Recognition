from typing import List, Optional, Dict, Any
from pydantic import BaseModel, EmailStr


# Auth
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    admin_name: Optional[str] = None
    name: str
    role: str  # "student", "staff", "admin"
    user_id: str
    department: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str
    role: Optional[str] = None


# Department
class DepartmentCreate(BaseModel):
    code: str
    name: str


class DepartmentOut(DepartmentCreate):
    id: int

    class Config:
        from_attributes = True


# Subject
class SubjectCreate(BaseModel):
    code: str
    name: str
    department_id: int


class SubjectOut(SubjectCreate):
    id: int
    department_code: Optional[str] = None

    class Config:
        from_attributes = True


# Faculty
class FacultyCreate(BaseModel):
    faculty_id: str
    name: str
    email: str
    phone: Optional[str] = None
    department_id: int


class FacultyOut(FacultyCreate):
    id: int

    class Config:
        from_attributes = True


# Student
class StudentCreate(BaseModel):
    student_id: str
    name: str
    department: str
    year: str
    section: str
    email: Optional[str] = None
    phone: Optional[str] = None
    password: Optional[str] = "student123"
    academic_year: str = "2025-2026"
    semester: str = "VII"


class StudentUpdate(BaseModel):
    name: Optional[str] = None
    department: Optional[str] = None
    year: Optional[str] = None
    section: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    is_active: Optional[bool] = None


class FaceEmbeddingOut(BaseModel):
    id: int
    sample_angle: str
    quality_score: float
    created_at: str


class StudentOut(BaseModel):
    id: int
    student_id: str
    name: str
    department: str
    year: str
    section: str
    email: Optional[str] = None
    phone: Optional[str] = None
    academic_year: str
    semester: str
    is_active: bool
    created_at: str
    profile_photo_url: Optional[str] = None
    face_sample_count: int = 0
    face_samples: List[FaceEmbeddingOut] = []

    class Config:
        from_attributes = True


# Timetable
class TimetableCreate(BaseModel):
    department: str
    year: str
    section: str
    day_of_week: str
    start_time: str
    end_time: str
    subject_id: int
    faculty_id: Optional[int] = None
    classroom: str = "Room-101"
    academic_year: str = "2025-2026"
    semester: str = "VII"


class TimetableOut(TimetableCreate):
    id: int
    is_active: bool
    subject_name: Optional[str] = None
    faculty_name: Optional[str] = None

    class Config:
        from_attributes = True


# Attendance Session
class AttendanceSessionOut(BaseModel):
    id: int
    timetable_id: Optional[int] = None
    subject_id: int
    subject_name: Optional[str] = None
    department: str
    year: str
    section: str
    session_date: str
    start_time: str
    end_time: str
    status: str
    total_students: int = 0
    present_students: int = 0
    absent_students: int = 0


# Attendance Record
class AttendanceRecordOut(BaseModel):
    id: int
    session_id: int
    student_id: str
    student_name: str
    department: str
    year: str
    section: str
    status: str
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    similarity_score: Optional[float] = None
    liveness_status: Optional[str] = None
    photo_url: Optional[str] = None
    subject_name: Optional[str] = None
    session_date: Optional[str] = None
    attendance_source: Optional[str] = "FACE_RECOGNITION"
    correction_reason: Optional[str] = None
    modified_by: Optional[str] = None


class ManualAttendanceUpdate(BaseModel):
    attendance_id: int
    new_status: str  # "PRESENT", "LATE", or "ABSENT"
    reason: Optional[str] = "Manual administrative update"


class AttendanceEditRequest(BaseModel):
    attendance_id: int
    new_status: str  # "PRESENT", "LATE", "ABSENT"
    new_time: Optional[str] = None  # e.g. "09:02:37 AM"
    new_date: Optional[str] = None
    reason: str  # Mandatory correction reason



# Frame Processing
class ProcessFrameRequest(BaseModel):
    image_base64: str  # JPEG / PNG data URL or base64 string
    session_id: Optional[int] = None


class DetectedFaceOut(BaseModel):
    bbox: List[int]  # [x, y, w, h]
    student_id: str
    student_name: str
    status: str  # "PRESENT", "VERIFYING", "INVALID"
    confidence: float
    liveness_status: str
    liveness_score: float
    quality_score: float
    message: str
    matched: bool = False
    attendance_marked: bool = False
    already_marked: bool = False
    first_seen: Optional[str] = None
    photo_url: Optional[str] = None


class ProcessFrameResponse(BaseModel):
    total_faces: int
    active_session_id: Optional[int] = None
    active_subject: Optional[str] = None
    detected_faces: List[DetectedFaceOut]
    timestamp: str


# System Settings
class SystemSettingsUpdate(BaseModel):
    recognition_threshold: Optional[float] = None
    blur_threshold: Optional[float] = None
    min_face_size: Optional[int] = None
    temporal_frames_required: Optional[int] = None


# Student Dashboard Schemas
class StudentProfileOut(BaseModel):
    id: int
    student_id: str
    name: str
    department: str
    year: str
    section: str
    email: Optional[str] = None
    phone: Optional[str] = None
    academic_year: str
    semester: str
    profile_photo_url: str
    total_classes: int = 0
    present_count: int = 0
    absent_count: int = 0
    late_count: int = 0
    attendance_percentage: float = 0.0


class StudentAttendanceHistoryItem(BaseModel):
    id: int
    session_id: int
    subject_name: str
    subject_code: str
    date: str
    time_slot: str
    classroom: str
    faculty_name: Optional[str] = None
    status: str
    entry_time: Optional[str] = None
    similarity: Optional[float] = None
    liveness: Optional[str] = None
    source: Optional[str] = "FACE_RECOGNITION"


# Staff Dashboard Schemas
class StaffSessionDetailsOut(BaseModel):
    has_active_session: bool
    session_id: Optional[int] = None
    timetable_id: Optional[int] = None
    subject_name: Optional[str] = None
    subject_code: Optional[str] = None
    department: Optional[str] = None
    year: Optional[str] = None
    section: Optional[str] = None
    session_date: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    classroom: Optional[str] = None
    faculty_name: Optional[str] = None
    status: Optional[str] = None
    total_students: int = 0
    present_count: int = 0
    absent_count: int = 0
    late_count: int = 0
    attendance_percentage: float = 0.0
    message: Optional[str] = None


class ClassStudentOut(BaseModel):
    student_id: str
    name: str
    department: str
    year: str
    section: str
    photo_url: str
    status: str  # "PRESENT", "ABSENT", "NOT VERIFIED"
    entry_time: Optional[str] = None
    similarity: Optional[float] = None
    liveness: Optional[str] = None
