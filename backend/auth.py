import hashlib
import os
import hmac
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
import jwt
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from backend.config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from backend.database import get_db, Admin, Faculty, Student, AuditLog

security = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> dict:
    """
    Decodes JWT token and resolves identity for STUDENT, STAFF, or ADMIN.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        sub: str = payload.get("sub")
        role: str = payload.get("role", "student")
        if not sub:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials / Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Standardize roles
    if role in ["admin", "superadmin"]:
        admin = db.query(Admin).filter(Admin.username == sub).first()
        if not admin or not admin.is_active:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Admin not found or inactive")
        return {
            "id": admin.id,
            "username": admin.username,
            "role": "admin",
            "name": admin.name,
            "email": admin.email,
            "department": "Administration",
            "record": admin
        }
    elif role in ["staff", "faculty"]:
        fac = db.query(Faculty).filter((Faculty.faculty_id == sub) | (Faculty.email == sub)).first()
        if not fac or not getattr(fac, "is_active", True):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Staff account not found or inactive")
        dept_name = fac.department_rel.code if fac.department_rel else ""
        return {
            "id": fac.id,
            "faculty_id": fac.faculty_id,
            "username": fac.faculty_id,
            "role": "staff",
            "name": fac.name,
            "email": fac.email,
            "department": dept_name,
            "record": fac
        }
    elif role == "student":
        stu = db.query(Student).filter(Student.student_id == sub).first()
        if not stu or not stu.is_active:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Student not found or inactive")
        return {
            "id": stu.id,
            "student_id": stu.student_id,
            "username": stu.student_id,
            "role": "student",
            "name": stu.name,
            "email": stu.email,
            "department": stu.department,
            "year": stu.year,
            "section": stu.section,
            "record": stu
        }
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unrecognized user role in token")


def require_roles(allowed_roles: List[str]):
    """FastAPI dependency to enforce role-based access control."""
    def role_checker(current_user: dict = Depends(get_current_user)):
        user_role = current_user.get("role")
        # Allow superadmin whenever admin is allowed
        if user_role in allowed_roles or (user_role in ["admin", "superadmin"] and "admin" in allowed_roles):
            return current_user
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied: Role '{user_role}' not permitted for this resource."
        )
    return role_checker


def get_current_admin(
    current_user: dict = Depends(require_roles(["admin"]))
) -> Admin:
    """Returns the Admin ORM record for existing admin endpoints."""
    return current_user["record"]


def hash_password(password: str) -> str:
    """Uses PBKDF2-HMAC-SHA256 with a unique salt for secure password hashing."""
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    return f"{salt.hex()}${key.hex()}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain text password against the salted PBKDF2 hash."""
    try:
        salt_hex, key_hex = hashed_password.split("$")
        salt = bytes.fromhex(salt_hex)
        expected_key = bytes.fromhex(key_hex)
        key = hashlib.pbkdf2_hmac("sha256", plain_password.encode("utf-8"), salt, 100000)
        return hmac.compare_digest(key, expected_key)
    except Exception:
        return False


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt




def log_admin_action(
    db: Session,
    admin_id: Optional[int] = None,
    action: str = "",
    target_type: str = "",
    target_id: Optional[str] = None,
    old_value: Optional[str] = None,
    new_value: Optional[str] = None,
    details: Optional[str] = None,
    ip_address: Optional[str] = None,
    staff_id: Optional[str] = None,
):
    """Creates a persistent audit log entry for regulatory and academic compliance."""
    log_entry = AuditLog(
        admin_id=admin_id,
        staff_id=staff_id,
        action=action,
        target_type=target_type,
        target_id=target_id,
        old_value=old_value,
        new_value=new_value,
        details=details,
        ip_address=ip_address,
        timestamp=datetime.now(timezone.utc)
    )
    db.add(log_entry)
    db.commit()
    db.refresh(log_entry)
    return log_entry


# Alias for general audit logging
log_audit_action = log_admin_action

