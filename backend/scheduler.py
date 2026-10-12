import threading
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from backend.database import SessionLocal, Timetable, AttendanceSession, Attendance, Student, Subject


class TimetableScheduler:
    """
    Autonomous Background Timetable Scheduler Daemon.
    Continuously monitors the system clock and the academic timetable:
    
    1. At slot start_time:
       - Automatically spins up an AttendanceSession with status="active".
       - Automatically pre-populates all registered students for that Department/Year/Section with STATUS="ABSENT".
       - Prevents duplicate sessions for the same timetable slot on the same date.
       
    2. At slot end_time:
       - Automatically marks the session as "completed".
       
    3. Exposes real-time state for the Live Attendance dashboard without any manual or QR activation.
    """

    def __init__(self, poll_interval_seconds: int = 10):
        self.poll_interval = poll_interval_seconds
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.current_active_session_id: Optional[int] = None
        self.last_check_status: str = "Initialized"

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True, name="TimetableSchedulerWorker")
        self.thread.start()
        print("[TimetableScheduler] Autonomous Timetable Scheduler Daemon started.")

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        print("[TimetableScheduler] Scheduler Daemon stopped.")

    def _run_loop(self):
        while self.is_running:
            try:
                self.check_and_update_sessions()
            except Exception as e:
                print(f"[TimetableScheduler] Exception in scheduler loop: {e}")
            time.sleep(self.poll_interval)

    def get_current_time_slot(self) -> Tuple[str, str, str]:
        """
        Returns (current_date_str "YYYY-MM-DD", current_day "Monday", current_time_str "HH:MM")
        """
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        day_str = now.strftime("%A")
        time_str = now.strftime("%H:%M")
        return date_str, day_str, time_str

    def check_and_update_sessions(self):
        """
        Core timetable synchronization logic.
        """
        db: Session = SessionLocal()
        try:
            now = datetime.now()
            today_date = now.strftime("%Y-%m-%d")
            today_day = now.strftime("%A")
            current_time = now.strftime("%H:%M")

            # 1. Close any active sessions whose end_time has passed
            active_sessions = db.query(AttendanceSession).filter(
                AttendanceSession.status == "active"
            ).all()

            for sess in active_sessions:
                # If date is in past OR date is today but current_time >= end_time
                if sess.session_date < today_date or (sess.session_date == today_date and current_time >= sess.end_time):
                    sess.status = "completed"
                    sess.updated_at = datetime.now(timezone.utc)
                    db.commit()
                    print(f"[TimetableScheduler] Auto-completed Session #{sess.id} ({sess.department} {sess.year}-{sess.section} Subject ID {sess.subject_id}).")
                    if self.current_active_session_id == sess.id:
                        self.current_active_session_id = None

            # 2. Check for matching timetable slots right now
            slots = db.query(Timetable).filter(
                Timetable.is_active == True,
                Timetable.day_of_week == today_day
            ).all()

            for slot in slots:
                # Slot is current if start_time <= current_time < end_time
                if slot.start_time <= current_time < slot.end_time:
                    # Check if session already created today for this timetable slot
                    existing_session = db.query(AttendanceSession).filter(
                        AttendanceSession.timetable_id == slot.id,
                        AttendanceSession.session_date == today_date
                    ).first()

                    if not existing_session:
                        # Auto-create active session
                        new_session = AttendanceSession(
                            timetable_id=slot.id,
                            subject_id=slot.subject_id,
                            department=slot.department,
                            year=slot.year,
                            section=slot.section,
                            session_date=today_date,
                            start_time=slot.start_time,
                            end_time=slot.end_time,
                            status="active"
                        )
                        db.add(new_session)
                        db.commit()
                        db.refresh(new_session)

                        # Auto-populate all enrolled section students with STATUS="ABSENT"
                        students = db.query(Student).filter(
                            Student.department == slot.department,
                            Student.year == slot.year,
                            Student.section == slot.section,
                            Student.is_active == True
                        ).all()

                        for stu in students:
                            att = Attendance(
                                session_id=new_session.id,
                                student_id=stu.student_id,
                                status="ABSENT"
                            )
                            db.add(att)
                        db.commit()

                        self.current_active_session_id = new_session.id
                        print(f"[TimetableScheduler] >>> AUTO-ACTIVATED SESSION #{new_session.id} for {slot.department} {slot.year}-{slot.section} ({slot.start_time}-{slot.end_time}) with {len(students)} students initialized to ABSENT.")
                    
                    elif existing_session.status == "upcoming":
                        existing_session.status = "active"
                        existing_session.updated_at = datetime.now(timezone.utc)
                        db.commit()
                        self.current_active_session_id = existing_session.id
                        print(f"[TimetableScheduler] Activated upcoming Session #{existing_session.id}.")

            # Update latest active session pointer
            current_active = db.query(AttendanceSession).filter(
                AttendanceSession.status == "active"
            ).order_by(AttendanceSession.id.desc()).first()

            if current_active:
                self.current_active_session_id = current_active.id
                self.last_check_status = f"Active session #{current_active.id} ({current_active.department} {current_active.year}-{current_active.section})"
            else:
                self.current_active_session_id = None
                self.last_check_status = f"No active slot at {current_time} on {today_day}."

        finally:
            db.close()

    def get_current_active_session(self, db: Session) -> Optional[AttendanceSession]:
        """
        Retrieves the current active attendance session.
        If no active session exists by clock, allows picking the most recent active or creates on-demand if slot configured.
        """
        session = db.query(AttendanceSession).filter(
            AttendanceSession.status == "active"
        ).order_by(AttendanceSession.id.desc()).first()
        return session


# Global scheduler singleton
scheduler_engine = TimetableScheduler(poll_interval_seconds=10)
