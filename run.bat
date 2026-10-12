@echo off
title Smart Attendance System - High-Accuracy Multi-Face Recognition
echo =====================================================================
echo           SMART ATTENDANCE SYSTEM (TIMETABLE-DRIVEN, QR-FREE)
echo =====================================================================
echo.
echo Starting FastAPI + Uvicorn server on http://127.0.0.1:8000 ...
echo.

start "" http://127.0.0.1:8000/

python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
pause
