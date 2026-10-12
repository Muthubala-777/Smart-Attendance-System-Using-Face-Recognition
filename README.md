# Smart Attendance System: High-Accuracy Multi-Face Recognition with Timetable-Based Session Management

## Key Features
- **Strictly QR-Free**: Automatic session activation/completion governed solely by the department timetable.
- **Biometric Pipeline**: OpenCV YuNet Face Detection + 5-point Affine Alignment + Face Quality Assessment (FQA) + ArcFace 512-D Deep Embeddings + Cosine Similarity.
- **Appearance Robustness**: Multi-angle enrollment (Front, Left, Right, Up, Down) to recognize students despite beards, clean shaves, hairstyles, or glasses.
- **Anti-Spoofing & Stability**: Eye Aspect Ratio (EAR) blink tracking, micro-motion analysis, and multi-frame temporal verification.
- **Real-Time Attendance**: Tracks `first_seen` arrival time and updates `last_seen`.
- **Institutional Reporting**: One-click download of certified CSV, Excel (.xlsx), and PDF reports.
- **Academic Benchmark**: Live empirical comparison of Base Paper (Haar + HOG + KNN) vs Proposed System.

---

## Quick Start Guide

### Step 1: Open Terminal in Project Folder
```powershell
cd "c:\Users\A.KANNAN\smart attendance system"
```

### Step 2: Install Dependencies (First Time Setup)
```powershell
pip install -r requirements.txt
```

### Step 3: Run the System
You can either double-click **`run.bat`** or run:
```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### Step 4: Access in Browser
Navigate to: **`http://127.0.0.1:8000/`**

### Step 5: Default Admin Credentials
- **Username**: `admin`
- **Password**: `admin123`

---

## Step-by-Step Operations

### 1. Enrolling a New Student
1. In the left navigation sidebar, click **Face Enrollment**.
2. Fill in the student's Roll Number, Full Name, Department, Year, and Section.
3. Click **Start Camera**.
4. Capture each of the 5 face angles:
   - **Front Face** (look directly at camera)
   - **Slight Left** (turn head ~15° left)
   - **Slight Right** (turn head ~15° right)
   - **Slight Up** (tilt head ~10° up)
   - **Slight Down** (tilt head ~10° down)
5. The FQA engine ensures blur variance > 65 and rejects multi-face or occluded frames.
6. Click **Save & Enroll Biometrics**.

### 2. Live Attendance Terminal
1. The background scheduler automatically detects the current day and active time slot.
2. Click **Live Attendance** in the sidebar.
3. Click **Start Camera**.
4. The system detects multiple faces simultaneously:
   - **Green Box**: Confirmed `PRESENT` with Student Name, ID, Confidence %, and `LIVE` badge.
   - **Blue Box**: Verifying identity across consecutive frames (`VERIFYING`).
   - **Red Box**: `UNKNOWN / INVALID STUDENT`.
5. The attendance ledger updates in real-time with `First Seen` and `Last Seen` timestamps.

### 3. Timetable Matrix & On-Demand Testing
1. Click **Timetable Matrix** in the sidebar to view weekly slots.
2. To test attendance for any slot immediately (without waiting for the clock), click the **⚡ Activate** button next to that slot.

### 4. Exporting Reports
1. Click **Reports (CSV/Excel/PDF)** in the sidebar.
2. Filter by date or department.
3. Click **Export to CSV**, **Export to Excel (.xlsx)**, or **Export to PDF Report** to download the file directly.

### 5. Running the Academic Benchmark
1. Click **Academic Evaluation** in the sidebar.
2. Click **Execute Live Benchmark ⚡**.
3. View the empirical comparison table showing real Accuracy, Precision, Recall, F1-Score, False Acceptance Rate (FAR), False Rejection Rate (FRR), FPS, and Latency between the Base Paper (Haar+KNN) and the Proposed System (YuNet+ArcFace).
