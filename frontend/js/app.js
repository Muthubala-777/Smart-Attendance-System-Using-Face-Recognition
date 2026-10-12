/**
 * SMART ATTENDANCE SYSTEM - FRONTEND APPLICATION CORE
 * Multi-Face Recognition, ArcFace Deep Embeddings, and Timetable-Based Session Management
 * Strictly QR-Free.
 */

// Global State
const state = {
  token: localStorage.getItem("token") || null,
  user: JSON.parse(localStorage.getItem("user") || "null"),
  role: localStorage.getItem("role") || null,
  admin: JSON.parse(localStorage.getItem("admin") || "null"), // backward compatibility
  currentView: "view-dashboard",
  activeSession: null,
  isCameraRunning: false,
  webcamStream: null,
  isStudentCameraRunning: false,
  studentWebcamStream: null,
  isEnrollCameraRunning: false,
  enrollStream: null,
  enrollSamples: {}, // angle -> dataURL
  activeEnrollAngle: "front",
  lastFrameTime: performance.now(),
  fps: 0,
  staffPollingTimer: null,
  holdDurationMs: 0,
  lastPoseTime: 0,
  uploadedFile: null,
};

// Available Face Enrollment Angles (Intuitive Directives with Flexible Ranges)
const ENROLL_ANGLES = [
  { id: "front", name: "Look Front", desc: "Look directly into camera with natural expression" },
  { id: "slight_left", name: "Turn Left", desc: "Turn head gently toward the left" },
  { id: "slight_right", name: "Turn Right", desc: "Turn head gently toward the right" },
  { id: "slight_up", name: "Look Up", desc: "Tilt chin slightly upward" },
  { id: "slight_down", name: "Look Down", desc: "Tilt chin slightly downward" },
];

// Helper: Toast Notifications
function showToast(message, type = "info") {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${type === 'success' ? '✅' : type === 'danger' ? '❌' : 'ℹ️'}</span> <div>${message}</div>`;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

// Helper: Authenticated API Fetch
async function apiFetch(endpoint, options = {}) {
  const headers = options.headers || {};
  if (state.token) {
    headers["Authorization"] = `Bearer ${state.token}`;
  }
  if (!(options.body instanceof FormData) && options.body && typeof options.body === "object") {
    headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(options.body);
  }
  options.headers = headers;

  try {
    const res = await fetch(endpoint, options);
    if (res.status === 401) {
      // Auth expired
      state.token = null;
      localStorage.removeItem("token");
      openLoginModal();
      throw new Error("Authentication required");
    }
    return res;
  } catch (err) {
    console.error(`API Error on ${endpoint}:`, err);
    throw err;
  }
}

// Navigation & View Switching
function switchView(viewId) {
  // Role Access Guard
  if (state.role === "student") {
    const studentAllowed = ["view-student-dashboard", "view-student-history", "view-student-profile"];
    if (!studentAllowed.includes(viewId)) {
      showToast("Access restricted: Student Portal only", "warning");
      viewId = "view-student-dashboard";
    }
  } else if (state.role === "staff") {
    const staffAllowed = [
      "view-staff-dashboard",
      "view-live",
      "view-students",
      "view-enroll",
      "view-history",
      "view-reports",
    ];
    if (!staffAllowed.includes(viewId)) {
      showToast("Access restricted to Faculty resources", "warning");
      viewId = "view-staff-dashboard";
    }
  }

  // Cleanly release camera streams if navigating away to prevent device locking
  if (state.currentView === "view-student-dashboard" && viewId !== "view-student-dashboard") {
    stopStudentCamera();
  }
  if (state.currentView === "view-enroll" && viewId !== "view-enroll") {
    stopEnrollCamera();
  }
  if (state.currentView === "view-live" && viewId !== "view-live") {
    stopLiveCamera();
  }
  if (state.currentView === "view-staff-dashboard" && viewId !== "view-staff-dashboard") {
    if (state.staffPollingTimer) {
      clearInterval(state.staffPollingTimer);
      state.staffPollingTimer = null;
    }
  }

  state.currentView = viewId;
  document.querySelectorAll(".view-section").forEach((sec) => sec.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach((item) => item.classList.remove("active"));

  const targetSec = document.getElementById(viewId);
  if (targetSec) targetSec.classList.add("active");

  const targetNav = document.querySelector(`.nav-item[data-view="${viewId}"]`);
  if (targetNav) targetNav.classList.add("active");

  // Update Page Title
  const titleMap = {
    "view-student-dashboard": "Student Portal - Attendance & Biometrics",
    "view-student-history": "My Verified Attendance Record",
    "view-student-profile": "Student Identity & Biometric Profile",
    "view-staff-dashboard": "Faculty Class Session & Live Attendance",
    "view-dashboard": "Live Dashboard",
    "view-live": "Live Attendance Terminal",
    "view-enroll": "Student Face Enrollment Studio",
    "view-students": "Student Directory & Gallery",
    "view-timetable": "Academic Timetable Matrix",
    "view-subjects": "Subjects & Faculty Directory",
    "view-history": "Attendance History & Audited Overrides",
    "view-reports": "Institutional Reports Exporter",
    "view-benchmark": "Academic Baseline vs Proposed Benchmark",
    "view-audit": "Administrative Audit Trail",
    "view-settings": "Computer Vision Parameters",
  };
  document.getElementById("pageTitleText").textContent = titleMap[viewId] || "VisionAttend";

  // Trigger View-Specific Loads
  if (viewId === "view-student-dashboard") loadStudentDashboard();
  if (viewId === "view-student-history") loadStudentAttendanceHistory();
  if (viewId === "view-student-profile") loadStudentProfileView();
  if (viewId === "view-staff-dashboard") loadStaffDashboard();
  if (viewId === "view-dashboard") loadDashboardData();
  if (viewId === "view-students") loadStudentsList();
  if (viewId === "view-timetable") loadTimetable();
  if (viewId === "view-subjects") loadSubjectsAndFaculty();
  if (viewId === "view-history") loadAttendanceHistory();
  if (viewId === "view-audit") loadAuditLogs();
  if (viewId === "view-settings") loadSystemSettings();
  if (viewId === "view-live") checkActiveSession();
}

// Live Clock Formatter: Exact format "23 September 2026, 02:45:32 PM"
function formatLiveClock(date = new Date()) {
  const day = date.getDate();
  const monthNames = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
  ];
  const month = monthNames[date.getMonth()];
  const year = date.getFullYear();

  let hours = date.getHours();
  const minutes = String(date.getMinutes()).padStart(2, "0");
  const seconds = String(date.getSeconds()).padStart(2, "0");
  const ampm = hours >= 12 ? "PM" : "AM";
  hours = hours % 12;
  hours = hours ? String(hours).padStart(2, "0") : "12";

  return `${day} ${month} ${year}, ${hours}:${minutes}:${seconds} ${ampm}`;
}

// Clock & Timetable Header Polling (visual live display, authoritative timestamp is server-side)
function startClock() {
  function tick() {
    const formatted = formatLiveClock();
    const clockEl = document.getElementById("liveClockDisplay");
    if (clockEl) clockEl.textContent = formatted;

    document.querySelectorAll(".live-clock-text").forEach((el) => {
      el.textContent = formatted;
    });
  }
  tick();
  setInterval(tick, 1000);
}

// Check Active Timetable Session
async function checkActiveSession() {
  try {
    const res = await apiFetch("/api/sessions/current");
    const data = await res.json();
    const indText = document.getElementById("timetableStatusText");
    const indEl = document.getElementById("timetableStatusIndicator");

    if (data.has_active_session) {
      state.activeSession = data;
      indText.textContent = `Active: ${data.subject_name} (${data.department} ${data.year}-${data.section})`;
      indEl.style.borderColor = "rgba(16, 185, 129, 0.4)";

      // Update Dashboard Banner
      document.getElementById("bannerSubjectName").textContent = `${data.subject_name} (${data.subject_code || ''})`;
      document.getElementById("bannerDeptYear").textContent = `Class: ${data.department} ${data.year}-${data.section}`;
      document.getElementById("bannerSlot").textContent = `Slot: ${data.start_time} - ${data.end_time}`;
      document.getElementById("bannerStatus").textContent = `Session #${data.session_id} - ACTIVE`;

      // Update Live Attendance Screen Header
      document.getElementById("liveSessionTitle").textContent = `${data.subject_name} (${data.department} ${data.year}-${data.section})`;
      document.getElementById("liveSessionSubtitle").textContent = `Auto Timetable Slot: ${data.start_time} - ${data.end_time} | Session #${data.session_id}`;
      document.getElementById("liveSessionBadge").textContent = "ACTIVE";
      document.getElementById("liveSessionBadge").className = "badge badge-present";

      loadLiveAttendanceList(data.session_id);
    } else {
      state.activeSession = null;
      indText.textContent = "No Timetable Session Active";
      indEl.style.borderColor = "rgba(245, 158, 11, 0.4)";

      document.getElementById("bannerSubjectName").textContent = "No Active Timetable Session";
      document.getElementById("bannerDeptYear").textContent = "Department: None";
      document.getElementById("bannerSlot").textContent = "Slot: None";
      document.getElementById("bannerStatus").textContent = "Status: IDLE";

      document.getElementById("liveSessionTitle").textContent = "No Active Session";
      document.getElementById("liveSessionSubtitle").textContent = "Attendance session activates automatically when scheduled slot arrives.";
      document.getElementById("liveSessionBadge").textContent = "IDLE";
      document.getElementById("liveSessionBadge").className = "badge badge-absent";
    }
  } catch (err) {
    console.error("Timetable check failed:", err);
  }
}

// -------------------------------------------------------------
// DASHBOARD LOGIC
// -------------------------------------------------------------
async function loadDashboardData() {
  await checkActiveSession();
  try {
    const res = await apiFetch("/api/dashboard/stats");
    const data = await res.json();

    document.getElementById("statTotalStudents").textContent = data.total_students || 0;
    document.getElementById("statTodayPresent").textContent = data.today_present || 0;
    document.getElementById("statTodayAbsent").textContent = data.today_absent || 0;
    document.getElementById("statAttendancePct").textContent = `${data.attendance_percentage || 0}%`;

    // Feed table
    const feedBody = document.getElementById("dashboardFeedBody");
    if (data.recent_feed && data.recent_feed.length > 0) {
      feedBody.innerHTML = data.recent_feed
        .map(
          (item) => `
        <tr>
          <td><strong>${item.student_id}</strong></td>
          <td>${item.name}</td>
          <td>${item.department}</td>
          <td>${item.time}</td>
          <td><span class="badge badge-present">${item.confidence}</span></td>
          <td><span class="badge badge-live">LIVE</span></td>
          <td><span class="badge badge-present">PRESENT</span></td>
        </tr>
      `
        )
        .join("");
    } else {
      feedBody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted);">No attendance recorded yet today. Launch Live Terminal to take attendance.</td></tr>`;
    }
  } catch (err) {
    console.error("Failed to load dashboard stats:", err);
  }
}

// -------------------------------------------------------------
// LIVE ATTENDANCE TERMINAL LOGIC
// -------------------------------------------------------------
async function startLiveCamera() {
  const video = document.getElementById("webcamVideo");
  const overlay = document.getElementById("cameraOverlay");
  const btnStart = document.getElementById("btnStartCamera");
  const btnStop = document.getElementById("btnStopCamera");

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
    });
    state.webcamStream = stream;
    video.srcObject = stream;
    await video.play();

    overlay.width = video.videoWidth || 640;
    overlay.height = video.videoHeight || 480;

    state.isCameraRunning = true;
    btnStart.style.display = "none";
    btnStop.style.display = "inline-flex";

    showToast("Live Camera initialized. Multi-Face inference running.", "success");
    runFrameProcessingLoop();
  } catch (err) {
    console.error("Camera access failed:", err);
    showToast("Webcam error: " + err.message, "danger");
  }
}

function stopLiveCamera() {
  state.isCameraRunning = false;
  if (state.webcamStream) {
    state.webcamStream.getTracks().forEach((track) => track.stop());
    state.webcamStream = null;
  }
  const video = document.getElementById("webcamVideo");
  video.srcObject = null;

  const overlay = document.getElementById("cameraOverlay");
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  document.getElementById("btnStartCamera").style.display = "inline-flex";
  document.getElementById("btnStopCamera").style.display = "none";
}

async function runFrameProcessingLoop() {
  if (!state.isCameraRunning) return;

  const video = document.getElementById("webcamVideo");
  const overlay = document.getElementById("cameraOverlay");
  const ctx = overlay.getContext("2d");

  // Create an offscreen canvas to capture frame
  const offCanvas = document.createElement("canvas");
  offCanvas.width = video.videoWidth || 640;
  offCanvas.height = video.videoHeight || 480;
  const offCtx = offCanvas.getContext("2d");
  offCtx.drawImage(video, 0, 0, offCanvas.width, offCanvas.height);

  const base64Data = offCanvas.toDataURL("image/jpeg", 0.85);

  try {
    const sessionId = state.activeSession ? state.activeSession.session_id : null;
    const res = await apiFetch("/api/attendance/process-frame", {
      method: "POST",
      body: { image_base64: base64Data, session_id: sessionId },
    });

    const data = await res.json();

    // Calculate FPS
    const now = performance.now();
    state.fps = Math.round(1000 / (now - state.lastFrameTime));
    state.lastFrameTime = now;
    document.getElementById("liveFpsBadge").textContent = `FPS: ${state.fps}`;

    // Render Overlays
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    renderBoundingBoxes(ctx, data.detected_faces);

    // If attendance was logged and we have active session, refresh list periodically
    if (state.activeSession) {
      loadLiveAttendanceList(state.activeSession.session_id);
    }
  } catch (err) {
    console.error("Frame processing error:", err);
  }

  // Next frame schedule (~12-15 FPS is optimal for deep multi-face detection without browser lag)
  if (state.isCameraRunning) {
    setTimeout(runFrameProcessingLoop, 80);
  }
}

function renderBoundingBoxes(ctx, faces, isMirrored = false) {
  if (!faces || faces.length === 0) return;
  const canvasW = ctx.canvas.width;
  const canvasH = ctx.canvas.height;

  faces.forEach((face) => {
    const [origX, origY, w, h] = face.bbox;
    // Map X coordinate if preview video is horizontally mirrored (selfie mode)
    const x = isMirrored ? (canvasW - (origX + w)) : origX;
    const y = origY;

    const isPresent = face.status === "PRESENT";
    const isVerifying = face.status === "VERIFYING";

    // Color Code: Green for PRESENT, Blue for VERIFYING, Red for UNKNOWN / INVALID
    let boxColor = "#ef4444"; // danger / unknown
    let labelBg = "rgba(239, 68, 68, 0.90)";

    if (isPresent) {
      boxColor = "#10b981"; // success / present
      labelBg = "rgba(16, 185, 129, 0.92)";
    } else if (isVerifying) {
      boxColor = "#3b82f6"; // verifying
      labelBg = "rgba(59, 130, 246, 0.90)";
    }

    // 1. Draw Bounding Box with rounded corners
    ctx.strokeStyle = boxColor;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, 8);
    ctx.stroke();

    // 2. Corner decorative accents
    const len = Math.min(16, Math.floor(w / 4));
    ctx.lineWidth = 4;
    ctx.strokeStyle = boxColor;
    // Top-left
    ctx.beginPath();
    ctx.moveTo(x, y + len); ctx.lineTo(x, y); ctx.lineTo(x + len, y);
    ctx.stroke();
    // Top-right
    ctx.beginPath();
    ctx.moveTo(x + w - len, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + len);
    ctx.stroke();
    // Bottom-left
    ctx.beginPath();
    ctx.moveTo(x, y + h - len); ctx.lineTo(x, y + h); ctx.lineTo(x + len, y + h);
    ctx.stroke();
    // Bottom-right
    ctx.beginPath();
    ctx.moveTo(x + w - len, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - len);
    ctx.stroke();

    // 3. Header Label (UPRIGHT, READABLE, HORIZONTAL, NEVER ROTATED, NEVER INVERTED)
    ctx.font = "bold 12px Inter, system-ui, -apple-system, sans-serif";
    const confPct = Math.round(face.confidence * 100);
    const tagText = `${face.student_name} • ${confPct}%`;
    const textWidth = ctx.measureText(tagText).width;
    const tagWidth = Math.max(textWidth + 20, 140);
    const tagHeight = 24;

    // Label Y position: place ABOVE bounding box; if not enough room at top (< 28px), place inside top of box
    let labelY = y - tagHeight - 4;
    if (labelY < 4) {
      labelY = y + 4; // Intelligently inside top of box, still upright!
    }

    // Ensure label does not overflow canvas edges horizontally
    let labelX = Math.max(4, Math.min(x, canvasW - tagWidth - 4));

    // Label Background pill
    ctx.fillStyle = labelBg;
    ctx.beginPath();
    ctx.roundRect(labelX, labelY, tagWidth, tagHeight, 4);
    ctx.fill();

    // Upright Label Text
    ctx.fillStyle = "#ffffff";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(tagText, labelX + 10, labelY + (tagHeight / 2));

    // 4. Status Subtext Footer (Only if enough space below)
    if (y + h + 26 <= canvasH) {
      const subText = `${face.status}${face.liveness_status ? ' • ' + face.liveness_status : ''}`;
      ctx.font = "bold 10px Inter, system-ui, sans-serif";
      const subWidth = Math.max(ctx.measureText(subText).width + 16, 110);
      const subHeight = 18;
      const subY = y + h + 4;
      const subX = Math.max(4, Math.min(x, canvasW - subWidth - 4));

      ctx.fillStyle = "rgba(0, 0, 0, 0.78)";
      ctx.beginPath();
      ctx.roundRect(subX, subY, subWidth, subHeight, 3);
      ctx.fill();

      ctx.fillStyle = isPresent ? "#34d399" : (isVerifying ? "#60a5fa" : "#fca5a5");
      ctx.fillText(subText, subX + 8, subY + (subHeight / 2));
    }
  });
}

// =============================================================
// STUDENT DASHBOARD & CAMERA CONTROLLERS
// =============================================================

async function loadStudentDashboard() {
  try {
    const res = await apiFetch("/api/student/profile");
    const data = await res.json();

    document.getElementById("studentWelcomeName").textContent = `Welcome, ${data.name}`;
    document.getElementById("studentRollBadge").textContent = `Roll No: ${data.student_id}`;
    document.getElementById("studentDeptBadge").textContent = `Department: ${data.department}`;
    document.getElementById("studentClassBadge").textContent = `Class: ${data.year}-${data.section}`;

    document.getElementById("profileStudentName").textContent = data.name;
    document.getElementById("profileStudentRoll").textContent = data.student_id;
    document.getElementById("profileStudentDept").textContent = data.department;
    document.getElementById("profileStudentYrSec").textContent = `${data.year} Year, Sec ${data.section}`;
    document.getElementById("profileStudentAcdYr").textContent = `${data.academic_year} (Sem ${data.semester})`;
    document.getElementById("profileStudentPct").textContent = `${data.attendance_percentage}%`;

    const img = document.getElementById("studentPhotoImg");
    if (img) img.src = `${data.profile_photo_url}?t=${Date.now()}`;

    document.getElementById("stuStatTotal").textContent = data.total_classes;
    document.getElementById("stuStatPresent").textContent = data.present_count;
    document.getElementById("stuStatAbsent").textContent = data.absent_count;
    document.getElementById("stuStatLate").textContent = data.late_count;
    document.getElementById("stuProgressPct").textContent = `${data.attendance_percentage}%`;
    document.getElementById("stuProgressBar").style.width = `${Math.min(100, data.attendance_percentage)}%`;

    // Active session info
    checkStudentActiveSession();
    // Load attendance table
    loadStudentAttendanceHistory();
  } catch (err) {
    console.error("Error loading student profile:", err);
  }
}

async function checkStudentActiveSession() {
  try {
    const res = await apiFetch("/api/sessions/current");
    const data = await res.json();
    if (data.has_active_session) {
      document.getElementById("stuActiveSubject").textContent = `${data.subject_code} - ${data.subject_name}`;
      document.getElementById("stuActiveTime").textContent = `${data.start_time} - ${data.end_time}`;
      document.getElementById("stuActiveRoom").textContent = `${data.department} ${data.year}-${data.section}`;
      document.getElementById("stuActiveFaculty").textContent = data.status.toUpperCase();
    } else {
      document.getElementById("stuActiveSubject").textContent = "No session currently active";
      document.getElementById("stuActiveTime").textContent = "--";
      document.getElementById("stuActiveRoom").textContent = "--";
      document.getElementById("stuActiveFaculty").textContent = "--";
    }
  } catch (err) {
    console.error("Error checking student session:", err);
  }
}

async function loadStudentAttendanceHistory() {
  try {
    const res = await apiFetch("/api/student/attendance");
    const records = await res.json();

    const tbody1 = document.getElementById("studentAttendanceTableBody");
    const tbody2 = document.getElementById("studentFullHistoryTableBody");

    if (!records || records.length === 0) {
      const emptyRow = '<tr><td colspan="9" class="text-center py-4">No verified attendance records found.</td></tr>';
      if (tbody1) tbody1.innerHTML = emptyRow;
      if (tbody2) tbody2.innerHTML = emptyRow;
      return;
    }

    const rowsHtml1 = records.slice(0, 10).map(r => `
      <tr>
        <td class="font-bold">${r.date}</td>
        <td>${r.time_slot}</td>
        <td><span class="font-mono text-primary">${r.subject_code}</span> ${r.subject_name}</td>
        <td>${r.classroom}</td>
        <td class="font-mono font-bold text-success">${r.entry_time || '--'}</td>
        <td class="font-mono">${r.similarity ? (r.similarity * 100).toFixed(1) + '%' : '--'}</td>
        <td><span class="badge badge-success">${r.liveness || 'VERIFIED'}</span></td>
        <td><span class="badge ${r.status === 'PRESENT' ? 'badge-success' : 'badge-danger'}">${r.status}</span></td>
      </tr>
    `).join("");

    if (tbody1) tbody1.innerHTML = rowsHtml1;

    if (tbody2) {
      const rowsHtml2 = records.map(r => `
        <tr>
          <td class="font-bold">${r.date}</td>
          <td>${r.time_slot}</td>
          <td><span class="font-mono text-primary">${r.subject_code}</span> ${r.subject_name}</td>
          <td>${r.classroom}</td>
          <td>${r.faculty_name || 'Faculty Member'}</td>
          <td class="font-mono font-bold text-success">${r.entry_time || '--'}</td>
          <td class="font-mono">${r.similarity ? (r.similarity * 100).toFixed(1) + '%' : '--'}</td>
          <td><span class="badge badge-success">${r.liveness || 'VERIFIED'}</span></td>
          <td><span class="badge ${r.status === 'PRESENT' ? 'badge-success' : 'badge-danger'}">${r.status}</span></td>
        </tr>
      `).join("");
      tbody2.innerHTML = rowsHtml2;
    }
  } catch (err) {
    console.error("Error loading student attendance history:", err);
  }
}

// =============================================================
// STUDENT PROFILE DEDICATED VIEW LOADER
// =============================================================

async function loadStudentProfileView() {
  try {
    const res = await apiFetch("/api/student/profile");
    const data = await res.json();

    const setText = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.textContent = (val !== null && val !== undefined) ? val : "--";
    };

    setText("studentProfileHeaderName", data.name);
    setText("studentProfileHeaderRoll", `Roll: ${data.student_id}`);
    setText("studentProfileHeaderDept", `Department: ${data.department}`);

    const photoEl = document.getElementById("profilePagePhoto");
    if (photoEl) {
      photoEl.src = `${data.profile_photo_url}?t=${Date.now()}`;
    }

    setText("profilePageName", data.name);
    setText("profilePageRoll", data.student_id);
    setText("profilePageDeptBadge", data.department);
    setText("profilePageDept", data.department);
    setText("profilePageYear", `${data.year} Year`);
    setText("profilePageSection", `Section ${data.section}`);
    setText("profilePageAcdYear", data.academic_year || "2026-2027");
    setText("profilePageSemester", data.semester ? `Semester ${data.semester}` : "--");
    setText("profilePageEmail", data.email || "student@institution.edu");
    setText("profilePagePhone", data.phone || "--");

    setText("profileStatTotal", data.total_classes);
    setText("profileStatPresent", data.present_count);
    setText("profileStatAbsent", data.absent_count);
    setText("profileStatLate", data.late_count);

    const pct = `${data.attendance_percentage}%`;
    setText("profileProgressPct", pct);
    const pbar = document.getElementById("profileProgressBar");
    if (pbar) {
      pbar.style.width = `${Math.min(100, Math.max(0, data.attendance_percentage))}%`;
    }
  } catch (err) {
    console.error("Error loading student profile view:", err);
    showToast("Failed to load profile details", "danger");
  }
}

// Student Webcam Logic
async function startStudentCamera() {
  if (state.isStudentCameraRunning) return;
  const video = document.getElementById("studentWebcamVideo");
  const overlay = document.getElementById("studentCameraOverlay");
  const standby = document.getElementById("studentCameraStandby");

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showToast("Camera API not supported or requires secure context (localhost/HTTPS)", "danger");
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false,
    });
    state.studentWebcamStream = stream;
    video.srcObject = stream;
    state.isStudentCameraRunning = true;

    try {
      await video.play();
    } catch (e) {
      console.warn("video.play() autoplay handling:", e);
    }

    const initActiveCamera = () => {
      if (overlay) {
        overlay.width = video.videoWidth || 640;
        overlay.height = video.videoHeight || 480;
      }
      if (standby) {
        standby.style.display = "none";
        standby.classList.add("hidden");
      }
      const btnStart = document.getElementById("btnStudentStartCamera");
      const btnStop = document.getElementById("btnStudentStopCamera");
      if (btnStart) btnStart.style.display = "none";
      if (btnStop) btnStop.style.display = "inline-flex";
      state.lastFrameTime = performance.now();
      runStudentFrameProcessingLoop();
    };

    video.onloadedmetadata = initActiveCamera;
    if (video.videoWidth > 0) {
      initActiveCamera();
    }
  } catch (err) {
    console.error("Student camera access error:", err);
    showToast("Camera access failed: " + err.message, "danger");
  }
}

function stopStudentCamera() {
  if (!state.isStudentCameraRunning) return;
  state.isStudentCameraRunning = false;
  if (state.studentWebcamStream) {
    state.studentWebcamStream.getTracks().forEach(t => t.stop());
    state.studentWebcamStream = null;
  }
  const video = document.getElementById("studentWebcamVideo");
  if (video) video.srcObject = null;
  const standby = document.getElementById("studentCameraStandby");
  if (standby) {
    standby.style.display = "flex";
    standby.classList.remove("hidden");
  }
  const overlay = document.getElementById("studentCameraOverlay");
  if (overlay) {
    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, overlay.width, overlay.height);
  }
  const btnStart = document.getElementById("btnStudentStartCamera");
  const btnStop = document.getElementById("btnStudentStopCamera");
  if (btnStart) btnStart.style.display = "inline-flex";
  if (btnStop) btnStop.style.display = "none";
  const badge = document.getElementById("studentFaceCountBadge");
  if (badge) badge.textContent = "Detected Faces: 0";
}

async function runStudentFrameProcessingLoop() {
  if (!state.isStudentCameraRunning) return;
  const video = document.getElementById("studentWebcamVideo");
  const overlay = document.getElementById("studentCameraOverlay");
  if (!video || !overlay) return;
  const ctx = overlay.getContext("2d");

  const offCanvas = document.createElement("canvas");
  offCanvas.width = video.videoWidth || 640;
  offCanvas.height = video.videoHeight || 480;
  const offCtx = offCanvas.getContext("2d");
  offCtx.drawImage(video, 0, 0, offCanvas.width, offCanvas.height);

  const base64Data = offCanvas.toDataURL("image/jpeg", 0.85);

  try {
    const res = await apiFetch("/api/attendance/process-frame", {
      method: "POST",
      body: { image_base64: base64Data },
    });
    const data = await res.json();

    // FPS
    const now = performance.now();
    state.fps = Math.round(1000 / (now - state.lastFrameTime));
    state.lastFrameTime = now;
    document.getElementById("studentFpsBadge").textContent = `FPS: ${state.fps}`;
    document.getElementById("studentFaceCountBadge").textContent = `Detected Faces: ${data.total_faces}`;

    // Render Overlays (mirrored preview mode)
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    renderBoundingBoxes(ctx, data.detected_faces, true);

    // Update Result Card
    renderStudentResultCard(data.detected_faces, data.active_subject);

    // If any student marked attendance, refresh student dashboard
    const anyMarked = data.detected_faces.some(f => f.attendance_marked);
    if (anyMarked) {
      loadStudentDashboard();
    }
  } catch (err) {
    console.error("Student camera frame processing error:", err);
  }

  if (state.isStudentCameraRunning) {
    setTimeout(runStudentFrameProcessingLoop, 85);
  }
}

function renderStudentResultCard(faces, activeSubject) {
  const card = document.getElementById("studentResultCard");
  if (!card) return;

  if (!faces || faces.length === 0) {
    card.className = "result-status-card";
    card.innerHTML = `
      <div class="result-placeholder">
        <div class="result-icon-ph">👁️</div>
        <div class="result-text-ph">No faces detected. Position your face in front of the camera.</div>
      </div>
    `;
    return;
  }

  const markedFace = faces.find(f => f.attendance_marked);
  const alreadyMarkedFace = faces.find(f => f.already_marked && f.matched);
  const verifyingFace = faces.find(f => f.status === "VERIFYING");
  const invalidFace = faces.find(f => f.matched === false || f.student_id === "UNKNOWN");
  const livenessFailed = faces.find(f => f.liveness_status === "SPOOF" || f.liveness_status === "FAKE");

  if (markedFace) {
    card.className = "result-status-card result-card-success";
    card.innerHTML = `
      <div class="result-status-title text-success">
        <span>✓</span> <span>ATTENDANCE MARKED</span>
      </div>
      <div class="result-details-grid">
        <div class="row"><span class="lbl">Name:</span> <span class="val font-bold">${markedFace.student_name}</span></div>
        <div class="row"><span class="lbl">Roll No:</span> <span class="val font-mono">${markedFace.student_id}</span></div>
        <div class="row"><span class="lbl">Time:</span> <span class="val font-bold">${markedFace.first_seen || new Date().toLocaleTimeString()}</span></div>
        <div class="row"><span class="lbl">Subject:</span> <span class="val">${activeSubject || 'Active Class'}</span></div>
        <div class="row"><span class="lbl">Confidence:</span> <span class="val text-success font-bold">${(markedFace.confidence * 100).toFixed(1)}%</span></div>
        <div class="row"><span class="lbl">Status:</span> <span class="badge badge-success">PRESENT</span></div>
      </div>
    `;
    showToast(`✓ Attendance confirmed for ${markedFace.student_name} (${markedFace.student_id})`, "success");
  } else if (alreadyMarkedFace) {
    card.className = "result-status-card result-card-info";
    card.innerHTML = `
      <div class="result-status-title text-info">
        <span>ℹ</span> <span>ATTENDANCE ALREADY MARKED</span>
      </div>
      <div class="result-details-grid">
        <div class="row"><span class="lbl">Name:</span> <span class="val font-bold">${alreadyMarkedFace.student_name}</span></div>
        <div class="row"><span class="lbl">Roll No:</span> <span class="val font-mono">${alreadyMarkedFace.student_id}</span></div>
        <div class="row"><span class="lbl">Entry Recorded At:</span> <span class="val">${alreadyMarkedFace.first_seen || 'Earlier this session'}</span></div>
        <div class="row"><span class="lbl">Status:</span> <span class="badge badge-success">PRESENT</span></div>
      </div>
    `;
  } else if (verifyingFace) {
    card.className = "result-status-card result-card-info";
    card.innerHTML = `
      <div class="result-status-title text-info">
        <span>🔍</span> <span>VERIFYING IDENTITY</span>
      </div>
      <div class="result-details-grid">
        <div class="row"><span class="lbl">Candidate:</span> <span class="val font-bold">${verifyingFace.student_name}</span></div>
        <div class="row"><span class="lbl">Roll No:</span> <span class="val font-mono">${verifyingFace.student_id}</span></div>
        <div class="row"><span class="lbl">Progress:</span> <span class="val">${verifyingFace.message}</span></div>
        <div class="row"><span class="lbl">Liveness:</span> <span class="badge badge-live">ANALYZING</span></div>
      </div>
    `;
  } else if (livenessFailed) {
    card.className = "result-status-card result-card-danger";
    card.innerHTML = `
      <div class="result-status-title text-danger">
        <span>⚠</span> <span>LIVENESS VERIFICATION FAILED</span>
      </div>
      <div class="result-details-grid">
        <div class="row"><span class="lbl">Notice:</span> <span class="val">Biometric anti-spoofing check failed.</span></div>
        <div class="row"><span class="lbl">Guidance:</span> <span class="val">Blink naturally and face camera directly.</span></div>
      </div>
    `;
  } else if (invalidFace) {
    card.className = "result-status-card result-card-warning";
    card.innerHTML = `
      <div class="result-status-title text-warning">
        <span>⚠</span> <span>UNKNOWN STUDENT</span>
      </div>
      <div class="result-details-grid">
        <div class="row"><span class="lbl">Notice:</span> <span class="val">Face not found in enrolled biometric gallery.</span></div>
        <div class="row"><span class="lbl">Status:</span> <span class="badge badge-danger">NOT MARKED</span></div>
      </div>
    `;
  }
}

// =============================================================
// STAFF DASHBOARD CONTROLLERS
// =============================================================

async function loadStaffDashboard() {
  try {
    const res = await apiFetch("/api/staff/session");
    const data = await res.json();

    document.getElementById("staffBannerSubjectName").textContent = data.has_active_session
      ? `${data.subject_code} - ${data.subject_name}`
      : (data.message || "No Active Class Session");

    document.getElementById("staffBannerDeptYear").textContent = `Department: ${data.department || '--'} ${data.year || ''}-${data.section || ''}`;
    document.getElementById("staffBannerSlot").textContent = `Time Slot: ${data.start_time || '--'} - ${data.end_time || '--'}`;
    document.getElementById("staffBannerRoom").textContent = `Classroom: ${data.classroom || '--'}`;

    const statusBadge = document.getElementById("staffBannerStatus");
    if (data.has_active_session) {
      statusBadge.textContent = "ACTIVE";
      statusBadge.className = "badge badge-success";
    } else {
      statusBadge.textContent = (data.status || "UPCOMING").toUpperCase();
      statusBadge.className = "badge badge-warning";
    }

    document.getElementById("staffStatTotalStudents").textContent = data.total_students;
    document.getElementById("staffStatPresent").textContent = data.present_count;
    document.getElementById("staffStatAbsent").textContent = data.absent_count;
    document.getElementById("staffStatLate").textContent = data.late_count;
    document.getElementById("staffStatPercentage").textContent = `${data.attendance_percentage}%`;

    const btnActivate = document.getElementById("btnStaffActivateSlot");
    if (data.timetable_id && !data.has_active_session) {
      btnActivate.style.display = "inline-flex";
      btnActivate.onclick = () => activateStaffSession(data.timetable_id);
    } else {
      btnActivate.style.display = "none";
    }

    if (data.has_active_session && data.session_id) {
      loadStaffSessionAttendance(data.session_id);
      loadStaffClassStudents(data.session_id);

      // Start live sync polling every 2.5s for real-time updates without manual refresh
      if (state.staffPollingTimer) clearInterval(state.staffPollingTimer);
      state.staffPollingTimer = setInterval(() => {
        if (state.currentView === "view-staff-dashboard") {
          loadStaffSessionAttendance(data.session_id);
        }
      }, 2500);
    } else {
      document.getElementById("staffLiveAttendanceBody").innerHTML =
        '<tr><td colspan="7" class="text-center py-4">No active session currently in progress.</td></tr>';
      document.getElementById("staffClassStudentsBody").innerHTML =
        '<tr><td colspan="7" class="text-center py-4">Activate a session to display class roster.</td></tr>';
    }
  } catch (err) {
    console.error("Load staff dashboard error:", err);
  }
}

async function activateStaffSession(timetableId) {
  try {
    const res = await apiFetch(`/api/staff/session/activate/${timetableId}`, { method: "POST" });
    const data = await res.json();
    if (res.ok) {
      showToast("Class session activated successfully!", "success");
      loadStaffDashboard();
    } else {
      showToast(data.detail || "Failed to activate session", "danger");
    }
  } catch (err) {
    showToast("Error activating session: " + err.message, "danger");
  }
}

async function loadStaffSessionAttendance(sessionId) {
  try {
    const res = await apiFetch(`/api/sessions/${sessionId}/attendance`);
    const records = await res.json();

    const tbody = document.getElementById("staffLiveAttendanceBody");
    if (!tbody) return;

    const presentRecords = records.filter(r => r.status === "PRESENT");

    // Live update stat metrics
    document.getElementById("staffStatPresent").textContent = presentRecords.length;
    const total = parseInt(document.getElementById("staffStatTotalStudents").textContent) || presentRecords.length;
    const absent = Math.max(0, total - presentRecords.length);
    document.getElementById("staffStatAbsent").textContent = absent;
    document.getElementById("staffStatPercentage").textContent = `${Math.round((presentRecords.length / Math.max(1, total)) * 100)}%`;

    if (presentRecords.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7" class="text-center py-4">Waiting for students to mark attendance...</td></tr>';
      return;
    }

    tbody.innerHTML = presentRecords.map(r => `
      <tr>
        <td>
          <img src="/api/students/${r.student_id}/photo" class="student-ref-thumb" alt="${r.student_name}">
        </td>
        <td class="font-mono font-bold">${r.student_id}</td>
        <td class="font-bold">${r.student_name}</td>
        <td class="font-mono text-success font-bold">${r.first_seen || r.last_seen || '--'}</td>
        <td class="font-mono">${r.similarity_score ? (r.similarity_score * 100).toFixed(1) + '%' : '--'}</td>
        <td><span class="badge badge-success">${r.liveness_status || 'VERIFIED'}</span></td>
        <td><span class="badge badge-success">PRESENT</span></td>
      </tr>
    `).join("");
  } catch (err) {
    console.error("Load staff live attendance error:", err);
  }
}

async function loadStaffClassStudents(sessionId) {
  try {
    const res = await apiFetch(`/api/staff/session/${sessionId}/students`);
    const students = await res.json();
    const tbody = document.getElementById("staffClassStudentsBody");
    if (!tbody) return;

    if (students.length === 0) {
      tbody.innerHTML = '<tr><td colspan="8" class="text-center py-4">No enrolled students found for this class.</td></tr>';
      return;
    }

    tbody.innerHTML = students.map(s => {
      const isPresent = s.status === "PRESENT";
      return `
        <tr>
          <td>
            <img src="${s.photo_url}" class="student-ref-thumb" alt="${s.name}" onerror="this.src='/api/students/${s.student_id}/photo';">
          </td>
          <td class="font-mono font-bold">${s.student_id}</td>
          <td class="font-bold">${s.name}</td>
          <td>${s.department} ${s.year}-${s.section}</td>
          <td class="font-mono">${s.entry_time || '--'}</td>
          <td>${isPresent ? `<span class="text-success font-bold">Verified (${s.similarity ? (s.similarity * 100).toFixed(0) + '%' : 'ArcFace'})</span>` : '<span class="text-muted">Not Yet Scanned</span>'}</td>
          <td><span class="badge ${isPresent ? 'badge-success' : 'badge-danger'}">${s.status}</span></td>
          <td>
            <div style="display:flex; gap:6px;">
              <button class="btn btn-outline btn-sm" onclick="openViewStudentModal('${s.student_id}')" title="View Profile">👁️</button>
              <button class="btn btn-outline btn-sm" onclick="openEditStudentModal('${s.student_id}')" title="Edit Student">✏️</button>
              <button class="btn btn-danger btn-sm" onclick="deleteStudent('${s.student_id}')" title="Delete Student">🗑️</button>
            </div>
          </td>
        </tr>
      `;
    }).join("");
  } catch (err) {
    console.error("Load staff class students error:", err);
  }
}

async function loadLiveAttendanceList(sessionId) {
  try {
    const res = await apiFetch(`/api/sessions/${sessionId}/attendance`);
    const records = await res.json();
    const tbody = document.getElementById("liveAttendanceTableBody");

    if (records.length === 0) {
      tbody.innerHTML = `<tr><td colspan="4" style="text-align:center;">No students enrolled for this session.</td></tr>`;
      return;
    }

    tbody.innerHTML = records
      .map(
        (r) => `
      <tr>
        <td>
          <div style="font-weight:600;">${r.student_name}</div>
          <div style="font-size:10px; color:var(--text-subtle);">${r.student_id}</div>
        </td>
        <td>${r.first_seen || "--"}</td>
        <td>${r.last_seen || "--"}</td>
        <td>
          <span class="badge ${r.status === "PRESENT" ? "badge-present" : "badge-absent"}">
            ${r.status}
          </span>
        </td>
      </tr>
    `
      )
      .join("");
  } catch (err) {
    console.error("Failed to load live attendance list:", err);
  }
}

// -------------------------------------------------------------
// STUDENT MULTI-ANGLE ENROLLMENT STUDIO
// -------------------------------------------------------------

// Web Audio API Confirmation Chime
function playConfirmationChime() {
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    const ctx = new AudioCtx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sine";
    // Pleasant two-tone chime (F5 -> A5)
    osc.frequency.setValueAtTime(698.46, ctx.currentTime);
    osc.frequency.setValueAtTime(880.00, ctx.currentTime + 0.12);
    gain.gain.setValueAtTime(0.18, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.35);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.35);
  } catch (err) {
    console.debug("AudioContext prevented or not supported:", err);
  }
}

function getAngleCardId(angleId) {
  const map = {
    front: "cardAngleFront",
    slight_left: "cardAngleLeft",
    slight_right: "cardAngleRight",
    slight_up: "cardAngleUp",
    slight_down: "cardAngleDown",
  };
  return map[angleId] || `cardAngle${capitalize(angleId)}`;
}

async function startEnrollCamera() {
  const video = document.getElementById("enrollWebcamVideo");
  const btnToggle = document.getElementById("btnToggleEnrollCamera");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
    });
    state.enrollStream = stream;
    video.srcObject = stream;
    await video.play();
    state.isEnrollCameraRunning = true;
    state.holdDurationMs = 0;
    state.lastPoseTime = performance.now();

    if (btnToggle) {
      btnToggle.textContent = "Turn Off Camera ⏹️";
      btnToggle.classList.remove("btn-outline");
      btnToggle.classList.add("btn-danger");
    }

    const ringText = document.getElementById("guideRingText");
    if (ringText) ringText.textContent = "Position Face in Ring";
    const ringIcon = document.getElementById("guideRingIcon");
    if (ringIcon) ringIcon.textContent = "🎯";

    showToast("Enrollment Camera Ready. Align face with the guide ring.", "success");
    runEnrollPoseLoop();
  } catch (err) {
    console.error("Enrollment camera error:", err);
    showToast("Enrollment camera error: " + err.message, "danger");
  }
}

function stopEnrollCamera() {
  state.isEnrollCameraRunning = false;
  state.holdDurationMs = 0;
  if (state.enrollStream) {
    state.enrollStream.getTracks().forEach((track) => track.stop());
    state.enrollStream = null;
  }
  const video = document.getElementById("enrollWebcamVideo");
  if (video) video.srcObject = null;

  const overlay = document.getElementById("enrollOverlayCanvas");
  if (overlay) {
    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, overlay.width, overlay.height);
  }

  const btnToggle = document.getElementById("btnToggleEnrollCamera");
  if (btnToggle) {
    btnToggle.textContent = "Turn On Camera 🎥";
    btnToggle.classList.remove("btn-danger");
    btnToggle.classList.add("btn-outline");
  }

  const ringText = document.getElementById("guideRingText");
  if (ringText) ringText.textContent = "Turn On Camera";
  const ringIcon = document.getElementById("guideRingIcon");
  if (ringIcon) ringIcon.textContent = "⏳";

  const guideRing = document.getElementById("guideRingState");
  if (guideRing) guideRing.classList.remove("matched");

  const holdBadge = document.getElementById("enrollHoldBadge");
  if (holdBadge) {
    holdBadge.textContent = "Hold Still: 0ms / 500ms";
    holdBadge.className = "badge badge-verifying";
  }
}

function toggleEnrollCamera() {
  if (state.isEnrollCameraRunning) {
    stopEnrollCamera();
    showToast("Enrollment Camera turned off.", "info");
  } else {
    startEnrollCamera();
  }
}

function drawEnrollGuideRing(ctx, width, height, isMatched, holdPct) {
  ctx.clearRect(0, 0, width, height);

  const cx = width / 2;
  const cy = height * 0.48;
  const rx = width * 0.23;
  const ry = height * 0.35;

  ctx.save();
  // Guide Ring Outline
  ctx.lineWidth = isMatched ? 4 : 2.5;
  ctx.strokeStyle = isMatched ? "#10b981" : "rgba(6, 182, 212, 0.6)";
  if (isMatched) {
    ctx.shadowColor = "#10b981";
    ctx.shadowBlur = 18;
  } else {
    ctx.setLineDash([8, 8]);
  }

  ctx.beginPath();
  ctx.ellipse(cx, cy, rx, ry, 0, 0, 2 * Math.PI);
  ctx.stroke();

  // Circular progress arc along the ring indicating 500ms hold progress
  if (isMatched && holdPct > 0) {
    ctx.setLineDash([]);
    ctx.lineWidth = 6;
    ctx.strokeStyle = "#34d399";
    ctx.shadowColor = "#34d399";
    ctx.shadowBlur = 12;
    ctx.beginPath();
    const startAngle = -Math.PI / 2;
    const endAngle = startAngle + (Math.PI * 2 * Math.min(1, holdPct));
    ctx.ellipse(cx, cy, rx, ry, 0, startAngle, endAngle);
    ctx.stroke();
  }

  // Crosshair corner marks
  ctx.setLineDash([]);
  ctx.lineWidth = 1;
  ctx.strokeStyle = isMatched ? "rgba(16, 185, 129, 0.5)" : "rgba(6, 182, 212, 0.35)";
  ctx.beginPath();
  ctx.moveTo(cx, cy - ry - 12); ctx.lineTo(cx, cy - ry + 12);
  ctx.moveTo(cx, cy + ry - 12); ctx.lineTo(cx, cy + ry + 12);
  ctx.moveTo(cx - rx - 12, cy); ctx.lineTo(cx - rx + 12, cy);
  ctx.moveTo(cx + rx - 12, cy); ctx.lineTo(cx + rx + 12, cy);
  ctx.stroke();

  ctx.restore();
}

async function runEnrollPoseLoop() {
  if (!state.isEnrollCameraRunning) return;

  const video = document.getElementById("enrollWebcamVideo");
  const overlay = document.getElementById("enrollOverlayCanvas");
  if (!video || !video.videoWidth) {
    if (state.isEnrollCameraRunning) {
      setTimeout(runEnrollPoseLoop, 100);
    }
    return;
  }

  if (overlay.width !== video.videoWidth || overlay.height !== video.videoHeight) {
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
  }
  const ctx = overlay.getContext("2d");

  // Downsample to 320x240 for fast pose inference
  const offCanvas = document.createElement("canvas");
  offCanvas.width = 320;
  offCanvas.height = 240;
  const offCtx = offCanvas.getContext("2d");
  offCtx.drawImage(video, 0, 0, 320, 240);
  const dataUrl = offCanvas.toDataURL("image/jpeg", 0.75);

  const now = performance.now();
  const dt = state.lastPoseTime ? (now - state.lastPoseTime) : 150;
  state.lastPoseTime = now;

  let isMatched = false;
  try {
    const res = await apiFetch("/api/students/evaluate-pose", {
      method: "POST",
      body: {
        image_base64: dataUrl,
        target_angle: state.activeEnrollAngle,
      },
    });

    if (res.ok) {
      const data = await res.json();
      const yawEl = document.getElementById("poseYawVal");
      const pitchEl = document.getElementById("posePitchVal");
      const sharpEl = document.getElementById("poseSharpVal");
      const fqaStatus = document.getElementById("enrollFqaStatus");

      if (yawEl) yawEl.textContent = `${data.yaw > 0 ? "+" : ""}${data.yaw.toFixed(1)}°`;
      if (pitchEl) pitchEl.textContent = `${data.pitch > 0 ? "+" : ""}${data.pitch.toFixed(1)}°`;
      if (sharpEl) sharpEl.textContent = Math.round(data.sharpness);
      if (fqaStatus) fqaStatus.textContent = data.feedback;

      isMatched = data.angle_matched && data.face_detected;

      const guideRing = document.getElementById("guideRingState");
      const ringText = document.getElementById("guideRingText");
      const ringIcon = document.getElementById("guideRingIcon");
      const holdBadge = document.getElementById("enrollHoldBadge");

      if (isMatched) {
        state.holdDurationMs += dt;
        const progress = Math.min(1, state.holdDurationMs / 500);

        if (guideRing) guideRing.classList.add("matched");
        if (ringIcon) ringIcon.textContent = "✅";
        if (ringText) ringText.textContent = "Hold Still... Steady!";

        if (holdBadge) {
          holdBadge.textContent = `Hold Still: ${Math.round(progress * 500)}ms / 500ms`;
          holdBadge.className = "badge badge-present";
        }

        drawEnrollGuideRing(ctx, overlay.width, overlay.height, true, progress);

        if (state.holdDurationMs >= 500) {
          // Trigger Auto-Capture with Audio Chime
          playConfirmationChime();
          captureActiveEnrollAngle();
          state.holdDurationMs = 0;
          if (guideRing) guideRing.classList.remove("matched");

          // Brief cooldown so user can orient for next angle
          if (state.isEnrollCameraRunning) {
            setTimeout(runEnrollPoseLoop, 850);
            return;
          }
        }
      } else {
        state.holdDurationMs = 0;
        if (guideRing) guideRing.classList.remove("matched");
        if (ringIcon) ringIcon.textContent = data.face_detected ? "🎯" : "👤";
        if (ringText) ringText.textContent = data.face_detected ? "Align with Directive" : "No Face Detected";

        if (holdBadge) {
          holdBadge.textContent = "Hold Still: 0ms / 500ms";
          holdBadge.className = "badge badge-verifying";
        }

        drawEnrollGuideRing(ctx, overlay.width, overlay.height, false, 0);
      }
    }
  } catch (err) {
    console.debug("Pose loop error:", err);
    drawEnrollGuideRing(ctx, overlay.width, overlay.height, false, 0);
  }

  if (state.isEnrollCameraRunning) {
    setTimeout(runEnrollPoseLoop, 140);
  }
}

function selectEnrollAngle(angleId) {
  state.activeEnrollAngle = angleId;
  state.holdDurationMs = 0;
  document.querySelectorAll(".angle-card").forEach((card) => card.classList.remove("active"));
  const cardId = getAngleCardId(angleId);
  const activeCard = document.getElementById(cardId);
  if (activeCard) activeCard.classList.add("active");

  const angleObj = ENROLL_ANGLES.find((a) => a.id === angleId);
  if (angleObj) {
    document.getElementById("enrollFeedbackTitle").textContent = `Position Face: ${angleObj.name}`;
    document.getElementById("enrollFeedbackDesc").textContent = `${angleObj.desc}. Keep head steady for 500ms for automatic capture.`;
  }
}

function captureActiveEnrollAngle() {
  const video = document.getElementById("enrollWebcamVideo");
  if (!video || !video.videoWidth) {
    showToast("Please start camera before capturing.", "warning");
    return;
  }

  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const dataUrl = canvas.toDataURL("image/jpeg", 0.95);
  const angle = state.activeEnrollAngle;
  state.enrollSamples[angle] = dataUrl;

  // Visual confirmation on card
  const cardId = getAngleCardId(angle);
  const card = document.getElementById(cardId);
  if (card) {
    card.classList.add("captured");
    const fill = card.querySelector(".quality-fill");
    if (fill) fill.style.width = "100%";
  }

  const angleObj = ENROLL_ANGLES.find((a) => a.id === angle);
  showToast(`Captured: ${angleObj ? angleObj.name : angle}!`, "success");

  // Advance to next uncaptured angle automatically
  const uncaptured = ENROLL_ANGLES.find((a) => !state.enrollSamples[a.id]);
  if (uncaptured) {
    selectEnrollAngle(uncaptured.id);
  } else {
    showToast("All 5 face angles captured! Click 'Save & Enroll Biometrics'.", "info");
  }
}

async function submitStudentEnrollment() {
  const studentId = document.getElementById("enrollStudentId").value.trim();
  const name = document.getElementById("enrollName").value.trim();
  const department = document.getElementById("enrollDept").value;
  const year = document.getElementById("enrollYear").value;
  const section = document.getElementById("enrollSection").value;
  const email = document.getElementById("enrollEmail").value.trim();

  if (!studentId || !name) {
    showToast("Please enter Student ID and Name.", "warning");
    document.getElementById("enrollStudentId").focus();
    return;
  }

  const capturedCount = Object.keys(state.enrollSamples).length;
  if (capturedCount === 0) {
    showToast("Please capture at least 1 face angle sample (front).", "warning");
    return;
  }

  try {
    // 1. Create or ensure student record
    const studentRes = await apiFetch("/api/students", {
      method: "POST",
      body: { student_id: studentId, name, department, year, section, email },
    });

    if (!studentRes.ok) {
      const errJson = await studentRes.json();
      if (!errJson.detail || !errJson.detail.includes("already registered")) {
        showToast(errJson.detail || "Failed to create student.", "danger");
        return;
      }
    }

    // 2. Upload face angles
    const samplesPayload = Object.entries(state.enrollSamples).map(([angle, imgBase64]) => ({
      angle,
      image_base64: imgBase64,
    }));

    const enrollRes = await apiFetch(`/api/students/${studentId}/enroll-faces`, {
      method: "POST",
      body: { samples: samplesPayload },
    });

    const enrollData = await enrollRes.json();
    if (enrollRes.ok) {
      showToast(`Student ${name} successfully enrolled with ${enrollData.enrolled_count} samples!`, "success");
      // Reset form
      document.getElementById("enrollStudentForm").reset();
      state.enrollSamples = {};
      document.querySelectorAll(".angle-card").forEach((c) => {
        c.classList.remove("captured");
        const fill = c.querySelector(".quality-fill");
        if (fill) fill.style.width = "0%";
      });
      selectEnrollAngle("front");
      loadStudentsList();
    } else {
      showToast(enrollData.detail || "Enrollment rejected by FQA.", "danger");
    }
  } catch (err) {
    showToast("Enrollment error: " + err.message, "danger");
  }
}

// -------------------------------------------------------------
// ENROLLMENT MODE SWITCHER & FILE UPLOAD
// -------------------------------------------------------------
function setupEnrollmentModeSwitcher() {
  const tabLive = document.getElementById("tabLiveScan");
  const tabUpload = document.getElementById("tabFileUpload");
  const panelLive = document.getElementById("panelLiveScan");
  const panelUpload = document.getElementById("panelFileUpload");

  if (!tabLive || !tabUpload || !panelLive || !panelUpload) return;

  tabLive.addEventListener("click", () => {
    tabLive.classList.add("active");
    tabUpload.classList.remove("active");
    panelLive.style.display = "block";
    panelUpload.style.display = "none";
  });

  tabUpload.addEventListener("click", () => {
    tabUpload.classList.add("active");
    tabLive.classList.remove("active");
    panelLive.style.display = "none";
    panelUpload.style.display = "block";
    // Release live camera tracks immediately on tab switch
    stopEnrollCamera();
  });
}

function setupFileUploadDropzone() {
  const dropzone = document.getElementById("fileDropzone");
  const fileInput = document.getElementById("fileInput");
  const dropzonePrompt = document.getElementById("dropzonePrompt");
  const dropzonePreview = document.getElementById("dropzonePreview");
  const previewImg = document.getElementById("uploadPreviewImg");
  const previewFileName = document.getElementById("previewFileName");
  const previewFileSize = document.getElementById("previewFileSize");
  const btnRemoveFile = document.getElementById("btnRemoveFile");
  const btnUploadAndEnroll = document.getElementById("btnUploadAndEnroll");

  if (!dropzone || !fileInput) return;

  dropzone.addEventListener("click", (e) => {
    if (e.target.closest("#btnRemoveFile")) return;
    fileInput.click();
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  btnRemoveFile.addEventListener("click", (e) => {
    e.stopPropagation();
    clearUploadedFile();
  });

  btnUploadAndEnroll.addEventListener("click", handleUploadAndEnroll);

  function handleFileSelected(file) {
    const validTypes = ["image/jpeg", "image/png", "image/jpg"];
    if (!validTypes.includes(file.type)) {
      showToast("Invalid file type. Please select a JPG or PNG image.", "warning");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      showToast("File is too large. Maximum allowed size is 10MB.", "warning");
      return;
    }

    state.uploadedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      previewFileName.textContent = file.name;
      previewFileSize.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
      dropzonePrompt.style.display = "none";
      dropzonePreview.style.display = "flex";
      btnUploadAndEnroll.removeAttribute("disabled");
    };
    reader.readAsDataURL(file);
  }

  function clearUploadedFile() {
    state.uploadedFile = null;
    fileInput.value = "";
    previewImg.src = "";
    dropzonePrompt.style.display = "block";
    dropzonePreview.style.display = "none";
    btnUploadAndEnroll.setAttribute("disabled", "true");
  }
}

async function handleUploadAndEnroll() {
  const studentId = document.getElementById("enrollStudentId").value.trim();
  const name = document.getElementById("enrollName").value.trim();
  const department = document.getElementById("enrollDept").value;
  const year = document.getElementById("enrollYear").value;
  const section = document.getElementById("enrollSection").value;
  const email = document.getElementById("enrollEmail").value.trim();

  if (!studentId || !name) {
    showToast("Please enter Student Roll / ID and Full Name first.", "warning");
    document.getElementById("enrollStudentId").focus();
    return;
  }

  if (!state.uploadedFile) {
    showToast("Please select a student portrait photo first.", "warning");
    return;
  }

  const btnUpload = document.getElementById("btnUploadAndEnroll");
  btnUpload.disabled = true;
  btnUpload.innerHTML = `<span>⏳</span> Validating & Extracting Embeddings...`;

  try {
    // 1. Ensure student record exists in database
    const studentRes = await apiFetch("/api/students", {
      method: "POST",
      body: { student_id: studentId, name, department, year, section, email },
    });

    if (!studentRes.ok) {
      const errJson = await studentRes.json();
      if (!errJson.detail || !errJson.detail.includes("already registered")) {
        showToast(errJson.detail || "Failed to create student record.", "danger");
        btnUpload.disabled = false;
        btnUpload.innerHTML = `<span>🚀</span> Verify & Enroll Photo`;
        return;
      }
    }

    // 2. Upload photo for validation and multi-angle auto-augmentation
    const formData = new FormData();
    formData.append("file", state.uploadedFile);

    const uploadRes = await apiFetch(`/api/students/${studentId}/upload-photo`, {
      method: "POST",
      body: formData,
    });

    const uploadData = await uploadRes.json();
    if (uploadRes.ok) {
      showToast(uploadData.message || `Photo validated! 5 biometric angles generated for ${name}.`, "success");
      // Reset form and file
      document.getElementById("enrollStudentForm").reset();
      const dropzonePrompt = document.getElementById("dropzonePrompt");
      const dropzonePreview = document.getElementById("dropzonePreview");
      const fileInput = document.getElementById("fileInput");
      state.uploadedFile = null;
      if (fileInput) fileInput.value = "";
      if (dropzonePrompt) dropzonePrompt.style.display = "block";
      if (dropzonePreview) dropzonePreview.style.display = "none";
      loadStudentsList();
    } else {
      showToast(uploadData.detail || "Photo rejected by biometric validation.", "danger");
    }
  } catch (err) {
    console.error("Upload enroll error:", err);
    showToast("Upload error: " + err.message, "danger");
  } finally {
    btnUpload.disabled = false;
    btnUpload.innerHTML = `<span>🚀</span> Verify & Enroll Photo`;
  }
}

// -------------------------------------------------------------
// STUDENT DIRECTORY & PROFILE MANAGEMENT
// -------------------------------------------------------------
async function loadStudentsList() {
  const dept = document.getElementById("filterStudentDept")?.value || "";
  try {
    const url = dept ? `/api/students?department=${encodeURIComponent(dept)}` : "/api/students";
    const res = await apiFetch(url);
    const students = await res.json();
    const tbody = document.getElementById("studentsTableBody");
    if (!tbody) return;

    if (students.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;">No students found. Enroll new students in Face Enrollment Studio.</td></tr>`;
      return;
    }

    tbody.innerHTML = students
      .map(
        (s) => `
      <tr>
        <td>
          <img src="/api/students/${s.student_id}/photo" class="student-ref-thumb" alt="${s.name}" onerror="this.onerror=null;this.src='data:image/svg+xml,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'42\\' height=\\'42\\' viewBox=\\'0 0 24 24\\' fill=\\'%2364748b\\'><path d=\\'M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z\\'/></svg>';">
        </td>
        <td class="font-mono font-bold">${s.student_id}</td>
        <td class="font-bold">${s.name}</td>
        <td>${s.department}</td>
        <td>${s.year}-${s.section}</td>
        <td>${s.email || "--"}</td>
        <td><span class="badge badge-present">${s.face_sample_count || 0} Samples</span></td>
        <td>
          <div style="display:flex; gap:6px;">
            <button class="btn btn-outline btn-sm" onclick="openViewStudentModal('${s.student_id}')" title="View Biometric Profile">👁️</button>
            <button class="btn btn-outline btn-sm" onclick="openEditStudentModal('${s.student_id}')" title="Edit Student">✏️</button>
            <button class="btn btn-danger btn-sm" onclick="deleteStudent('${s.student_id}')" title="Delete Student">🗑️</button>
          </div>
        </td>
      </tr>
    `
      )
      .join("");
  } catch (err) {
    console.error("Failed to load students:", err);
  }
}

async function openViewStudentModal(studentId) {
  try {
    const res = await apiFetch(`/api/students/${studentId}`);
    if (!res.ok) {
      showToast("Student not found", "danger");
      return;
    }
    const s = await res.json();

    const photoEl = document.getElementById("viewStudentPhoto");
    if (photoEl) photoEl.src = `/api/students/${s.student_id}/photo?t=${Date.now()}`;
    document.getElementById("viewStudentName").textContent = s.name;
    document.getElementById("viewStudentRoll").textContent = s.student_id;
    document.getElementById("viewStudentDept").textContent = s.department;
    document.getElementById("viewStudentClass").textContent = `${s.year} Year, Sec ${s.section}`;
    document.getElementById("viewStudentEmail").textContent = s.email || "--";
    document.getElementById("viewStudentPhone").textContent = s.phone || "--";
    document.getElementById("viewStudentSamples").textContent = `${s.face_sample_count || 0} Enrolled Samples`;

    const badge = document.getElementById("viewStudentStatusBadge");
    if (badge) {
      badge.textContent = s.is_active ? "Enrolled & Active" : "Deactivated";
      badge.className = `badge ${s.is_active ? 'badge-success' : 'badge-danger'}`;
    }

    openModal("modalViewStudent");
  } catch (err) {
    showToast("Error loading student profile: " + err.message, "danger");
  }
}

async function openEditStudentModal(studentId) {
  try {
    const res = await apiFetch(`/api/students/${studentId}`);
    if (!res.ok) {
      showToast("Student not found", "danger");
      return;
    }
    const s = await res.json();

    document.getElementById("editStudentIdHidden").value = s.student_id;
    document.getElementById("editModalName").value = s.name;
    document.getElementById("editModalDept").value = s.department;
    document.getElementById("editModalYear").value = s.year;
    document.getElementById("editModalSection").value = s.section;
    document.getElementById("editModalEmail").value = s.email || "";
    document.getElementById("editModalPhone").value = s.phone || "";

    openModal("modalEditStudent");
  } catch (err) {
    showToast("Error loading student for edit: " + err.message, "danger");
  }
}

async function handleSaveEditStudent(e) {
  e.preventDefault();
  const studentId = document.getElementById("editStudentIdHidden").value;
  const name = document.getElementById("editModalName").value.trim();
  const department = document.getElementById("editModalDept").value;
  const year = document.getElementById("editModalYear").value;
  const section = document.getElementById("editModalSection").value;
  const email = document.getElementById("editModalEmail").value.trim();
  const phone = document.getElementById("editModalPhone").value.trim();

  try {
    const res = await apiFetch(`/api/students/${studentId}`, {
      method: "PUT",
      body: { name, department, year, section, email, phone }
    });
    if (res.ok) {
      showToast(`Student ${studentId} updated successfully.`, "success");
      closeModal("modalEditStudent");
      loadStudentsList();
      if (state.currentView === "view-staff-dashboard") {
        loadStaffDashboard();
      }
    } else {
      const err = await res.json();
      showToast(err.detail || "Failed to update student", "danger");
    }
  } catch (err) {
    showToast("Update error: " + err.message, "danger");
  }
}

async function deleteStudent(studentId) {
  if (!confirm(`Are you sure you want to delete student ${studentId}? This will permanently remove their profile and enrolled face biometrics.`)) return;
  try {
    const res = await apiFetch(`/api/students/${studentId}`, { method: "DELETE" });
    if (res.ok) {
      const data = await res.json().catch(() => ({}));
      showToast(data.message || `Student ${studentId} deleted successfully.`, "success");
      loadStudentsList();
      if (state.currentView === "view-staff-dashboard" && typeof loadStaffDashboard === "function") {
        loadStaffDashboard();
      }
    } else {
      const err = await res.json().catch(() => ({}));
      showToast(err.detail || err.message || `Failed to delete student ${studentId}.`, "danger");
    }
  } catch (err) {
    showToast(`Error deleting student: ${err.message}`, "danger");
  }
}

// -------------------------------------------------------------
// TIMETABLE MATRIX & SCHEDULING
// -------------------------------------------------------------
async function loadTimetable() {
  try {
    const res = await apiFetch("/api/timetable");
    const slots = await res.json();
    const tbody = document.getElementById("timetableTableBody");

    if (slots.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;">No timetable slots configured. Add one using the button above.</td></tr>`;
      return;
    }

    tbody.innerHTML = slots
      .map(
        (s) => `
      <tr>
        <td><strong>${s.day_of_week}</strong></td>
        <td>${s.start_time} - ${s.end_time}</td>
        <td>${s.department}</td>
        <td>${s.year}-${s.section}</td>
        <td>${s.subject_name || "Subject " + s.subject_id}</td>
        <td>${s.faculty_name || "Faculty Assigned"}</td>
        <td>${s.classroom}</td>
        <td>
          <div style="display:flex; gap:6px;">
            <button class="btn btn-success btn-sm" onclick="forceActivateSlot(${s.id})" title="Immediately trigger attendance session for this slot">
              ⚡ Activate
            </button>
            <button class="btn btn-outline btn-sm" onclick="deleteTimetableSlot(${s.id})">
              🗑️
            </button>
          </div>
        </td>
      </tr>
    `
      )
      .join("");
  } catch (err) {
    console.error("Failed to load timetable:", err);
  }
}

async function forceActivateSlot(slotId) {
  try {
    const res = await apiFetch(`/api/sessions/force-activate/${slotId}`, { method: "POST" });
    const data = await res.json();
    showToast("Timetable Session force-activated! Switching to Live Terminal.", "success");
    await checkActiveSession();
    switchView("view-live");
    startLiveCamera();
  } catch (err) {
    showToast("Failed to activate slot: " + err.message, "danger");
  }
}

async function deleteTimetableSlot(slotId) {
  if (!confirm("Deactivate this timetable slot?")) return;
  try {
    const res = await apiFetch(`/api/timetable/${slotId}`, { method: "DELETE" });
    if (res.ok) {
      showToast("Slot removed.", "success");
      loadTimetable();
    }
  } catch (err) {
    showToast("Error deleting slot", "danger");
  }
}

async function handleAddTimetableSlot(e) {
  e.preventDefault();
  const department = document.getElementById("slotDept").value;
  const year = document.getElementById("slotYear").value;
  const section = document.getElementById("slotSection").value;
  const day_of_week = document.getElementById("slotDay").value;
  const start_time = document.getElementById("slotStartTime").value;
  const end_time = document.getElementById("slotEndTime").value;
  const subject_id = parseInt(document.getElementById("slotSubjectSelect").value, 10);
  const classroom = document.getElementById("slotClassroom").value;

  try {
    const res = await apiFetch("/api/timetable", {
      method: "POST",
      body: { department, year, section, day_of_week, start_time, end_time, subject_id, classroom },
    });
    if (res.ok) {
      showToast("Timetable slot created.", "success");
      closeModal("modalAddSlot");
      loadTimetable();
      checkActiveSession();
    }
  } catch (err) {
    showToast("Error creating timetable slot", "danger");
  }
}

// -------------------------------------------------------------
// SUBJECTS & FACULTY
// -------------------------------------------------------------
async function loadSubjectsAndFaculty() {
  try {
    const [subRes, facRes] = await Promise.all([apiFetch("/api/subjects"), apiFetch("/api/faculty")]);
    const subjects = await subRes.json();
    const faculty = await facRes.json();

    const subTbody = document.getElementById("subjectsTableBody");
    subTbody.innerHTML = subjects
      .map(
        (s) => `
      <tr>
        <td><strong>${s.code}</strong></td>
        <td>${s.name}</td>
        <td>${s.department_code}</td>
      </tr>
    `
      )
      .join("");

    const facTbody = document.getElementById("facultyTableBody");
    facTbody.innerHTML = faculty
      .map(
        (f) => `
      <tr>
        <td><strong>${f.faculty_id}</strong></td>
        <td>${f.name}</td>
        <td>${f.email}</td>
      </tr>
    `
      )
      .join("");

    // Populate Subject select in Add Slot Modal
    const select = document.getElementById("slotSubjectSelect");
    select.innerHTML = subjects.map((s) => `<option value="${s.id}">${s.code} - ${s.name}</option>`).join("");
  } catch (err) {
    console.error("Failed to load subjects/faculty:", err);
  }
}

// -------------------------------------------------------------
// ATTENDANCE HISTORY & AUDITED STAFF ADJUSTMENTS
// -------------------------------------------------------------
async function loadAttendanceHistory() {
  const startDate = document.getElementById("historyStartDate")?.value || "";
  const endDate = document.getElementById("historyEndDate")?.value || "";
  const dept = document.getElementById("historyDeptFilter")?.value || "";
  const year = document.getElementById("historyYearFilter")?.value || "";
  const section = document.getElementById("historySectionFilter")?.value || "";
  const status = document.getElementById("historyStatusFilter")?.value || "";
  const search = document.getElementById("historySearchInput")?.value?.trim() || "";

  const params = new URLSearchParams();
  if (startDate) params.append("date_start", startDate);
  if (endDate) params.append("date_end", endDate);
  if (dept) params.append("department", dept);
  if (year) params.append("year", year);
  if (section) params.append("section", section);
  if (status) params.append("status", status);
  if (search) params.append("student", search);

  const url = `/api/staff/attendance-history${params.toString() ? '?' + params.toString() : ''}`;

  try {
    const res = await apiFetch(url);
    const records = await res.json();
    const tbody = document.getElementById("historyTableBody");
    if (!tbody) return;

    if (!records || records.length === 0) {
      tbody.innerHTML = `<tr><td colspan="10" style="text-align:center; padding: 24px;">No attendance records found matching filters.</td></tr>`;
      return;
    }

    tbody.innerHTML = records
      .map(
        (r) => {
          const isPresent = r.status === "PRESENT";
          const isLate = r.status === "LATE";
          const statusBadge = isPresent
            ? "badge-success"
            : isLate
            ? "badge-warning"
            : "badge-danger";

          const isStaffOverride = r.attendance_source === "STAFF_CORRECTION";
          const sourceBadge = isStaffOverride
            ? `<span class="badge badge-source badge-source-staff">STAFF OVERRIDE</span>`
            : `<span class="badge badge-source badge-source-camera">FACE RECOGNITION</span>`;

          const auditDetails = isStaffOverride && r.correction_reason
            ? `<div style="font-size: 11px;"><strong style="color: #fbbf24;">${r.modified_by || 'Staff'}:</strong> ${r.correction_reason}</div>`
            : `<div style="font-size: 11px; color: var(--text-muted);">${r.liveness_status || 'Verified Biometrics'}</div>`;

          const escapedName = (r.student_name || '').replace(/'/g, "\\'");

          return `
            <tr>
              <td>
                <img src="/api/students/${r.student_id}/photo" class="student-ref-thumb" alt="${r.student_name}" onerror="this.onerror=null;this.src='data:image/svg+xml,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'42\\' height=\\'42\\' viewBox=\\'0 0 24 24\\' fill=\\'%2364748b\\'><path d=\\'M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z\\'/></svg>';">
              </td>
              <td class="font-mono font-bold">${r.student_id}</td>
              <td class="font-bold">${r.student_name}</td>
              <td>${r.subject_name || '--'}</td>
              <td>${r.session_date || '--'}</td>
              <td class="font-mono ${isPresent ? 'text-success font-bold' : ''}">${r.first_seen || '--'}</td>
              <td><span class="badge ${statusBadge}">${r.status}</span></td>
              <td>${sourceBadge}</td>
              <td>${auditDetails}</td>
              <td>
                <button class="btn btn-outline btn-sm" onclick="openEditAttendanceModal(${r.id}, '${r.student_id}', '${escapedName}', '${r.status}', '${r.first_seen || ''}')" title="Audit Correct Attendance">
                  ✏️ Edit
                </button>
              </td>
            </tr>
          `;
        }
      )
      .join("");
  } catch (err) {
    console.error("Failed to load history:", err);
  }
}

function openEditAttendanceModal(recordId, studentId, studentName, currentStatus, currentTime) {
  document.getElementById("editRecordId").value = recordId;
  document.getElementById("editStudentName").textContent = studentName;
  document.getElementById("editStudentRoll").textContent = studentId;

  const badge = document.getElementById("editCurrentStatusBadge");
  if (badge) {
    badge.textContent = currentStatus;
    badge.className = `badge ${currentStatus === 'PRESENT' ? 'badge-success' : currentStatus === 'LATE' ? 'badge-warning' : 'badge-danger'}`;
  }

  document.getElementById("editNewStatus").value = currentStatus;
  document.getElementById("editNewTime").value = (currentTime && currentTime !== '--') ? currentTime : "";
  document.getElementById("editReason").value = "";

  openModal("modalEditAttendance");
}

async function handleSaveEditAttendance(e) {
  e.preventDefault();
  const recordId = document.getElementById("editRecordId").value;
  const newStatus = document.getElementById("editNewStatus").value;
  const newTime = document.getElementById("editNewTime").value.trim();
  const reason = document.getElementById("editReason").value.trim();

  if (!reason) {
    showToast("Reason for modification is strictly required for official audit logging.", "warning");
    document.getElementById("editReason").focus();
    return;
  }

  try {
    const res = await apiFetch(`/api/staff/attendance/${recordId}`, {
      method: "PUT",
      body: {
        attendance_id: parseInt(recordId, 10),
        new_status: newStatus,
        new_time: newTime || null,
        reason: reason
      }
    });

    if (res.ok) {
      showToast(`Attendance updated to ${newStatus} with audit trail logged.`, "success");
      closeModal("modalEditAttendance");
      loadAttendanceHistory();
      if (state.currentView === "view-staff-dashboard") {
        loadStaffDashboard();
      }
    } else {
      const err = await res.json();
      showToast(err.detail || "Failed to update attendance record.", "danger");
    }
  } catch (err) {
    showToast("Update error: " + err.message, "danger");
  }
}

function resetAttendanceHistoryFilters() {
  const fields = ["historyStartDate", "historyEndDate", "historyDeptFilter", "historyYearFilter", "historySectionFilter", "historyStatusFilter", "historySearchInput"];
  fields.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.value = "";
  });
  loadAttendanceHistory();
}

// -------------------------------------------------------------
// REPORTS EXPORTER (CSV / EXCEL / PDF)
// -------------------------------------------------------------
function downloadReport(format) {
  const date = document.getElementById("reportDate").value;
  const dept = document.getElementById("reportDept").value;

  let url = `/api/reports/export?format=${format}`;
  if (date) url += `&date=${encodeURIComponent(date)}`;
  if (dept) url += `&department=${encodeURIComponent(dept)}`;

  showToast(`Generating ${format.toUpperCase()} attendance report...`, "info");
  window.open(url, "_blank");
}

// -------------------------------------------------------------
// ACADEMIC BASELINE VS PROPOSED BENCHMARK
// -------------------------------------------------------------
async function runAcademicBenchmark() {
  const btn = document.getElementById("btnRunBenchmark");
  btn.disabled = true;
  btn.textContent = "Executing Benchmark...";
  showToast("Running empirical benchmark comparing Haar+KNN baseline with Proposed YuNet+ArcFace...", "info");

  try {
    const res = await apiFetch("/api/eval/benchmark");
    const data = await res.json();
    btn.disabled = false;
    btn.textContent = "Execute Live Benchmark ⚡";

    if (!data.has_data) {
      showToast(data.message || "Benchmark failed: insufficient samples.", "warning");
      return;
    }

    // Populate Baseline
    const b = data.baseline;
    document.getElementById("baseAcc").textContent = `${b.accuracy}%`;
    document.getElementById("basePrec").textContent = `${b.precision}%`;
    document.getElementById("baseRec").textContent = `${b.recall}%`;
    document.getElementById("baseF1").textContent = `${b.f1_score}%`;
    document.getElementById("baseFar").textContent = `${b.far}%`;
    document.getElementById("baseFrr").textContent = `${b.frr}%`;
    document.getElementById("baseSpeed").textContent = `${b.latency_ms} ms / ${b.fps} FPS`;

    // Populate Proposed
    const p = data.proposed;
    document.getElementById("propAcc").textContent = `${p.accuracy}%`;
    document.getElementById("propPrec").textContent = `${p.precision}%`;
    document.getElementById("propRec").textContent = `${p.recall}%`;
    document.getElementById("propF1").textContent = `${p.f1_score}%`;
    document.getElementById("propFar").textContent = `${p.far}%`;
    document.getElementById("propFrr").textContent = `${p.frr}%`;
    document.getElementById("propSpeed").textContent = `${p.latency_ms} ms / ${p.fps} FPS`;

    showToast(`Benchmark completed across ${data.test_trials_count} trials! Accuracy delta: +${data.improvement.accuracy_delta}%`, "success");
  } catch (err) {
    btn.disabled = false;
    btn.textContent = "Execute Live Benchmark ⚡";
    showToast("Benchmark error: " + err.message, "danger");
  }
}

// -------------------------------------------------------------
// ADMIN AUDIT LOGS
// -------------------------------------------------------------
async function loadAuditLogs() {
  try {
    const res = await apiFetch("/api/audit-logs");
    const logs = await res.json();
    const tbody = document.getElementById("auditTableBody");

    if (logs.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;">No audit records found.</td></tr>`;
      return;
    }

    tbody.innerHTML = logs
      .map(
        (l) => `
      <tr>
        <td><small>${l.timestamp}</small></td>
        <td><strong>${l.admin_name}</strong></td>
        <td><span class="badge badge-verifying">${l.action}</span></td>
        <td>${l.target_type}: ${l.target_id || ''}</td>
        <td>${l.old_value || '--'}</td>
        <td>${l.new_value || '--'}</td>
        <td><small>${l.details || ''}</small></td>
      </tr>
    `
      )
      .join("");
  } catch (err) {
    console.error("Failed to load audit logs:", err);
  }
}

// -------------------------------------------------------------
// SYSTEM SETTINGS
// -------------------------------------------------------------
async function loadSystemSettings() {
  try {
    const res = await apiFetch("/api/settings");
    const settings = await res.json();
    document.getElementById("settingThreshold").value = settings.recognition_threshold;
    document.getElementById("settingBlur").value = settings.blur_threshold;
    document.getElementById("settingFaceSize").value = settings.min_face_size;
    document.getElementById("settingTemporalFrames").value = settings.temporal_frames_required;
  } catch (err) {
    console.error("Failed to load settings:", err);
  }
}

async function saveSystemSettings(e) {
  e.preventDefault();
  const recognition_threshold = parseFloat(document.getElementById("settingThreshold").value);
  const blur_threshold = parseFloat(document.getElementById("settingBlur").value);
  const min_face_size = parseInt(document.getElementById("settingFaceSize").value, 10);
  const temporal_frames_required = parseInt(document.getElementById("settingTemporalFrames").value, 10);

  try {
    const res = await apiFetch("/api/settings", {
      method: "POST",
      body: { recognition_threshold, blur_threshold, min_face_size, temporal_frames_required },
    });
    if (res.ok) {
      showToast("Vision parameters updated and calibrated.", "success");
    }
  } catch (err) {
    showToast("Failed to save settings", "danger");
  }
}

// -------------------------------------------------------------
// -------------------------------------------------------------
// AUTHENTICATION & MODAL UTILITIES
// -------------------------------------------------------------
function openLoginModal() {
  const modal = document.getElementById("modalLogin");
  if (modal) modal.classList.add("active");
}

function closeLoginModal() {
  const modal = document.getElementById("modalLogin");
  if (modal) modal.classList.remove("active");
}

function openModal(modalId) {
  const el = document.getElementById(modalId);
  if (el) el.classList.add("active");
}

function closeModal(modalId) {
  const el = document.getElementById(modalId);
  if (el) el.classList.remove("active");
}

function setLoginRole(role) {
  const roleInput = document.getElementById("loginRole");
  const userLabel = document.getElementById("loginUserLabel");
  const userInput = document.getElementById("loginUsername");
  const passInput = document.getElementById("loginPassword");
  const btnSubmit = document.getElementById("btnLoginSubmit");

  if (roleInput) roleInput.value = role;

  // Update tabs active state
  document.querySelectorAll(".auth-role-tab").forEach((tab) => {
    if (tab.getAttribute("data-auth-role") === role) {
      tab.classList.add("active");
    } else {
      tab.classList.remove("active");
    }
  });

  if (role === "student") {
    if (userLabel) userLabel.textContent = "Roll Number / Student ID";
    if (userInput) {
      userInput.placeholder = "e.g. 23AD015";
      userInput.value = "23AD015";
    }
    if (passInput) passInput.value = "student123";
    if (btnSubmit) btnSubmit.textContent = "Sign In as Student";
  } else if (role === "staff") {
    if (userLabel) userLabel.textContent = "Faculty / Staff ID";
    if (userInput) {
      userInput.placeholder = "e.g. FAC01";
      userInput.value = "FAC01";
    }
    if (passInput) passInput.value = "staff123";
    if (btnSubmit) btnSubmit.textContent = "Sign In as Faculty";
  } else {
    // Admin
    if (userLabel) userLabel.textContent = "Administrator Username";
    if (userInput) {
      userInput.placeholder = "e.g. admin";
      userInput.value = "admin";
    }
    if (passInput) passInput.value = "admin123";
    if (btnSubmit) btnSubmit.textContent = "Sign In as Admin";
  }
}

function applyRoleToUI(role, displayName) {
  const safeRole = role || "admin";
  document.body.setAttribute("data-user-role", safeRole);

  const nameEl = document.getElementById("adminNameDisplay");
  const roleEl = document.getElementById("adminRoleDisplay");
  const avatarEl = document.getElementById("adminAvatar");

  if (nameEl) nameEl.textContent = displayName || "User";
  if (roleEl) roleEl.textContent = capitalize(safeRole);
  if (avatarEl) avatarEl.textContent = (displayName || safeRole).charAt(0).toUpperCase();
}

async function handleLogin(e) {
  e.preventDefault();
  const role = document.getElementById("loginRole")?.value || "student";
  const username = document.getElementById("loginUsername").value.trim();
  const password = document.getElementById("loginPassword").value;

  try {
    const res = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password, role }),
    });
    const data = await res.json();
    if (res.ok) {
      state.token = data.access_token;
      state.role = data.role;
      state.user = {
        id: data.user_id,
        name: data.admin_name,
        role: data.role,
        department: data.department || null,
      };
      state.admin = state.user; // backward compatibility

      localStorage.setItem("token", data.access_token);
      localStorage.setItem("role", data.role);
      localStorage.setItem("user", JSON.stringify(state.user));
      localStorage.setItem("admin", JSON.stringify(state.admin));

      applyRoleToUI(data.role, data.admin_name);
      closeLoginModal();
      showToast(`Welcome back, ${data.admin_name}!`, "success");

      // Role-based routing
      if (data.role === "student") {
        switchView("view-student-dashboard");
      } else if (data.role === "staff") {
        switchView("view-staff-dashboard");
      } else {
        switchView("view-dashboard");
      }
    } else {
      showToast(data.detail || "Invalid login credentials", "danger");
    }
  } catch (err) {
    showToast("Sign in failed: " + err.message, "danger");
  }
}

function handleLogout() {
  stopStudentCamera();
  stopLiveCamera();
  if (state.staffPollingTimer) {
    clearInterval(state.staffPollingTimer);
    state.staffPollingTimer = null;
  }

  state.token = null;
  state.role = null;
  state.user = null;
  state.admin = null;
  localStorage.removeItem("token");
  localStorage.removeItem("role");
  localStorage.removeItem("user");
  localStorage.removeItem("admin");

  document.body.removeAttribute("data-user-role");
  openLoginModal();
  showToast("Logged out successfully.", "info");
}

function capitalize(str) {
  if (!str) return "";
  return str.charAt(0).toUpperCase() + str.slice(1);
}

// -------------------------------------------------------------
// INITIALIZATION
// -------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  startClock();

  // Navigation Links
  document.querySelectorAll(".nav-item").forEach((item) => {
    item.addEventListener("click", () => {
      const view = item.getAttribute("data-view");
      if (view) switchView(view);
    });
  });

  // Buttons & Actions
  const btnGoToLive = document.getElementById("btnGoToLive");
  if (btnGoToLive) btnGoToLive.addEventListener("click", () => switchView("view-live"));
  const btnRefreshDashboard = document.getElementById("btnRefreshDashboard");
  if (btnRefreshDashboard) btnRefreshDashboard.addEventListener("click", loadDashboardData);

  // Live Camera (Admin / Live terminal)
  const btnStartCamera = document.getElementById("btnStartCamera");
  if (btnStartCamera) btnStartCamera.addEventListener("click", startLiveCamera);
  const btnStopCamera = document.getElementById("btnStopCamera");
  if (btnStopCamera) btnStopCamera.addEventListener("click", stopLiveCamera);

  // Student Dashboard Camera & Action Handlers
  const btnStudentStartCamera = document.getElementById("btnStudentStartCamera");
  if (btnStudentStartCamera) btnStudentStartCamera.addEventListener("click", startStudentCamera);
  const btnStudentStopCamera = document.getElementById("btnStudentStopCamera");
  if (btnStudentStopCamera) btnStudentStopCamera.addEventListener("click", stopStudentCamera);

  const btnScrollToCamera = document.getElementById("btnScrollToCamera");
  if (btnScrollToCamera) {
    btnScrollToCamera.addEventListener("click", () => {
      const camCard = document.getElementById("studentCameraSection") || document.getElementById("studentCameraCard");
      if (camCard) camCard.scrollIntoView({ behavior: "smooth", block: "center" });
      startStudentCamera();
    });
  }

  // Sidebar "Start Attendance" shortcut for students
  const navStudentStartCamera = document.getElementById("navStudentStartCamera");
  if (navStudentStartCamera) {
    navStudentStartCamera.addEventListener("click", () => {
      switchView("view-student-dashboard");
      setTimeout(() => {
        const camSection = document.getElementById("studentCameraSection") || document.getElementById("studentCameraCard");
        if (camSection) camSection.scrollIntoView({ behavior: "smooth", block: "center" });
        startStudentCamera();
      }, 100);
    });
  }

  // Sidebar "My Profile" shortcut for students
  const navStudentProfile = document.getElementById("navStudentProfile");
  if (navStudentProfile) {
    navStudentProfile.addEventListener("click", () => {
      switchView("view-student-profile");
    });
  }

  // Profile page action buttons
  const btnProfileGoToCamera = document.getElementById("btnProfileGoToCamera");
  if (btnProfileGoToCamera) {
    btnProfileGoToCamera.addEventListener("click", () => {
      switchView("view-student-dashboard");
      setTimeout(() => {
        const camSection = document.getElementById("studentCameraSection") || document.getElementById("studentCameraCard");
        if (camSection) camSection.scrollIntoView({ behavior: "smooth", block: "center" });
        startStudentCamera();
      }, 100);
    });
  }

  const btnProfileGoToHistory = document.getElementById("btnProfileGoToHistory");
  if (btnProfileGoToHistory) {
    btnProfileGoToHistory.addEventListener("click", () => {
      switchView("view-student-history");
    });
  }

  const btnRefreshStudentHistory = document.getElementById("btnRefreshStudentHistory");
  if (btnRefreshStudentHistory) btnRefreshStudentHistory.addEventListener("click", loadStudentDashboard);
  const btnRefreshStudentFullHistory = document.getElementById("btnRefreshStudentFullHistory");
  if (btnRefreshStudentFullHistory) btnRefreshStudentFullHistory.addEventListener("click", loadStudentAttendanceHistory);

  // Staff Dashboard Handlers
  const btnStaffGoToLive = document.getElementById("btnStaffGoToLive");
  if (btnStaffGoToLive) {
    btnStaffGoToLive.addEventListener("click", () => {
      switchView("view-live");
      startLiveCamera();
    });
  }

  const btnRefreshStaffLive = document.getElementById("btnRefreshStaffLive");
  if (btnRefreshStaffLive) btnRefreshStaffLive.addEventListener("click", loadStaffDashboard);

  const staffStudentSearch = document.getElementById("staffStudentSearch");
  if (staffStudentSearch) {
    staffStudentSearch.addEventListener("input", (e) => {
      const q = e.target.value.toLowerCase().trim();
      const rows = document.querySelectorAll("#staffClassStudentsBody tr");
      rows.forEach((row) => {
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(q) ? "" : "none";
      });
    });
  }

  // Enrollment Mode Switcher & File Upload Setup
  setupEnrollmentModeSwitcher();
  setupFileUploadDropzone();

  // Enrollment Camera & Actions
  const btnToggleEnroll = document.getElementById("btnToggleEnrollCamera");
  if (btnToggleEnroll) btnToggleEnroll.addEventListener("click", toggleEnrollCamera);
  const btnCaptureAngle = document.getElementById("btnCaptureCurrentAngle");
  if (btnCaptureAngle) btnCaptureAngle.addEventListener("click", captureActiveEnrollAngle);
  const btnSubmitEnroll = document.getElementById("btnSubmitEnrollment");
  if (btnSubmitEnroll) btnSubmitEnroll.addEventListener("click", submitStudentEnrollment);

  // Angle Card Click Handlers
  ENROLL_ANGLES.forEach((a) => {
    const cardId = getAngleCardId(a.id);
    const el = document.getElementById(cardId);
    if (el) el.addEventListener("click", () => selectEnrollAngle(a.id));
  });

  // Students management & filtering
  const btnStaffAddNewStudent = document.getElementById("btnStaffAddNewStudent");
  if (btnStaffAddNewStudent) btnStaffAddNewStudent.addEventListener("click", () => switchView("view-enroll"));
  const btnDirectoryAddStudent = document.getElementById("btnDirectoryAddStudent");
  if (btnDirectoryAddStudent) btnDirectoryAddStudent.addEventListener("click", () => switchView("view-enroll"));

  const searchStudentDirectory = document.getElementById("searchStudentDirectory");
  if (searchStudentDirectory) {
    searchStudentDirectory.addEventListener("input", (e) => {
      const q = e.target.value.toLowerCase().trim();
      const rows = document.querySelectorAll("#studentsTableBody tr");
      rows.forEach((row) => {
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(q) ? "" : "none";
      });
    });
  }

  const filterStudentDept = document.getElementById("filterStudentDept");
  if (filterStudentDept) filterStudentDept.addEventListener("change", loadStudentsList);
  const btnRefreshStudents = document.getElementById("btnRefreshStudents");
  if (btnRefreshStudents) btnRefreshStudents.addEventListener("click", loadStudentsList);

  // Student Edit & View Modals
  const formEditStudent = document.getElementById("formEditStudent");
  if (formEditStudent) formEditStudent.addEventListener("submit", handleSaveEditStudent);
  const btnCloseEditStudentModal = document.getElementById("btnCloseEditStudentModal");
  if (btnCloseEditStudentModal) btnCloseEditStudentModal.addEventListener("click", () => closeModal("modalEditStudent"));
  const btnCancelEditStudent = document.getElementById("btnCancelEditStudent");
  if (btnCancelEditStudent) btnCancelEditStudent.addEventListener("click", () => closeModal("modalEditStudent"));

  const btnCloseViewStudentModal = document.getElementById("btnCloseViewStudentModal");
  if (btnCloseViewStudentModal) btnCloseViewStudentModal.addEventListener("click", () => closeModal("modalViewStudent"));
  const btnCloseViewStudentBtn = document.getElementById("btnCloseViewStudentBtn");
  if (btnCloseViewStudentBtn) btnCloseViewStudentBtn.addEventListener("click", () => closeModal("modalViewStudent"));

  // Timetable
  const btnOpenAddSlotModal = document.getElementById("btnOpenAddSlotModal");
  if (btnOpenAddSlotModal) {
    btnOpenAddSlotModal.addEventListener("click", () => {
      loadSubjectsAndFaculty();
      openModal("modalAddSlot");
    });
  }
  const btnCloseAddSlotModal = document.getElementById("btnCloseAddSlotModal");
  if (btnCloseAddSlotModal) btnCloseAddSlotModal.addEventListener("click", () => closeModal("modalAddSlot"));
  const formAddTimetableSlot = document.getElementById("formAddTimetableSlot");
  if (formAddTimetableSlot) formAddTimetableSlot.addEventListener("submit", handleAddTimetableSlot);

  // History & Audited Overrides
  const btnFilterHistory = document.getElementById("btnFilterHistory");
  if (btnFilterHistory) btnFilterHistory.addEventListener("click", loadAttendanceHistory);
  const btnResetHistoryFilters = document.getElementById("btnResetHistoryFilters");
  if (btnResetHistoryFilters) btnResetHistoryFilters.addEventListener("click", resetAttendanceHistoryFilters);

  ["historyDeptFilter", "historyYearFilter", "historySectionFilter", "historyStatusFilter"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("change", loadAttendanceHistory);
  });
  const historySearchInput = document.getElementById("historySearchInput");
  if (historySearchInput) {
    historySearchInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        loadAttendanceHistory();
      }
    });
  }

  // Attendance Edit Modal Handlers
  const formEditAttendance = document.getElementById("formEditAttendance");
  if (formEditAttendance) formEditAttendance.addEventListener("submit", handleSaveEditAttendance);
  const btnCloseEditAttendanceModal = document.getElementById("btnCloseEditAttendanceModal");
  if (btnCloseEditAttendanceModal) btnCloseEditAttendanceModal.addEventListener("click", () => closeModal("modalEditAttendance"));
  const btnCancelEditAttendance = document.getElementById("btnCancelEditAttendance");
  if (btnCancelEditAttendance) btnCancelEditAttendance.addEventListener("click", () => closeModal("modalEditAttendance"));

  // Reports Exports
  const btnExportCsv = document.getElementById("btnExportCsv");
  if (btnExportCsv) btnExportCsv.addEventListener("click", () => downloadReport("csv"));
  const btnExportExcel = document.getElementById("btnExportExcel");
  if (btnExportExcel) btnExportExcel.addEventListener("click", () => downloadReport("excel"));
  const btnExportPdf = document.getElementById("btnExportPdf");
  if (btnExportPdf) btnExportPdf.addEventListener("click", () => downloadReport("pdf"));

  // Academic Benchmark
  const btnRunBenchmark = document.getElementById("btnRunBenchmark");
  if (btnRunBenchmark) btnRunBenchmark.addEventListener("click", runAcademicBenchmark);

  // Audit Logs
  const btnRefreshAudit = document.getElementById("btnRefreshAudit");
  if (btnRefreshAudit) btnRefreshAudit.addEventListener("click", loadAuditLogs);

  // Settings
  const btnSaveSettings = document.getElementById("btnSaveSettings");
  if (btnSaveSettings) btnSaveSettings.addEventListener("click", saveSystemSettings);

  // Auth Modal Role Tabs
  const tabRoleStudent = document.getElementById("tabRoleStudent");
  if (tabRoleStudent) tabRoleStudent.addEventListener("click", () => setLoginRole("student"));
  const tabRoleStaff = document.getElementById("tabRoleStaff");
  if (tabRoleStaff) tabRoleStaff.addEventListener("click", () => setLoginRole("staff"));
  const tabRoleAdmin = document.getElementById("tabRoleAdmin");
  if (tabRoleAdmin) tabRoleAdmin.addEventListener("click", () => setLoginRole("admin"));

  // Quick-fill credentials buttons
  const btnQuickStudent = document.getElementById("btnQuickFillStudent");
  if (btnQuickStudent) btnQuickStudent.addEventListener("click", () => setLoginRole("student"));
  const btnQuickStaff = document.getElementById("btnQuickFillStaff");
  if (btnQuickStaff) btnQuickStaff.addEventListener("click", () => setLoginRole("staff"));
  const btnQuickAdmin = document.getElementById("btnQuickFillAdmin");
  if (btnQuickAdmin) btnQuickAdmin.addEventListener("click", () => setLoginRole("admin"));

  // Auth Form & Logout
  const loginForm = document.getElementById("loginForm");
  if (loginForm) loginForm.addEventListener("submit", handleLogin);
  const btnLogout = document.getElementById("btnLogout");
  if (btnLogout) btnLogout.addEventListener("click", handleLogout);

  // Check login state on load
  if (!state.token || !state.role) {
    setLoginRole("student");
    openLoginModal();
  } else {
    const displayName = state.user?.name || state.admin?.name || "User";
    applyRoleToUI(state.role, displayName);

    if (state.role === "student") {
      switchView("view-student-dashboard");
    } else if (state.role === "staff") {
      switchView("view-staff-dashboard");
    } else {
      switchView("view-dashboard");
    }
  }

  // Background Timetable Poll every 8 seconds
  setInterval(checkActiveSession, 8000);
});
