import sys
from pathlib import Path
import uvicorn

# Ensure project root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if __name__ == "__main__":
    print("\n=======================================================")
    print("🚀 Smart Attendance System starting on http://127.0.0.1:8000")
    print("=======================================================\n")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
