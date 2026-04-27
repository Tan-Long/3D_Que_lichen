"""Application configuration."""

import os
from pathlib import Path

# Root of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Where uploaded zips and extracted datasets live
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

# Path to the processing scripts
SCRIPTS_DIR = BASE_DIR

# Python executable inside the venv (has open3d, cv2, etc.)
if os.name == "nt":
    VENV_PYTHON = str(BASE_DIR / "venv311" / "Scripts" / "python.exe")
else:
    VENV_PYTHON = str(BASE_DIR / "venv311" / "bin" / "python")

# Max upload size: 500 MB
MAX_UPLOAD_SIZE = 500 * 1024 * 1024

# SQLite database path
DATABASE_PATH = UPLOADS_DIR / "app.db"

# Base directory containing dataset folders for bulk import
# Set via environment variable or change this default
DEFAULT_EXTERNAL_DATA_DIR = Path(r"E:\3D_Que\Raw_data")
EXTERNAL_DATA_DIR = os.environ.get("SCAN_DATA_DIR", str(DEFAULT_EXTERNAL_DATA_DIR))

# Team member names (simple identity, no auth)
TEAM_MEMBERS = os.environ.get("TEAM_MEMBERS", "").split(",") if os.environ.get("TEAM_MEMBERS") else []
