"""Application configuration and filesystem paths."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

BACKEND_DIR = Path(__file__).resolve().parent

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8080"))
BOOKINGS_PASSWORD = os.getenv("BOOKINGS_PASSWORD", "")

DEFAULT_CSRF_SECRET = "dev-csrf-secret-change-in-production"
APP_ENV = os.getenv("APP_ENV", "development").lower()
CSRF_SECRET = os.getenv("CSRF_SECRET", DEFAULT_CSRF_SECRET)
if APP_ENV not in {"development", "dev", "test"} and CSRF_SECRET == DEFAULT_CSRF_SECRET:
    raise RuntimeError("Set CSRF_SECRET to a long random value outside development.")

CSRF_COOKIE = "csrf_token"
CSRF_MAX_AGE = 3600
UPLOAD_CHUNK_SIZE = 1024 * 1024
MAX_FILE_SIZE = 100 * 1024 * 1024

EXAMPLE_YOUTUBE_URL = "https://youtu.be/pthJ1IQPqGE?si=eVw5judspJHy5Ku4"
PRO_EXAMPLE_VIDEO_IDS = {"pthJ1IQPqGE"}
YOUTUBE_DOWNLOAD_FORMAT = "bv*[ext=mp4][height<=360]/b[ext=mp4][height<=360]/worst[height<=360]/worst"

STATIC_DIR = BACKEND_DIR / "static"
TEMPLATES_DIR = BACKEND_DIR / "templates"
COACHES_PATH = BACKEND_DIR / "coaches.json"
CV_MODEL_ARTIFACT = BACKEND_DIR / "models" / "cv_model.pkl"
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", str(BACKEND_DIR / "picklecoach.sqlite3")))
