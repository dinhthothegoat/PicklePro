"""Compatibility storage helpers.

Core product records now live in SQLite. The JSON helpers remain for any
miscellaneous prototype files that are not part of the database schema yet.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from filelock import FileLock

try:
    from .config import BACKEND_DIR
    from . import database
except ImportError:
    from config import BACKEND_DIR
    import database


logger = logging.getLogger(__name__)


def get_lock_path(filename: str) -> str:
    return str(Path(BACKEND_DIR) / filename) + ".lock"


def write_json(path: str | Path, records) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def read_json_records(path: str | Path, filename: str):
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as exc:
        logger.error("Corrupt JSON in %s: %s", filename, exc)
        return []


def load_json_records(filename: str):
    """Load records, using SQLite for core product files."""
    if filename == "data.json":
        return database.list_analysis_records()
    if filename == "bookings.json":
        return database.list_bookings()

    path = Path(BACKEND_DIR) / filename
    with FileLock(get_lock_path(filename)):
        return read_json_records(path, filename)


def save_json_records(filename: str, records) -> None:
    """Save records, using SQLite for core product files."""
    if filename == "data.json":
        database.replace_analysis_records(records)
        return
    if filename == "bookings.json":
        database.replace_bookings(records)
        return

    path = Path(BACKEND_DIR) / filename
    with FileLock(get_lock_path(filename)):
        write_json(path, records)


def append_analysis_record(record: dict) -> None:
    database.append_analysis_record(record)


def append_booking(booking: dict) -> None:
    database.append_booking(booking)


def create_analysis_job(job_id: str, filename: str, upload_path: str, details: dict):
    return database.create_analysis_job(job_id, filename, upload_path, details)


def get_analysis_job(job_id: str):
    return database.get_analysis_job(job_id)


def list_analysis_jobs_by_status(statuses: list[str]):
    return database.list_analysis_jobs_by_status(statuses)


def mark_analysis_job_processing(job_id: str):
    return database.update_analysis_job_status(job_id, "processing")


def update_analysis_job_progress(job_id: str, progress: dict):
    return database.update_analysis_job_progress(job_id, progress)


def complete_analysis_job(job_id: str, result: dict):
    return database.complete_analysis_job(job_id, result)


def replace_analysis_jobs(jobs: list[dict]) -> None:
    database.replace_analysis_jobs(jobs)


def fail_analysis_job(job_id: str, error: str):
    return database.update_analysis_job_status(job_id, "failed", error)


def update_booking_status(booking_id: str, status: str):
    """Update a booking status by id."""
    booking = database.update_booking_status(booking_id, status)
    if booking:
        logger.info("Booking %s -> %s", booking_id, status)
    return booking
