import os
from pathlib import Path
import uuid

import database
import storage


def use_temp_database(monkeypatch):
    artifact_dir = Path(__file__).resolve().parents[1] / "test-artifacts"
    artifact_dir.mkdir(exist_ok=True)
    db_path = artifact_dir / f"picklecoach-test-{uuid.uuid4().hex}.sqlite3"
    monkeypatch.setattr(database, "DATABASE_PATH", db_path)
    monkeypatch.setattr(database, "_INITIALIZED", False)
    return db_path


def test_analysis_records_are_stored_in_sqlite(monkeypatch):
    db_path = use_temp_database(monkeypatch)

    try:
        record = {
            "timestamp": "2026-04-19T10:00:00",
            "details": {"filename": "match-1.mp4", "name": "Test Player"},
            "report": {"rallies": 12, "unforced_errors": 3},
        }

        storage.save_json_records("data.json", [record])

        assert storage.load_json_records("data.json") == [record]
    finally:
        if db_path.exists():
            os.remove(db_path)


def test_booking_status_updates_in_sqlite(monkeypatch):
    db_path = use_temp_database(monkeypatch)

    try:
        booking = {
            "id": "booking-1",
            "timestamp": "2026-04-19T10:00:00",
            "status": "pending",
            "coach": "Sarah Johnson",
            "name": "Test Player",
            "email": "test@example.com",
            "preferred_date": "2026-05-01",
            "preferred_time": "10:30",
            "preferred_slot": "Mon 5:00 PM",
            "message": "SQLite test",
        }

        storage.save_json_records("bookings.json", [booking])
        updated = storage.update_booking_status("booking-1", "accepted")

        assert updated["status"] == "accepted"
        assert updated["updated_at"]
        assert storage.load_json_records("bookings.json")[0]["status"] == "accepted"
    finally:
        if db_path.exists():
            os.remove(db_path)


def test_analysis_job_lifecycle_in_sqlite(monkeypatch):
    db_path = use_temp_database(monkeypatch)

    try:
        job = storage.create_analysis_job(
            "job-1",
            "match-1.mp4",
            "backend/uploads/match-1.mp4",
            {"filename": "match-1.mp4", "name": "Test Player"},
        )

        assert job["status"] == "queued"
        assert storage.mark_analysis_job_processing("job-1")["status"] == "processing"

        record = {
            "timestamp": "2026-04-19T10:00:00",
            "details": {"filename": "match-1.mp4", "name": "Test Player"},
            "report": {"rallies": 12},
        }
        completed = storage.complete_analysis_job("job-1", record)

        assert completed["status"] == "complete"
        assert completed["result"] == record
    finally:
        if db_path.exists():
            os.remove(db_path)
