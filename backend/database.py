"""SQLite persistence for analyses and bookings.

The app still exposes the old storage helper names for compatibility, but
these tables are now the source of truth for core product records.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import json
import logging
import secrets
import sqlite3
import threading
from pathlib import Path
from typing import Any

try:
    from .config import BACKEND_DIR, DATABASE_PATH
except ImportError:
    from config import BACKEND_DIR, DATABASE_PATH


logger = logging.getLogger(__name__)
_INITIALIZED = False
_INIT_LOCK = threading.Lock()


def connect() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def session():
    connection = connect()
    try:
        yield connection
    finally:
        connection.close()


def init_db() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    with _INIT_LOCK:
        if _INITIALIZED:
            return

        with session() as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    filename TEXT,
                    details_json TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS bookings (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL,
                    coach TEXT NOT NULL,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    preferred_date TEXT NOT NULL,
                    preferred_time TEXT NOT NULL,
                    preferred_slot TEXT NOT NULL DEFAULT '',
                    message TEXT NOT NULL DEFAULT '',
                    updated_at TEXT
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    upload_path TEXT NOT NULL,
                    access_token TEXT,
                    details_json TEXT NOT NULL,
                    progress_json TEXT,
                    result_json TEXT,
                    error TEXT
                )
                """
            )
            _ensure_column(db, "analysis_jobs", "access_token", "TEXT")
            _ensure_column(db, "analysis_jobs", "progress_json", "TEXT")
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_login_at TEXT
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS user_locations (
                    user_id TEXT PRIMARY KEY,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    accuracy REAL,
                    source TEXT NOT NULL DEFAULT 'browser',
                    consented_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            )
            db.execute("CREATE INDEX IF NOT EXISTS idx_analysis_timestamp ON analysis_records(timestamp)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_booking_timestamp ON bookings(timestamp)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_booking_status ON bookings(status)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_analysis_jobs_status ON analysis_jobs(status)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_user_locations_updated ON user_locations(updated_at)")
            db.commit()

        _INITIALIZED = True
        migrate_json_if_empty()


def _read_legacy_json(filename: str):
    path = Path(BACKEND_DIR) / filename
    try:
        with path.open("r", encoding="utf-8") as f:
            records = json.load(f)
        return records if isinstance(records, list) else []
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as exc:
        logger.warning("Could not migrate corrupt legacy JSON %s: %s", filename, exc)
        return []


_ALLOWED_TABLES = {"analysis_records", "bookings", "analysis_jobs", "users", "user_locations"}
_ALLOWED_COLUMN_TYPES = {"TEXT", "INTEGER", "REAL", "BLOB"}


def _ensure_column(db: sqlite3.Connection, table: str, column: str, column_type: str) -> None:
    if table not in _ALLOWED_TABLES:
        raise ValueError(f"Unexpected table name: {table!r}")
    if column_type not in _ALLOWED_COLUMN_TYPES:
        raise ValueError(f"Unexpected column type: {column_type!r}")
    existing = {row["name"] for row in db.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in existing:
        db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")


def migrate_json_if_empty() -> None:
    """Import existing prototype JSON files once, without deleting them."""
    with session() as db:
        analysis_count = db.execute("SELECT COUNT(*) FROM analysis_records").fetchone()[0]
        booking_count = db.execute("SELECT COUNT(*) FROM bookings").fetchone()[0]

    if analysis_count == 0:
        legacy_records = _read_legacy_json("data.json")
        if legacy_records:
            replace_analysis_records(legacy_records)
            logger.info("Migrated %d legacy analysis records into SQLite.", len(legacy_records))

    if booking_count == 0:
        legacy_bookings = _read_legacy_json("bookings.json")
        if legacy_bookings:
            replace_bookings(legacy_bookings)
            logger.info("Migrated %d legacy bookings into SQLite.", len(legacy_bookings))


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _loads(value: str):
    return json.loads(value)


def create_user(user: dict[str, Any]) -> dict[str, Any]:
    init_db()
    with session() as db:
        db.execute(
            """
            INSERT INTO users (id, email, name, role, password_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                user["id"],
                user["email"].lower(),
                user["name"],
                user["role"],
                user["password_hash"],
                user.get("created_at") or datetime.now().isoformat(),
            ),
        )
        db.commit()
    created = get_user_by_id(user["id"])
    if created is None:
        raise RuntimeError("User was not created.")
    return created


def get_user_by_email(email: str):
    init_db()
    with session() as db:
        row = db.execute(
            """
            SELECT id, email, name, role, password_hash, created_at, last_login_at
            FROM users
            WHERE email = ?
            """,
            (email.lower(),),
        ).fetchone()
    return _user_from_row(row)


def get_user_by_id(user_id: str):
    init_db()
    with session() as db:
        row = db.execute(
            """
            SELECT id, email, name, role, password_hash, created_at, last_login_at
            FROM users
            WHERE id = ?
            """,
            (user_id,),
        ).fetchone()
    return _user_from_row(row)


def mark_user_login(user_id: str) -> None:
    init_db()
    with session() as db:
        db.execute(
            "UPDATE users SET last_login_at = ? WHERE id = ?",
            (datetime.now().isoformat(), user_id),
        )
        db.commit()


def _user_from_row(row):
    if row is None:
        return None
    return {
        "id": row["id"],
        "email": row["email"],
        "name": row["name"],
        "role": row["role"],
        "password_hash": row["password_hash"],
        "created_at": row["created_at"],
        "last_login_at": row["last_login_at"],
    }


def upsert_user_location(
    user_id: str,
    latitude: float,
    longitude: float,
    accuracy: float | None = None,
    source: str = "browser",
) -> dict[str, Any]:
    init_db()
    now = datetime.now().isoformat()
    with session() as db:
        existing = db.execute(
            "SELECT consented_at FROM user_locations WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        consented_at = existing["consented_at"] if existing else now
        db.execute(
            """
            INSERT INTO user_locations (
                user_id, latitude, longitude, accuracy, source, consented_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                latitude = excluded.latitude,
                longitude = excluded.longitude,
                accuracy = excluded.accuracy,
                source = excluded.source,
                updated_at = excluded.updated_at
            """,
            (user_id, latitude, longitude, accuracy, source, consented_at, now),
        )
        db.commit()
    location = get_user_location(user_id)
    if location is None:
        raise RuntimeError("Location was not saved.")
    return location


def get_user_location(user_id: str):
    init_db()
    with session() as db:
        row = db.execute(
            """
            SELECT user_id, latitude, longitude, accuracy, source, consented_at, updated_at
            FROM user_locations
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
    return _location_from_row(row)


def delete_user_location(user_id: str) -> None:
    init_db()
    with session() as db:
        db.execute("DELETE FROM user_locations WHERE user_id = ?", (user_id,))
        db.commit()


def _location_from_row(row):
    if row is None:
        return None
    return {
        "user_id": row["user_id"],
        "latitude": row["latitude"],
        "longitude": row["longitude"],
        "accuracy": row["accuracy"],
        "source": row["source"],
        "consented_at": row["consented_at"],
        "updated_at": row["updated_at"],
    }


def list_analysis_records() -> list[dict[str, Any]]:
    init_db()
    with session() as db:
        rows = db.execute(
            """
            SELECT timestamp, details_json, report_json
            FROM analysis_records
            ORDER BY id ASC
            """
        ).fetchall()
    return [
        {
            "timestamp": row["timestamp"],
            "details": _loads(row["details_json"]),
            "report": _loads(row["report_json"]),
        }
        for row in rows
    ]


def replace_analysis_records(records: list[dict[str, Any]]) -> None:
    init_db()
    with session() as db:
        db.execute("DELETE FROM analysis_records")
        db.executemany(
            """
            INSERT INTO analysis_records (timestamp, filename, details_json, report_json)
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    record.get("timestamp") or datetime.now().isoformat(),
                    record.get("details", {}).get("filename"),
                    _dumps(record.get("details", {})),
                    _dumps(record.get("report", {})),
                )
                for record in records
            ],
        )
        db.commit()


def append_analysis_record(record: dict[str, Any]) -> None:
    init_db()
    with session() as db:
        db.execute(
            """
            INSERT INTO analysis_records (timestamp, filename, details_json, report_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                record.get("timestamp") or datetime.now().isoformat(),
                record.get("details", {}).get("filename"),
                _dumps(record.get("details", {})),
                _dumps(record.get("report", {})),
            ),
        )
        db.commit()


def create_analysis_job(job_id: str, filename: str, upload_path: str, details: dict[str, Any]) -> dict[str, Any]:
    init_db()
    now = datetime.now().isoformat()
    job = {
        "id": job_id,
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        "filename": filename,
        "upload_path": upload_path,
        "access_token": secrets.token_urlsafe(24),
        "details": details,
        "progress": {
            "percent": 0,
            "phase": "Queued",
            "message": "Waiting for analysis worker.",
            "frames_analyzed": 0,
        },
    }
    with session() as db:
        db.execute(
            """
            INSERT INTO analysis_jobs (
                id, status, created_at, updated_at, filename, upload_path, access_token,
                details_json, progress_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job["id"],
                job["status"],
                job["created_at"],
                job["updated_at"],
                job["filename"],
                job["upload_path"],
                job["access_token"],
                _dumps(job["details"]),
                _dumps(job["progress"]),
            ),
        )
        db.commit()
    return job


def get_analysis_job(job_id: str):
    init_db()
    with session() as db:
        row = db.execute(
            """
            SELECT id, status, created_at, updated_at, filename, upload_path,
                   access_token, details_json, progress_json, result_json, error
            FROM analysis_jobs
            WHERE id = ?
            """,
            (job_id,),
        ).fetchone()
    if row is None:
        return None
    job = {
        "id": row["id"],
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "filename": row["filename"],
        "upload_path": row["upload_path"],
        "access_token": row["access_token"],
        "details": _loads(row["details_json"]),
    }
    if row["result_json"]:
        job["result"] = _loads(row["result_json"])
    if row["progress_json"]:
        job["progress"] = _loads(row["progress_json"])
    else:
        job["progress"] = {"percent": 100 if row["status"] == "complete" else 0, "phase": row["status"].title()}
    if row["error"]:
        job["error"] = row["error"]
    return job


def list_analysis_jobs_by_status(statuses: list[str]) -> list[dict[str, Any]]:
    init_db()
    placeholders = ", ".join("?" for _ in statuses)
    with session() as db:
        rows = db.execute(
            f"""
            SELECT id
            FROM analysis_jobs
            WHERE status IN ({placeholders})
            ORDER BY created_at ASC
            """,
            statuses,
        ).fetchall()
    return [get_analysis_job(row["id"]) for row in rows]


def update_analysis_job_status(job_id: str, status: str, error: str | None = None):
    init_db()
    with session() as db:
        db.execute(
            """
            UPDATE analysis_jobs
            SET status = ?, updated_at = ?, error = COALESCE(?, error)
            WHERE id = ?
            """,
            (status, datetime.now().isoformat(), error, job_id),
        )
        db.commit()
    return get_analysis_job(job_id)


def update_analysis_job_progress(job_id: str, progress: dict[str, Any]):
    init_db()
    with session() as db:
        db.execute(
            """
            UPDATE analysis_jobs
            SET updated_at = ?, progress_json = ?
            WHERE id = ?
            """,
            (datetime.now().isoformat(), _dumps(progress), job_id),
        )
        db.commit()
    return get_analysis_job(job_id)


def complete_analysis_job(job_id: str, result: dict[str, Any]):
    progress = {
        "percent": 100,
        "phase": "Complete",
        "message": "Full match analysis is ready.",
        "frames_analyzed": result.get("report", {}).get("video_metrics", {}).get("sampled_frames", 0),
    }
    init_db()
    with session() as db:
        db.execute(
            """
            UPDATE analysis_jobs
            SET status = ?, updated_at = ?, progress_json = ?, result_json = ?, error = NULL
            WHERE id = ?
            """,
            ("complete", datetime.now().isoformat(), _dumps(progress), _dumps(result), job_id),
        )
        db.commit()
    return get_analysis_job(job_id)


def replace_analysis_jobs(jobs: list[dict[str, Any]]) -> None:
    init_db()
    with session() as db:
        db.execute("DELETE FROM analysis_jobs")
        db.executemany(
            """
            INSERT INTO analysis_jobs (
                id, status, created_at, updated_at, filename, upload_path,
                access_token, details_json, progress_json, result_json, error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    job["id"],
                    job.get("status") or "complete",
                    job.get("created_at") or datetime.now().isoformat(),
                    job.get("updated_at") or datetime.now().isoformat(),
                    job.get("filename") or job.get("details", {}).get("filename") or "",
                    job.get("upload_path") or "",
                    job.get("access_token"),
                    _dumps(job.get("details", {})),
                    _dumps(job.get("progress", {})),
                    _dumps(job["result"]) if job.get("result") else None,
                    job.get("error"),
                )
                for job in jobs
            ],
        )
        db.commit()


def list_bookings() -> list[dict[str, Any]]:
    init_db()
    with session() as db:
        rows = db.execute(
            """
            SELECT id, timestamp, status, coach, name, email, preferred_date,
                   preferred_time, preferred_slot, message, updated_at
            FROM bookings
            ORDER BY timestamp ASC, id ASC
            """
        ).fetchall()
    return [
        {
            key: row[key]
            for key in row.keys()
            if row[key] is not None
        }
        for row in rows
    ]


def replace_bookings(bookings: list[dict[str, Any]]) -> None:
    init_db()
    with session() as db:
        db.execute("DELETE FROM bookings")
        db.executemany(
            """
            INSERT OR REPLACE INTO bookings (
                id, timestamp, status, coach, name, email, preferred_date,
                preferred_time, preferred_slot, message, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [_booking_tuple(booking) for booking in bookings],
        )
        db.commit()


def append_booking(booking: dict[str, Any]) -> None:
    init_db()
    with session() as db:
        db.execute(
            """
            INSERT OR REPLACE INTO bookings (
                id, timestamp, status, coach, name, email, preferred_date,
                preferred_time, preferred_slot, message, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            _booking_tuple(booking),
        )
        db.commit()


def update_booking_status(booking_id: str, status: str):
    init_db()
    updated_at = datetime.now().isoformat()
    with session() as db:
        cursor = db.execute(
            """
            UPDATE bookings
            SET status = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, updated_at, booking_id),
        )
        db.commit()
        if cursor.rowcount == 0:
            return None

    return get_booking(booking_id)


def get_booking(booking_id: str):
    init_db()
    with session() as db:
        row = db.execute(
            """
            SELECT id, timestamp, status, coach, name, email, preferred_date,
                   preferred_time, preferred_slot, message, updated_at
            FROM bookings
            WHERE id = ?
            """,
            (booking_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        key: row[key]
        for key in row.keys()
        if row[key] is not None
    }


def _booking_tuple(booking: dict[str, Any]):
    return (
        booking["id"],
        booking.get("timestamp") or datetime.now().isoformat(),
        booking.get("status") or "pending",
        booking.get("coach") or "",
        booking.get("name") or "",
        booking.get("email") or "",
        booking.get("preferred_date") or "",
        booking.get("preferred_time") or "",
        booking.get("preferred_slot") or "",
        booking.get("message") or "",
        booking.get("updated_at"),
    )
