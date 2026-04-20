"""CSRF token helpers for form-backed routes."""

from __future__ import annotations

import secrets
import uuid
import hashlib
import os

from itsdangerous import BadSignature, URLSafeTimedSerializer

try:
    from .config import APP_ENV, CSRF_COOKIE, CSRF_MAX_AGE, CSRF_SECRET
except ImportError:
    from config import APP_ENV, CSRF_COOKIE, CSRF_MAX_AGE, CSRF_SECRET


_csrf_serializer = URLSafeTimedSerializer(CSRF_SECRET)
_session_serializer = URLSafeTimedSerializer(CSRF_SECRET, salt="picklecoach-session")
SESSION_COOKIE = "session_token"
SESSION_MAX_AGE = 60 * 60 * 24 * 14
PASSWORD_ITERATIONS = 260_000


def generate_csrf_token() -> str:
    return _csrf_serializer.dumps({"purpose": "csrf", "nonce": uuid.uuid4().hex})


def validate_csrf_token(cookie_token: str, form_token: str) -> bool:
    try:
        if not secrets.compare_digest(cookie_token, form_token):
            return False
        payload = _csrf_serializer.loads(cookie_token, max_age=CSRF_MAX_AGE)
        _csrf_serializer.loads(form_token, max_age=CSRF_MAX_AGE)
        return payload.get("purpose") == "csrf"
    except BadSignature:
        return False


def set_csrf_cookie(response, csrf_token: str) -> None:
    response.set_cookie(
        CSRF_COOKIE,
        csrf_token,
        httponly=True,
        samesite="lax",
        secure=APP_ENV not in {"development", "dev", "test"},
    )


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS,
    )
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iterations, salt_hex, digest_hex = stored_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(iterations),
        )
        return secrets.compare_digest(digest.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


def create_session_token(user_id: str) -> str:
    return _session_serializer.dumps({"purpose": "session", "user_id": user_id})


def read_session_token(token: str):
    if not token:
        return None
    try:
        payload = _session_serializer.loads(token, max_age=SESSION_MAX_AGE)
    except BadSignature:
        return None
    if payload.get("purpose") != "session":
        return None
    return payload.get("user_id")


def set_session_cookie(response, user_id: str) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        create_session_token(user_id),
        httponly=True,
        samesite="lax",
        secure=APP_ENV not in {"development", "dev", "test"},
        max_age=SESSION_MAX_AGE,
    )


def clear_session_cookie(response) -> None:
    response.delete_cookie(SESSION_COOKIE)
