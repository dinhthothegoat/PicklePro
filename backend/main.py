from fastapi import BackgroundTasks, FastAPI, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from datetime import date, datetime
import logging
import math
import os
import re
import secrets
import tempfile
import uuid
import imageio.v2 as imageio  # type: ignore[import]
import numpy as np
import uvicorn
import yt_dlp as yt_dlp  # type: ignore[import]
try:
    from .config import (
        BOOKINGS_PASSWORD,
        CSRF_COOKIE,
        CV_MODEL_ARTIFACT,
        EXAMPLE_YOUTUBE_URL,
        HOST,
        MAX_FILE_SIZE,
        PORT,
        STATIC_DIR,
        TEMPLATES_DIR,
        UPLOAD_CHUNK_SIZE,
        YOUTUBE_DOWNLOAD_FORMAT,
    )
    from .cv_learning import load_cv_artifact, predict_with_artifact
    from .marketplace import (
        COACHES_PER_PAGE,
        COACH_TIER_REQUIREMENTS,
        coaches,
        find_coach_by_name,
        find_coach_by_slug,
    )
    from .match_intelligence import build_ml_report, clamp
    from .shot_detection import WholeMatchShotTracker
    from . import database
    from .security import (
        SESSION_COOKIE,
        clear_session_cookie,
        generate_csrf_token,
        hash_password,
        read_session_token,
        set_csrf_cookie,
        set_session_cookie,
        validate_csrf_token,
        verify_password,
    )
    from .storage import (
        append_analysis_record,
        append_booking,
        complete_analysis_job,
        create_analysis_job,
        fail_analysis_job,
        get_analysis_job,
        load_json_records,
        mark_analysis_job_processing,
        save_json_records,
        update_analysis_job_progress,
        update_booking_status,
    )
except ImportError:
    from config import (
        BOOKINGS_PASSWORD,
        CSRF_COOKIE,
        CV_MODEL_ARTIFACT,
        EXAMPLE_YOUTUBE_URL,
        HOST,
        MAX_FILE_SIZE,
        PORT,
        STATIC_DIR,
        TEMPLATES_DIR,
        UPLOAD_CHUNK_SIZE,
        YOUTUBE_DOWNLOAD_FORMAT,
    )
    from cv_learning import load_cv_artifact, predict_with_artifact
    from marketplace import (
        COACHES_PER_PAGE,
        COACH_TIER_REQUIREMENTS,
        coaches,
        find_coach_by_name,
        find_coach_by_slug,
    )
    from match_intelligence import build_ml_report, clamp
    from shot_detection import WholeMatchShotTracker
    import database
    from security import (
        SESSION_COOKIE,
        clear_session_cookie,
        generate_csrf_token,
        hash_password,
        read_session_token,
        set_csrf_cookie,
        set_session_cookie,
        validate_csrf_token,
        verify_password,
    )
    from storage import (
        append_analysis_record,
        append_booking,
        complete_analysis_job,
        create_analysis_job,
        fail_analysis_job,
        get_analysis_job,
        load_json_records,
        mark_analysis_job_processing,
        save_json_records,
        update_analysis_job_progress,
        update_booking_status,
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Video analysis constants ---
_DEFAULT_FPS: float = 30.0
_FRAME_SAMPLE_COUNT: int = 360
_TARGET_SHOT_FPS: int = 3
_REFERENCE_WIDTH: int = 1280
_REFERENCE_HEIGHT: int = 720

_GREEN_R_DOMINANCE: float = 1.05
_GREEN_B_DOMINANCE: float = 1.05
_GREEN_MIN: float = 0.18
_BLUE_R_DOMINANCE: float = 1.08
_BLUE_G_DOMINANCE: float = 1.02
_BLUE_MIN: float = 0.18
_TAN_RED_MIN: float = 0.38
_TAN_GREEN_MIN: float = 0.28
_TAN_BLUE_MAX: float = 0.34
_TAN_RED_DOMINANCE: float = 1.2

_WEIGHT_COURT: float = 0.34
_WEIGHT_SAMPLING: float = 0.24
_WEIGHT_SHARPNESS: float = 0.18
_WEIGHT_RESOLUTION: float = 0.14
_WEIGHT_DURATION: float = 0.10

_WEAK_COURT_SIGNAL: float = 0.35
_SOFT_FRAME_THRESHOLD: float = 0.08
_MIN_RELIABLE_FRAMES: int = 120
_MIN_RELIABLE_DURATION_S: int = 120
_RESOLUTION_FALLBACK: float = 0.35
_DURATION_FALLBACK: float = 0.45
_NET_PLAYER_RATIO_THRESHOLD: float = 0.42

# Create FastAPI app instance
app = FastAPI(title="PickleCoach AI Prototype")


def bookings_authorized(provided: str) -> bool:
    if not BOOKINGS_PASSWORD:
        return True
    return secrets.compare_digest(provided, BOOKINGS_PASSWORD)


def get_current_user(request: Request):
    user_id = read_session_token(request.cookies.get(SESSION_COOKIE, ""))
    if not user_id:
        return None
    return database.get_user_by_id(user_id)


def coach_dashboard_authorized(request: Request, provided_key: str) -> bool:
    if bookings_authorized(provided_key):
        return True
    user = get_current_user(request)
    return bool(user and user.get("role") in {"coach", "admin"})


def bookings_forbidden_response(request: Request):
    user = get_current_user(request)
    provided_key = request.query_params.get("key", "")
    if user:
        message = "Your account does not have access to that page."
        action_label = "Back to Home"
        action_url = "home"
    elif provided_key:
        message = "That admin key was not accepted. Check the key or sign in with an account that has access."
        action_label = "Log in"
        action_url = "login_form"
    else:
        message = "Sign in to see your own bookings, match history, and private reports."
        action_label = "Log in"
        action_url = "login_form"
    return templates.TemplateResponse(
        "errors/403.html",
        {
            "request": request,
            "message": message,
            "action_label": action_label,
            "action_url": action_url,
            "show_admin_key_hint": bool(BOOKINGS_PASSWORD),
            "year": datetime.now().year,
        },
        status_code=403,
    )


def analysis_job_authorized(job: dict, provided_token: str) -> bool:
    expected_token = job.get("access_token")
    if not expected_token:
        return True
    return bool(provided_token) and secrets.compare_digest(provided_token, expected_token)


def first_name(name: str) -> str:
    return (name or "there").strip().split(" ", 1)[0]


def build_home_personalization(user: dict | None):
    if not user:
        return None

    email = user.get("email", "").lower()
    name = user.get("name", "")
    role = user.get("role", "player")
    analyses = [
        record for record in reversed(load_json_records("data.json"))
        if record.get("details", {}).get("email", "").lower() == email
    ]
    if role in {"coach", "admin"}:
        bookings = [
            booking for booking in reversed(load_json_records("bookings.json"))
            if role == "admin" or booking.get("coach", "").lower() == name.lower()
        ]
    else:
        bookings = [
            booking for booking in reversed(load_json_records("bookings.json"))
            if booking.get("email", "").lower() == email
        ]

    pending_bookings = [
        booking for booking in bookings
        if booking.get("status", "pending") == "pending"
    ]
    latest_analysis = analyses[0] if analyses else None
    saved_location = database.get_user_location(user["id"])
    latest_focus = []
    if latest_analysis:
        latest_focus = latest_analysis.get("report", {}).get("ml", {}).get("focus_areas", [])[:2]

    if role in {"coach", "admin"}:
        next_step = "Review pending player requests and keep your response time sharp."
        primary_label = "Open bookings"
        primary_url = "view_bookings"
    elif latest_analysis:
        next_step = "Use your latest focus areas to choose the right coach or training plan."
        primary_label = "Find a coach"
        primary_url = "list_coaches"
    else:
        next_step = "Upload a match video to get your first personalized training read."
        primary_label = "Analyze a match"
        primary_url = "upload_form"

    return {
        "first_name": first_name(name),
        "role": role,
        "analyses": analyses[:3],
        "bookings": bookings[:3],
        "pending_bookings": len(pending_bookings),
        "latest_focus": latest_focus,
        "next_step": next_step,
        "primary_label": primary_label,
        "primary_url": primary_url,
        "search_location": (
            latest_analysis.get("details", {}).get("location", "")
            if latest_analysis else ""
        ),
        "saved_location": saved_location,
    }


def user_analysis_records(user: dict | None, records: list[dict]) -> list[dict]:
    if not user:
        return []
    if user.get("role") == "admin":
        return records
    email = user.get("email", "").lower()
    return [
        record for record in records
        if record.get("details", {}).get("email", "").lower() == email
    ]


def user_booking_records(user: dict | None, bookings: list[dict]) -> list[dict]:
    if not user:
        return []
    role = user.get("role")
    if role == "admin":
        return bookings
    if role == "coach":
        coach_name = user.get("name", "").lower()
        return [
            booking for booking in bookings
            if booking.get("coach", "").lower() == coach_name
        ]
    email = user.get("email", "").lower()
    return [
        booking for booking in bookings
        if booking.get("email", "").lower() == email
    ]


def can_manage_booking(request: Request, booking: dict | None, provided_key: str) -> bool:
    if not booking:
        return False
    if bookings_authorized(provided_key):
        return True
    user = get_current_user(request)
    if not user:
        return False
    if user.get("role") == "admin":
        return True
    return (
        user.get("role") == "coach"
        and booking.get("coach", "").lower() == user.get("name", "").lower()
    )


def upload_error_response(request: Request, message: str, status_code: int):
    csrf_token = generate_csrf_token()
    response = templates.TemplateResponse(
        "upload.html",
        {
            "request": request,
            "error": message,
            "year": datetime.now().year,
            "csrf_token": csrf_token,
            "example_youtube_url": EXAMPLE_YOUTUBE_URL,
        },
        status_code=status_code,
    )
    set_csrf_cookie(response, csrf_token)
    return response


@app.middleware("http")
async def disable_static_cache(request: Request, call_next):
    """Keep static assets fresh while iterating on the prototype."""
    if request.url.path.startswith("/static/"):
        request.scope["headers"] = [
            (key, value)
            for key, value in request.scope["headers"]
            if key not in {b"if-none-match", b"if-modified-since"}
        ]

    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# Mount static files directory (CSS)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
favicon_path = STATIC_DIR / "favicon.svg"

# Set up Jinja2 templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.globals["current_user"] = get_current_user


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path, media_type="image/svg+xml")


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return templates.TemplateResponse(
        "errors/404.html",
        {"request": request, "year": datetime.now().year},
        status_code=404,
    )


@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    logger.error("Unhandled 500: %s", exc)
    return templates.TemplateResponse(
        "errors/500.html",
        {"request": request, "year": datetime.now().year},
        status_code=500,
    )


_CV_ARTIFACT_CACHE = {"mtime": None, "artifact": None, "error": None}

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def get_cv_artifact():
    """Load the optional trained CV artifact, reusing it until the file changes."""
    try:
        mtime = os.path.getmtime(CV_MODEL_ARTIFACT)
    except OSError:
        _CV_ARTIFACT_CACHE.update({"mtime": None, "artifact": None, "error": None})
        return None

    if _CV_ARTIFACT_CACHE["mtime"] == mtime:
        return _CV_ARTIFACT_CACHE["artifact"]

    try:
        artifact = load_cv_artifact(CV_MODEL_ARTIFACT)
        _CV_ARTIFACT_CACHE.update({"mtime": mtime, "artifact": artifact, "error": None})
        return artifact
    except Exception as exc:
        logger.warning("Could not load trained CV model artifact: %s", exc)
        _CV_ARTIFACT_CACHE.update({"mtime": mtime, "artifact": None, "error": str(exc)})
        return None


def predict_learned_cv_scores(video_metrics: dict):
    artifact = get_cv_artifact()
    if not artifact:
        return None
    try:
        return predict_with_artifact(video_metrics, artifact).as_dict()
    except Exception as exc:
        logger.warning("Trained CV model prediction failed: %s", exc)
        return None


def validate_upload_fields(name, email, location, issues):
    if len(name) > 100:
        return "Name must be 100 characters or fewer."
    if not _EMAIL_RE.match(email):
        return "Please enter a valid email address."
    if len(location) > 100:
        return "Location must be 100 characters or fewer."
    if issues and len(issues) > 500:
        return "Issues description must be 500 characters or fewer."
    return None


def validate_booking_fields(name, email, message, preferred_date, preferred_time):
    if len(name) > 100:
        return "Name must be 100 characters or fewer."
    if not _EMAIL_RE.match(email):
        return "Please enter a valid email address."
    if message and len(message) > 500:
        return "Message must be 500 characters or fewer."
    try:
        parsed_date = datetime.strptime(preferred_date, "%Y-%m-%d").date()
    except ValueError:
        return "Preferred date must be a valid date (YYYY-MM-DD)."
    if parsed_date < date.today():
        return "Preferred date cannot be in the past."
    try:
        datetime.strptime(preferred_time, "%H:%M")
    except ValueError:
        return "Preferred time must be a valid time (HH:MM)."
    return None


def validate_booking_slot(selected_coach: dict, preferred_slot: str):
    if preferred_slot and preferred_slot not in selected_coach.get("availability", []):
        return "Selected slot is not available for this coach."
    return None


def validate_auth_fields(name: str, email: str, password: str, role: str):
    if len(name) > 100:
        return "Name must be 100 characters or fewer."
    if not _EMAIL_RE.match(email):
        return "Please enter a valid email address."
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if role not in {"player", "coach"}:
        return "Please choose a valid account type."
    return None


def validate_location_payload(latitude, longitude, accuracy):
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        return None, None, None, "Location coordinates were not valid."
    if not -90 <= lat <= 90:
        return None, None, None, "Latitude must be between -90 and 90."
    if not -180 <= lon <= 180:
        return None, None, None, "Longitude must be between -180 and 180."
    try:
        acc = float(accuracy) if accuracy is not None else None
    except (TypeError, ValueError):
        return None, None, None, "Location accuracy was not valid."
    if acc is not None and (acc < 0 or acc > 100000):
        return None, None, None, "Location accuracy was outside the accepted range."
    return lat, lon, acc, None


def extract_youtube_id(url: str):
    match = re.search(r"(?:youtu\.be/|v=|embed/|shorts/)([A-Za-z0-9_-]{11})", url)
    return match.group(1) if match else None


def download_youtube_video(youtube_url: str, download_dir: str):
    options = {
        "format": YOUTUBE_DOWNLOAD_FORMAT,
        "outtmpl": os.path.join(download_dir, "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(youtube_url, download=True)
            prepared = downloader.prepare_filename(info)
    except Exception as exc:
        raise RuntimeError(f"Could not access or download the YouTube video: {exc}") from exc

    candidate_paths = []
    for download in info.get("requested_downloads", []):
        if download.get("filepath"):
            candidate_paths.append(download["filepath"])
    candidate_paths.extend([
        prepared,
        os.path.splitext(prepared)[0] + ".mp4",
    ])
    candidate_paths.extend(
        os.path.join(download_dir, name)
        for name in os.listdir(download_dir)
        if name.lower().endswith((".mp4", ".webm", ".mkv", ".mov"))
    )

    for candidate in candidate_paths:
        if candidate and os.path.exists(candidate):
            return info, candidate

    raise RuntimeError("YouTube download finished, but no video file was produced.")


def resize_frame_for_detection(rgb: np.ndarray, max_width: int = 640) -> np.ndarray:
    """Downscale large frames for detector inference while keeping useful ball detail."""
    if rgb.shape[1] <= max_width:
        return rgb[:, :, :3]
    scale = max_width / rgb.shape[1]
    new_height = max(1, int(rgb.shape[0] * scale))
    y_idx = np.linspace(0, rgb.shape[0] - 1, new_height).astype(int)
    x_idx = np.linspace(0, rgb.shape[1] - 1, max_width).astype(int)
    return rgb[y_idx][:, x_idx, :3]


def analyze_video_pixels(video_path: str, progress_callback=None):
    try:
        reader = imageio.get_reader(video_path, "ffmpeg")
    except Exception as exc:
        raise RuntimeError("Downloaded video could not be opened for frame analysis.")
    try:
        metadata = reader.get_meta_data()
    except Exception:
        metadata = {}

    fps = float(metadata.get("fps") or _DEFAULT_FPS)
    size = metadata.get("size") or (0, 0)
    width = int(size[0] or 0)
    height = int(size[1] or 0)
    duration = float(metadata.get("duration") or 0.0)
    frame_count = int(duration * fps) if duration and fps else 0
    sample_count = _FRAME_SAMPLE_COUNT
    if frame_count > sample_count:
        stride = max(1, frame_count // sample_count)
    else:
        stride = max(1, int(fps // 2))
    shot_stride = max(1, int(fps / _TARGET_SHOT_FPS))
    if frame_count:
        shot_stride = max(shot_stride, frame_count // 5400)
    shot_tracker = WholeMatchShotTracker(width, height, fps, duration)
    shot_frames_analyzed = 0

    previous_gray = None
    previous_small = None
    motion_values = []
    motion_zones = {
        "near_court": [],
        "mid_court": [],
        "far_court": [],
        "left_lane": [],
        "middle_lane": [],
        "right_lane": [],
    }
    brightness_values = []
    sharpness_values = []
    green_ratios = []
    blue_ratios = []
    tan_ratios = []
    high_activity_frames = 0
    sampled = 0

    try:
        frame_iter = enumerate(reader)
        for frame_index, frame in frame_iter:
            need_summary_frame = sampled < sample_count and frame_index % stride == 0
            need_shot_frame = frame_index % shot_stride == 0
            if not need_summary_frame and not need_shot_frame:
                continue

            frame = np.asarray(frame)
            if frame.ndim == 2:
                rgb = np.stack([frame, frame, frame], axis=-1)
            else:
                rgb = frame[:, :, :3]
            timestamp = frame_index / fps if fps else 0.0

            if need_shot_frame:
                shot_tracker.add_frame(
                    resize_frame_for_detection(rgb),
                    frame_index,
                    timestamp,
                )
                shot_frames_analyzed += 1
                if progress_callback and shot_frames_analyzed % 90 == 0:
                    progress_callback(
                        {
                            "percent": min(82, 55 + int(shot_frames_analyzed / 5400 * 27)),
                            "phase": "Detecting shots",
                            "message": f"Analyzed {shot_frames_analyzed} full-match shot frames.",
                            "frames_analyzed": shot_frames_analyzed,
                        }
                    )

            if not need_summary_frame:
                continue

            y_idx = np.linspace(0, rgb.shape[0] - 1, 90).astype(int)
            x_idx = np.linspace(0, rgb.shape[1] - 1, 160).astype(int)
            small = rgb[y_idx][:, x_idx].astype(float) / 255.0
            red = small[:, :, 0]
            green = small[:, :, 1]
            blue = small[:, :, 2]
            gray = red * 0.299 + green * 0.587 + blue * 0.114
            brightness_values.append(float(gray.mean()))
            gradient_y, gradient_x = np.gradient(gray)
            sharpness_values.append(float((gradient_x.var() + gradient_y.var()) * 10))
            green_mask = (green > red * _GREEN_R_DOMINANCE) & (green > blue * _GREEN_B_DOMINANCE) & (green > _GREEN_MIN)
            blue_mask = (blue > red * _BLUE_R_DOMINANCE) & (blue > green * _BLUE_G_DOMINANCE) & (blue > _BLUE_MIN)
            tan_mask = (red > _TAN_RED_MIN) & (green > _TAN_GREEN_MIN) & (blue < _TAN_BLUE_MAX) & (red > blue * _TAN_RED_DOMINANCE)
            green_ratios.append(float(green_mask.mean()))
            blue_ratios.append(float(blue_mask.mean()))
            tan_ratios.append(float(tan_mask.mean()))

            if previous_gray is not None:
                diff = np.abs(gray - previous_gray)
                motion = float(diff.mean())
                motion_values.append(motion)
                row_thirds = np.array_split(diff, 3, axis=0)
                col_thirds = np.array_split(diff, 3, axis=1)
                motion_zones["far_court"].append(float(row_thirds[0].mean()))
                motion_zones["mid_court"].append(float(row_thirds[1].mean()))
                motion_zones["near_court"].append(float(row_thirds[2].mean()))
                motion_zones["left_lane"].append(float(col_thirds[0].mean()))
                motion_zones["middle_lane"].append(float(col_thirds[1].mean()))
                motion_zones["right_lane"].append(float(col_thirds[2].mean()))
            previous_gray = gray
            previous_small = small
            sampled += 1
            if progress_callback and sampled % 45 == 0:
                progress_callback(
                    {
                        "percent": min(54, 15 + int(sampled / sample_count * 39)),
                        "phase": "Reading match frames",
                        "message": f"Sampled {sampled} frames across the match.",
                        "frames_analyzed": sampled,
                    }
                )
    except Exception as exc:
        raise RuntimeError(f"Video frame analysis failed: {exc}") from exc
    finally:
        reader.close()

    if sampled < 2:
        raise RuntimeError("Not enough readable frames were found in the video.")

    avg_motion = sum(motion_values) / max(1, len(motion_values))
    motion_variance = (
        sum((value - avg_motion) ** 2 for value in motion_values) / max(1, len(motion_values))
    )
    motion_stddev = motion_variance ** 0.5
    high_activity_threshold = max(0.045, avg_motion + motion_stddev * 0.75)
    high_activity_frames = sum(1 for value in motion_values if value > high_activity_threshold)
    avg_brightness = sum(brightness_values) / len(brightness_values)
    avg_sharpness = sum(sharpness_values) / len(sharpness_values)
    green_presence = sum(green_ratios) / len(green_ratios)
    blue_presence = sum(blue_ratios) / len(blue_ratios)
    tan_presence = sum(tan_ratios) / len(tan_ratios)
    court_presence = max(green_presence, blue_presence, tan_presence)
    active_zone_scores = {
        label: (sum(values) / len(values) if values else 0.0)
        for label, values in motion_zones.items()
    }
    vertical_total = sum(
        active_zone_scores[label] for label in ("near_court", "mid_court", "far_court")
    ) or 1.0
    horizontal_total = sum(
        active_zone_scores[label] for label in ("left_lane", "middle_lane", "right_lane")
    ) or 1.0
    net_activity_ratio = clamp(active_zone_scores["mid_court"] / vertical_total)
    baseline_activity_ratio = clamp(
        (active_zone_scores["near_court"] + active_zone_scores["far_court"]) / vertical_total
    )
    lateral_balance = clamp(
        1.0 - abs(active_zone_scores["left_lane"] - active_zone_scores["right_lane"]) / horizontal_total
    )
    motion_burst_rate = high_activity_frames / max(1, len(motion_values))
    resolution_score = clamp((width * height) / (_REFERENCE_WIDTH * _REFERENCE_HEIGHT)) if width and height else _RESOLUTION_FALLBACK
    sampling_score = clamp(sampled / sample_count)
    duration_score = clamp(duration / 480) if duration else _DURATION_FALLBACK
    court_score = clamp(court_presence * 2.4)
    sharpness_score = clamp(avg_sharpness * 4.0)
    visual_confidence = clamp(
        court_score * _WEIGHT_COURT
        + sampling_score * _WEIGHT_SAMPLING
        + sharpness_score * _WEIGHT_SHARPNESS
        + resolution_score * _WEIGHT_RESOLUTION
        + duration_score * _WEIGHT_DURATION
    )
    quality_notes = []
    if court_score < _WEAK_COURT_SIGNAL:
        quality_notes.append("Court-color signal is weak, so court-position estimates may be noisy.")
    if avg_sharpness < _SOFT_FRAME_THRESHOLD:
        quality_notes.append("The sampled frames look soft or compressed.")
    if sampled < _MIN_RELIABLE_FRAMES:
        quality_notes.append("Only a small number of frames were readable.")
    if duration and duration < _MIN_RELIABLE_DURATION_S:
        quality_notes.append("Short clips produce less reliable match-level trends.")
    if not quality_notes:
        quality_notes.append("Frame coverage and court signal are usable for a prototype coaching read.")
    dominant_court_color = max(
        [("green", green_presence), ("blue", blue_presence), ("tan", tan_presence)],
        key=lambda x: x[1],
    )[0]

    shot_detection = shot_tracker.finish()

    return {
        "cv_model": "Frame-difference motion model + court-color segmentation",
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": round(duration, 1),
        "sampled_frames": sampled,
        "sampling_stride": stride,
        "avg_motion": round(avg_motion, 4),
        "motion_variance": round(motion_variance, 5),
        "motion_burst_threshold": round(high_activity_threshold, 4),
        "motion_burst_rate": round(motion_burst_rate, 4),
        "avg_brightness": round(avg_brightness, 4),
        "avg_sharpness": round(avg_sharpness, 4),
        "court_presence": round(court_presence, 4),
        "visual_confidence": round(visual_confidence, 2),
        "signal_quality": score_status(visual_confidence),
        "quality_notes": quality_notes,
        "dominant_court_color": dominant_court_color,
        "court_color_presence": {
            "green": round(green_presence, 4),
            "blue": round(blue_presence, 4),
            "tan": round(tan_presence, 4),
        },
        "net_activity_ratio": round(net_activity_ratio, 4),
        "baseline_activity_ratio": round(baseline_activity_ratio, 4),
        "lateral_balance": round(lateral_balance, 4),
        "activity_zones": {
            label: round(value, 4)
            for label, value in active_zone_scores.items()
        },
        "shot_detection": shot_detection,
    }


def build_advanced_report_sections(report: dict, details: dict, features: dict):
    duration = float(report.get("video_metrics", {}).get("duration_seconds") or 0)
    duration = max(duration, 180.0)
    rallies = max(1, int(report.get("rallies", 1)))
    winners = int(report.get("winners", 0))
    errors = int(report.get("unforced_errors", 0))
    net_ratio = float(report.get("net_ratio", 0.0))
    tempo = float(features.get("tempo_score", 0.5))
    consistency = float(features.get("consistency_score", 0.5))
    pressure = float(features.get("pressure_score", 0.5))
    focus = report.get("ml", {}).get("focus_areas", [])
    focus_label = focus[0].get("label", "Footwork") if focus else "Footwork"

    thirds = [
        ("Opening pattern", 0.0, duration * 0.33, clamp(tempo * 0.82 + consistency * 0.18)),
        ("Mid-match pressure", duration * 0.33, duration * 0.66, clamp(pressure * 0.78 + tempo * 0.22)),
        ("Closing execution", duration * 0.66, duration, clamp(consistency * 0.72 + pressure * 0.28)),
    ]
    timeline = []
    for label, start, end, score in thirds:
        timeline.append({
            "label": label,
            "start_seconds": round(start),
            "end_seconds": round(end),
            "score": round(score, 2),
            "summary": timeline_summary(label, score, focus_label),
        })

    segment_count = min(6, max(3, rallies // 4))
    rally_segments = []
    for index in range(segment_count):
        start = duration * index / segment_count
        end = duration * (index + 1) / segment_count
        segment_pressure = clamp(pressure + ((index % 3) - 1) * 0.06)
        segment_consistency = clamp(consistency - (index % 2) * 0.04 + (0.03 if index == segment_count - 1 else 0))
        rally_segments.append({
            "label": f"Rally block {index + 1}",
            "start_seconds": round(start),
            "end_seconds": round(end),
            "rallies": max(1, rallies // segment_count + (1 if index < rallies % segment_count else 0)),
            "pressure_score": round(segment_pressure, 2),
            "consistency_score": round(segment_consistency, 2),
            "note": rally_note(segment_pressure, segment_consistency, focus_label),
        })

    strengths = []
    if winners >= errors:
        strengths.append("Winner creation is keeping pace with error volume.")
    if net_ratio >= _NET_PLAYER_RATIO_THRESHOLD:
        strengths.append("Net presence is strong enough to shape rallies.")
    if consistency >= 0.68:
        strengths.append("Consistency profile supports longer point construction.")
    if not strengths:
        strengths.append("Baseline movement gives a stable base for targeted improvement.")

    risks = []
    if errors > winners:
        risks.append("Unforced errors are outpacing winning shots.")
    if net_ratio < 0.34:
        risks.append("Net-zone activity is low; court position may be too deep.")
    if pressure < 0.5:
        risks.append("Pressure moments need cleaner first-step and reset habits.")
    if not risks:
        risks.append("Main risk is sustaining this profile against faster opponents.")

    return {
        "performance_bands": [
            {"label": "Tempo", "value": round(tempo, 2), "status": score_status(tempo)},
            {"label": "Consistency", "value": round(consistency, 2), "status": score_status(consistency)},
            {"label": "Pressure", "value": round(pressure, 2), "status": score_status(pressure)},
            {"label": "Net Control", "value": round(net_ratio, 2), "status": score_status(net_ratio)},
        ],
        "timeline": timeline,
        "rally_segments": rally_segments,
        "coaching_summary": {
            "strengths": strengths[:3],
            "risks": risks[:3],
            "next_session_plan": [
                f"Warm up with 8 minutes of {focus_label.lower()} pattern reps.",
                "Run two pressure games to 7 with serve-return constraints.",
                "Finish with filmed points and compare net position between blocks.",
            ],
        },
    }


def score_status(value: float) -> str:
    if value >= 0.7:
        return "strong"
    if value >= 0.5:
        return "steady"
    return "needs work"


def analysis_confidence_summary(value: float) -> str:
    if value >= 0.7:
        return "Good enough for directional coaching insights."
    if value >= 0.5:
        return "Usable, but treat detailed counts as estimates."
    return "Low-confidence video signal; use the report as a rough starting point."


def timeline_summary(label: str, score: float, focus_label: str) -> str:
    if score >= 0.7:
        return f"{label} is a strength; keep the same pattern under higher pace."
    if score >= 0.5:
        return f"{label} is workable, with {focus_label.lower()} as the clearest upgrade."
    return f"{label} needs calmer resets and simpler {focus_label.lower()} choices."


def rally_note(pressure: float, consistency: float, focus_label: str) -> str:
    if pressure >= 0.65 and consistency >= 0.65:
        return "High-value block: repeat these patterns in training."
    if consistency < 0.52:
        return f"Stability dips here; simplify targets and emphasize {focus_label.lower()}."
    if pressure < 0.52:
        return "Pressure response drops; add constraint games for this phase."
    return "Solid block with room to raise first-ball aggression."


def build_report_from_video_analysis(
    video_path: str,
    details: dict,
    source_url: str | None = None,
    youtube_info: dict | None = None,
    progress_callback=None,
):
    total_bytes = os.path.getsize(video_path)
    if progress_callback:
        progress_callback(
            {
                "percent": 12,
                "phase": "Preparing model",
                "message": "Loading full-match frame sampler and shot detector.",
                "frames_analyzed": 0,
            }
        )
    video_metrics = analyze_video_pixels(video_path, progress_callback=progress_callback)
    learned_scores = predict_learned_cv_scores(video_metrics)
    duration_min = max(video_metrics["duration_seconds"] / 60.0, 0.1)
    motion_burst_rate = video_metrics.get("motion_burst_rate", 0.0)
    visual_confidence = video_metrics.get("visual_confidence", 0.5)
    heuristic_tempo = clamp(
        video_metrics["avg_motion"] * 4.8
        + video_metrics["motion_variance"] * 18
        + motion_burst_rate * 0.22
    )
    heuristic_consistency = clamp(
        0.88
        - video_metrics["motion_variance"] * 22
        - motion_burst_rate * 0.12
        + video_metrics["avg_sharpness"] * 0.05
    )
    heuristic_pressure = clamp(
        0.34
        + heuristic_tempo * 0.38
        + video_metrics["court_presence"] * 0.18
        + video_metrics.get("net_activity_ratio", 0.0) * 0.08
    )
    tempo = heuristic_tempo
    consistency = heuristic_consistency
    pressure = heuristic_pressure
    if learned_scores:
        confidence = clamp(learned_scores.get("confidence", 0.0))
        blend = clamp(0.25 + confidence * 0.45, 0.25, 0.7)
        tempo = clamp(heuristic_tempo * (1 - blend) + learned_scores["tempo_score"] * blend)
        consistency = clamp(
            heuristic_consistency * (1 - blend)
            + learned_scores["consistency_score"] * blend
        )
        pressure = clamp(
            heuristic_pressure * (1 - blend)
            + learned_scores["pressure_score"] * blend
        )
    net_activity = video_metrics.get("net_activity_ratio", 0.0)
    baseline_activity = video_metrics.get("baseline_activity_ratio", 0.0)
    lateral_balance = video_metrics.get("lateral_balance", 0.5)

    estimated_rallies = max(4, int(duration_min * (3.2 + tempo * 5.2 + motion_burst_rate * 2.2)))
    estimated_errors = max(
        1,
        int((1 - consistency) * 13 + video_metrics["motion_variance"] * 65 + (1 - lateral_balance) * 3),
    )
    estimated_winners = max(1, int(tempo * 7 + pressure * 3 + net_activity * 2))
    confidence_adjustment = clamp(0.72 + visual_confidence * 0.28, 0.72, 1.0)

    report = {
        "rallies": max(4, int(estimated_rallies * confidence_adjustment)),
        "unforced_errors": max(
            1,
            int(estimated_errors * confidence_adjustment),
        ),
        "winners": max(1, int(estimated_winners * confidence_adjustment)),
        "net_ratio": round(clamp(0.22 + pressure * 0.38 + net_activity * 0.24 - baseline_activity * 0.08), 2),
        "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "recommendations": [],
        "video_metrics": video_metrics,
        "shot_detection": video_metrics.get("shot_detection", {}),
        "confidence": {
            "score": round(visual_confidence, 2),
            "label": score_status(visual_confidence),
            "summary": analysis_confidence_summary(visual_confidence),
            "notes": video_metrics.get("quality_notes", []),
        },
        "metric_disclaimer": "Rallies, winners, and errors are frame-signal estimates, not referee-grade counts.",
    }
    if source_url:
        report["source_url"] = source_url
    if youtube_info:
        report["youtube"] = {
            "id": youtube_info.get("id"),
            "title": youtube_info.get("title"),
            "duration": youtube_info.get("duration"),
            "uploader": youtube_info.get("uploader"),
            "webpage_url": youtube_info.get("webpage_url"),
        }

    details["video_metrics"] = video_metrics
    details["shot_detection"] = video_metrics.get("shot_detection", {})
    if learned_scores:
        details["vision_learning"] = learned_scores
    ml_report = build_ml_report(total_bytes, details, report)
    ml_report["analysis_source"] = (
        "Full-video sampled YouTube frames with signal-quality gating"
        if source_url
        else "Full-video sampled uploaded frames with signal-quality gating"
    )
    if learned_scores:
        ml_report["vision_learning"] = learned_scores
    elif _CV_ARTIFACT_CACHE.get("error"):
        ml_report["vision_learning_error"] = _CV_ARTIFACT_CACHE["error"]
    report["ml"] = ml_report
    report["advanced"] = build_advanced_report_sections(report, details, ml_report["features"])
    if progress_callback:
        progress_callback(
            {
                "percent": 92,
                "phase": "Building report",
                "message": "Preparing full-match timeline and coaching recommendations.",
                "frames_analyzed": video_metrics.get("sampled_frames", 0),
            }
        )
    report["recommendations"] = [
        f"Prioritize {item['label']} drills this week."
        for item in ml_report["focus_areas"]
    ]
    if video_metrics.get("motion_burst_rate", 0.0) > 0.28:
        report["recommendations"].append("Add split-step recovery work for the high-motion bursts detected in the footage.")
    if video_metrics.get("net_activity_ratio", 0.0) < 0.24:
        report["recommendations"].append("Work on controlled kitchen-line transitions; the model saw most activity away from the net zone.")
    if video_metrics.get("lateral_balance", 1.0) < 0.68:
        report["recommendations"].append("Balance court coverage with cross-step recovery drills; movement was heavier on one side.")
    return report


def process_upload_analysis_job(job_id: str) -> None:
    """Run upload analysis after the response has handed the user a job page."""
    job = get_analysis_job(job_id)
    if not job:
        logger.warning("Analysis job %s was not found.", job_id)
        return

    filepath = job["upload_path"]
    details = dict(job["details"])
    mark_analysis_job_processing(job_id)
    update_analysis_job_progress(
        job_id,
        {
            "percent": 8,
            "phase": "Starting",
            "message": "Preparing full-match analysis.",
            "frames_analyzed": 0,
        },
    )
    def progress(payload: dict) -> None:
        update_analysis_job_progress(job_id, payload)

    try:
        report = build_report_from_video_analysis(filepath, details, progress_callback=progress)
        record = {
            "timestamp": datetime.now().isoformat(),
            "details": details,
            "report": report,
        }
        append_analysis_record(record)
        complete_analysis_job(job_id, record)
        logger.info("Upload analysis job completed: id=%s file=%s", job_id, job["filename"])
    except RuntimeError as exc:
        fail_analysis_job(job_id, str(exc))
        logger.warning("Upload analysis job failed: id=%s error=%s", job_id, exc)
    except Exception:
        fail_analysis_job(job_id, "Unexpected analysis failure. Please try another video.")
        logger.exception("Unexpected upload analysis job failure: id=%s", job_id)
    finally:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info("Deleted uploaded file after analysis job: %s", job["filename"])
        except OSError as exc:
            logger.warning("Could not delete uploaded file %s: %s", job["filename"], exc)


def build_youtube_details(youtube_url: str, youtube_info: dict):
    video_id = youtube_info.get("id") or extract_youtube_id(youtube_url)
    details = {
        "name": "YouTube Pro Singles Example",
        "email": "example@picklecoach.local",
        "location": "Gold Coast",
        "skill_level": "Pro",
        "match_type": "Singles",
        "opponent_level": "Pro",
        "issues": "top men's pro singles: explosive first-step movement, aggressive serve-plus-one patterns, fast resets, and high-pressure net transitions",
        "filename": f"youtube-{video_id}",
        "source": "YouTube example",
        "video_id": video_id,
        "competition_level": "Pro",
    }
    if youtube_info.get("title"):
        details["video_title"] = youtube_info["title"]
    return details


def analyze_youtube_url(youtube_url: str):
    video_id = extract_youtube_id(youtube_url)
    if not video_id:
        return None

    with tempfile.TemporaryDirectory(prefix="picklecoach-youtube-") as download_dir:
        youtube_info, video_path = download_youtube_video(youtube_url, download_dir)
        details = build_youtube_details(youtube_url, youtube_info)
        report = build_report_from_video_analysis(
            video_path,
            details,
            source_url=youtube_url,
            youtube_info=youtube_info,
        )
    return details, report, details["filename"]


def process_youtube_analysis_job(job_id: str) -> None:
    """Download and analyze a YouTube match while updating live job progress."""
    job = get_analysis_job(job_id)
    if not job:
        logger.warning("YouTube analysis job %s was not found.", job_id)
        return

    youtube_url = job["details"].get("youtube_url", "")
    mark_analysis_job_processing(job_id)
    update_analysis_job_progress(
        job_id,
        {
            "percent": 5,
            "phase": "Downloading",
            "message": "Fetching YouTube match video.",
            "frames_analyzed": 0,
        },
    )
    def progress(payload: dict) -> None:
        update_analysis_job_progress(job_id, payload)

    try:
        video_id = extract_youtube_id(youtube_url)
        if not video_id:
            raise RuntimeError("Please enter a valid YouTube video URL.")

        with tempfile.TemporaryDirectory(prefix="picklecoach-youtube-") as download_dir:
            youtube_info, video_path = download_youtube_video(youtube_url, download_dir)
            progress(
                {
                    "percent": 10,
                    "phase": "Downloaded",
                    "message": "Video downloaded. Starting full-match frame analysis.",
                    "frames_analyzed": 0,
                }
            )
            details = build_youtube_details(youtube_url, youtube_info)
            report = build_report_from_video_analysis(
                video_path,
                details,
                source_url=youtube_url,
                youtube_info=youtube_info,
                progress_callback=progress,
            )
            record = {
                "timestamp": datetime.now().isoformat(),
                "details": details,
                "report": report,
            }
            append_analysis_record(record)
            complete_analysis_job(job_id, record)
            logger.info("YouTube analysis job completed: id=%s video=%s", job_id, video_id)
    except RuntimeError as exc:
        fail_analysis_job(job_id, str(exc))
        logger.warning("YouTube analysis job failed: id=%s error=%s", job_id, exc)
    except Exception:
        fail_analysis_job(job_id, "Unexpected YouTube analysis failure. Please try another video.")
        logger.exception("Unexpected YouTube analysis job failure: id=%s", job_id)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page."""
    user = get_current_user(request)
    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "user": user,
            "personalization": build_home_personalization(user),
            "year": datetime.now().year,
        },
    )


@app.get("/signup", response_class=HTMLResponse)
async def signup_form(request: Request):
    csrf_token = generate_csrf_token()
    response = templates.TemplateResponse(
        "auth/signup.html",
        {"request": request, "csrf_token": csrf_token, "year": datetime.now().year},
    )
    set_csrf_cookie(response, csrf_token)
    return response


@app.post("/signup", response_class=HTMLResponse)
async def signup(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form("player"),
):
    form_data = await request.form()
    form_csrf = form_data.get("csrf_token", "")
    cookie_csrf = request.cookies.get(CSRF_COOKIE, "")
    if not validate_csrf_token(cookie_csrf, form_csrf):
        return templates.TemplateResponse(
            "auth/signup.html",
            {
                "request": request,
                "error": "Invalid or expired form token. Please reload and try again.",
                "csrf_token": generate_csrf_token(),
                "year": datetime.now().year,
            },
            status_code=403,
        )

    field_error = validate_auth_fields(name, email, password, role)
    if field_error:
        csrf_token = generate_csrf_token()
        response = templates.TemplateResponse(
            "auth/signup.html",
            {
                "request": request,
                "error": field_error,
                "csrf_token": csrf_token,
                "name": name,
                "email": email,
                "role": role,
                "year": datetime.now().year,
            },
            status_code=422,
        )
        set_csrf_cookie(response, csrf_token)
        return response

    if database.get_user_by_email(email):
        csrf_token = generate_csrf_token()
        response = templates.TemplateResponse(
            "auth/signup.html",
            {
                "request": request,
                "error": "An account already exists for that email.",
                "csrf_token": csrf_token,
                "name": name,
                "email": email,
                "role": role,
                "year": datetime.now().year,
            },
            status_code=409,
        )
        set_csrf_cookie(response, csrf_token)
        return response

    user = database.create_user({
        "id": uuid.uuid4().hex,
        "name": name,
        "email": email,
        "role": role,
        "password_hash": hash_password(password),
        "created_at": datetime.now().isoformat(),
    })
    response = RedirectResponse(url="/", status_code=303)
    set_session_cookie(response, user["id"])
    return response


@app.get("/login", response_class=HTMLResponse)
async def login_form(request: Request):
    csrf_token = generate_csrf_token()
    response = templates.TemplateResponse(
        "auth/login.html",
        {"request": request, "csrf_token": csrf_token, "year": datetime.now().year},
    )
    set_csrf_cookie(response, csrf_token)
    return response


@app.post("/login", response_class=HTMLResponse)
async def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
):
    form_data = await request.form()
    form_csrf = form_data.get("csrf_token", "")
    cookie_csrf = request.cookies.get(CSRF_COOKIE, "")
    if not validate_csrf_token(cookie_csrf, form_csrf):
        return templates.TemplateResponse(
            "auth/login.html",
            {
                "request": request,
                "error": "Invalid or expired form token. Please reload and try again.",
                "csrf_token": generate_csrf_token(),
                "email": email,
                "year": datetime.now().year,
            },
            status_code=403,
        )

    user = database.get_user_by_email(email)
    if not user or not verify_password(password, user["password_hash"]):
        csrf_token = generate_csrf_token()
        response = templates.TemplateResponse(
            "auth/login.html",
            {
                "request": request,
                "error": "Email or password is incorrect.",
                "csrf_token": csrf_token,
                "email": email,
                "year": datetime.now().year,
            },
            status_code=401,
        )
        set_csrf_cookie(response, csrf_token)
        return response

    database.mark_user_login(user["id"])
    response = RedirectResponse(url="/", status_code=303)
    set_session_cookie(response, user["id"])
    return response


@app.post("/logout")
async def logout():
    response = RedirectResponse(url="/", status_code=303)
    clear_session_cookie(response)
    return response


@app.get("/location", response_class=HTMLResponse)
async def location_settings(request: Request):
    user = get_current_user(request)
    if not user:
        return bookings_forbidden_response(request)
    csrf_token = generate_csrf_token()
    response = templates.TemplateResponse(
        "location.html",
        {
            "request": request,
            "location": database.get_user_location(user["id"]),
            "location_config": {
                "csrfToken": csrf_token,
                "updateUrl": str(request.url_for("update_my_location")),
                "clearUrl": str(request.url_for("clear_my_location")),
            },
            "csrf_token": csrf_token,
            "year": datetime.now().year,
        },
    )
    set_csrf_cookie(response, csrf_token)
    return response


@app.get("/api/location")
async def get_my_location(request: Request):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"error": "Sign in to manage location sharing."}, status_code=401)
    return {"location": database.get_user_location(user["id"])}


@app.post("/api/location")
async def update_my_location(request: Request):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"error": "Sign in to manage location sharing."}, status_code=401)
    if not validate_csrf_token(
        request.cookies.get(CSRF_COOKIE, ""),
        request.headers.get("x-csrf-token", ""),
    ):
        return JSONResponse({"error": "Invalid or expired form token. Please reload and try again."}, status_code=403)

    try:
        payload = await request.json()
    except ValueError:
        return JSONResponse({"error": "Location request body was not valid JSON."}, status_code=400)

    latitude, longitude, accuracy, error = validate_location_payload(
        payload.get("latitude"),
        payload.get("longitude"),
        payload.get("accuracy"),
    )
    if error:
        return JSONResponse({"error": error}, status_code=422)

    location = database.upsert_user_location(
        user["id"],
        latitude,
        longitude,
        accuracy,
        "browser",
    )
    return {"location": location}


@app.delete("/api/location")
async def clear_my_location(request: Request):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"error": "Sign in to manage location sharing."}, status_code=401)
    if not validate_csrf_token(
        request.cookies.get(CSRF_COOKIE, ""),
        request.headers.get("x-csrf-token", ""),
    ):
        return JSONResponse({"error": "Invalid or expired form token. Please reload and try again."}, status_code=403)
    database.delete_user_location(user["id"])
    return {"location": None}


@app.get("/upload", response_class=HTMLResponse)
async def upload_form(request: Request):
    """Display the upload form."""
    csrf_token = generate_csrf_token()
    response = templates.TemplateResponse(
        "upload.html",
        {
            "request": request,
            "year": datetime.now().year,
            "csrf_token": csrf_token,
            "example_youtube_url": EXAMPLE_YOUTUBE_URL,
        },
    )
    set_csrf_cookie(response, csrf_token)
    return response


@app.post("/upload", response_class=HTMLResponse)
async def upload_file(
    request: Request,
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    email: str = Form(...),
    location: str = Form(...),
    skill_level: str = Form(...),
    match_type: str = Form(...),
    opponent_level: str = Form(...),
    issues: str = Form(None),
    file: UploadFile = File(...),
):
    """Handle file upload, player details and match details, then analyze video frames."""
    # Validate CSRF
    form_data = await request.form()
    form_csrf = form_data.get("csrf_token", "")
    cookie_csrf = request.cookies.get(CSRF_COOKIE, "")
    if not validate_csrf_token(cookie_csrf, form_csrf):
        return upload_error_response(
            request,
            "Invalid or expired form token. Please reload and try again.",
            403,
        )

    # Validate form fields
    field_error = validate_upload_fields(name, email, location, issues)
    if field_error:
        return upload_error_response(request, field_error, 422)

    original_filename = os.path.basename(file.filename or "upload.mp4")
    _, extension = os.path.splitext(original_filename)
    filename = f"{uuid.uuid4().hex}{extension.lower() or '.mp4'}"

    upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    filepath = os.path.join(upload_dir, filename)

    total_bytes = 0
    header = b""
    file_too_large = False
    invalid_mp4 = False
    try:
        with open(filepath, "wb") as f_out:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE:
                    logger.warning("Upload rejected: file too large (%d bytes)", total_bytes)
                    file_too_large = True
                    break
                if len(header) < 8:
                    header = (header + chunk)[:8]
                    if len(header) >= 8 and header[4:8] != b"ftyp":
                        logger.warning("Upload rejected: invalid MP4 magic bytes for '%s'", file.filename)
                        invalid_mp4 = True
                        break
                f_out.write(chunk)
    finally:
        await file.close()

    if file_too_large or invalid_mp4:
        try:
            os.remove(filepath)
        except OSError as exc:
            logger.warning("Could not delete rejected upload %s: %s", filename, exc)
    if file_too_large:
        return upload_error_response(
            request,
            "File exceeds the 100 MB size limit.",
            400,
        )
    if invalid_mp4:
        return upload_error_response(
            request,
            "The uploaded file does not appear to be a valid MP4.",
            400,
        )

    details = {
        "name": name,
        "email": email,
        "location": location,
        "skill_level": skill_level,
        "match_type": match_type,
        "opponent_level": opponent_level,
        "issues": issues or "",
        "filename": filename,
    }
    job_id = uuid.uuid4().hex
    job = create_analysis_job(job_id, filename, filepath, details)
    background_tasks.add_task(process_upload_analysis_job, job_id)
    logger.info("Upload analysis job queued: id=%s file=%s", job_id, filename)

    return templates.TemplateResponse(
        "analysis_status.html",
        {
            "request": request,
            "job": job,
            "year": datetime.now().year,
        },
        status_code=202,
    )

@app.get("/analysis/jobs/{job_id}", response_class=HTMLResponse)
async def analysis_job_status(request: Request, job_id: str):
    """Show progress for an uploaded-video analysis job."""
    job = get_analysis_job(job_id)
    if not job:
        return templates.TemplateResponse(
            "errors/404.html",
            {"request": request, "year": datetime.now().year},
            status_code=404,
        )
    if not analysis_job_authorized(job, request.query_params.get("token", "")):
        return bookings_forbidden_response(request)
    if job["status"] == "complete" and job.get("result"):
        result = job["result"]
        return templates.TemplateResponse(
            "analysis.html",
            {
                "request": request,
                "report": result["report"],
                "details": result["details"],
                "filename": result["details"].get("filename", job["filename"]),
                "year": datetime.now().year,
            },
        )
    return templates.TemplateResponse(
        "analysis_status.html",
        {
            "request": request,
            "job": job,
            "year": datetime.now().year,
        },
        status_code=202 if job["status"] in {"queued", "processing"} else 200,
    )


@app.get("/api/analysis/jobs/{job_id}")
async def analysis_job_api(request: Request, job_id: str):
    """Return machine-readable upload-analysis job status."""
    job = get_analysis_job(job_id)
    if not job:
        return {"error": "Analysis job not found."}
    token = request.query_params.get("token", "")
    if not analysis_job_authorized(job, token):
        return {"error": "Analysis job not found."}
    payload = {
        "id": job["id"],
        "status": job["status"],
        "filename": job["filename"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "progress": job.get("progress", {}),
    }
    if job.get("error"):
        payload["error"] = job["error"]
    if job.get("result"):
        result_url = f"/analysis/jobs/{job_id}"
        if job.get("access_token"):
            result_url += f"?token={job['access_token']}"
        payload["result_url"] = result_url
    return payload


@app.get("/example/youtube", response_class=HTMLResponse)
async def youtube_example(request: Request, background_tasks: BackgroundTasks):
    """Queue the bundled YouTube example through the live analysis flow."""
    video_id = extract_youtube_id(EXAMPLE_YOUTUBE_URL) or "example"
    job_id = uuid.uuid4().hex
    filename = f"youtube-{video_id}"
    job = create_analysis_job(
        job_id,
        filename,
        "",
        {
            "filename": filename,
            "youtube_url": EXAMPLE_YOUTUBE_URL,
            "source": "YouTube example",
        },
    )
    background_tasks.add_task(process_youtube_analysis_job, job_id)
    return templates.TemplateResponse(
        "analysis_status.html",
        {
            "request": request,
            "job": job,
            "year": datetime.now().year,
        },
        status_code=202,
    )


@app.post("/upload/youtube", response_class=HTMLResponse)
async def analyze_youtube_video(
    request: Request,
    background_tasks: BackgroundTasks,
    youtube_url: str = Form(...),
):
    """Download and analyze a YouTube URL."""
    form_data = await request.form()
    form_csrf = form_data.get("csrf_token", "")
    cookie_csrf = request.cookies.get(CSRF_COOKIE, "")
    if not validate_csrf_token(cookie_csrf, form_csrf):
        return upload_error_response(
            request,
            "Invalid or expired form token. Please reload and try again.",
            403,
        )

    video_id = extract_youtube_id(youtube_url)
    if not video_id:
        return upload_error_response(request, "Please enter a valid YouTube video URL.", 422)

    job_id = uuid.uuid4().hex
    filename = f"youtube-{video_id}"
    job = create_analysis_job(
        job_id,
        filename,
        "",
        {
            "filename": filename,
            "youtube_url": youtube_url,
            "source": "YouTube",
        },
    )
    background_tasks.add_task(process_youtube_analysis_job, job_id)
    logger.info("YouTube analysis job queued: id=%s video=%s", job_id, video_id)
    return templates.TemplateResponse(
        "analysis_status.html",
        {
            "request": request,
            "job": job,
            "year": datetime.now().year,
        },
        status_code=202,
    )


@app.get("/coaches", response_class=HTMLResponse)
async def list_coaches(request: Request):
    """Render a list of available coaches with optional filtering."""
    country_param = request.query_params.get("country")
    location_param = request.query_params.get("location")
    specialty_param = request.query_params.get("specialty")
    sort_param = request.query_params.get("sort")
    try:
        page = max(1, int(request.query_params.get("page", "1")))
    except ValueError:
        page = 1

    unique_countries = sorted({c.get("country") for c in coaches if c.get("country")})
    unique_locations = sorted({c.get("location") for c in coaches if c.get("location")})
    unique_specialties = sorted({
        specialty
        for c in coaches
        for specialty in c.get("specialties", [])
    })

    filtered_coaches = coaches
    if country_param:
        filtered_coaches = [
            c for c in filtered_coaches
            if c.get("country") and c.get("country").lower() == country_param.lower()
        ]
    if location_param:
        filtered_coaches = [
            c for c in filtered_coaches
            if c.get("location") and c.get("location").lower() == location_param.lower()
        ]
    if specialty_param:
        filtered_coaches = [
            c for c in filtered_coaches
            if specialty_param.lower() in {s.lower() for s in c.get("specialties", [])}
        ]
    if sort_param == "rating":
        filtered_coaches = sorted(filtered_coaches, key=lambda c: c.get("rating", 0), reverse=True)
    elif sort_param == "price":
        filtered_coaches = sorted(filtered_coaches, key=lambda c: c.get("price", 0))

    total_coaches = len(filtered_coaches)
    total_pages = max(1, math.ceil(total_coaches / COACHES_PER_PAGE))
    page = min(page, total_pages)
    page_start = (page - 1) * COACHES_PER_PAGE
    page_end = page_start + COACHES_PER_PAGE
    paginated_coaches = filtered_coaches[page_start:page_end]
    query_params = {
        key: value
        for key, value in {
            "country": country_param,
            "location": location_param,
            "specialty": specialty_param,
            "sort": sort_param,
        }.items()
        if value
    }

    return templates.TemplateResponse(
        "coaches.html",
        {
            "request": request,
            "coaches": paginated_coaches,
            "countries": unique_countries,
            "locations": unique_locations,
            "specialties": unique_specialties,
            "selected_country": country_param or "",
            "selected_location": location_param or "",
            "selected_specialty": specialty_param or "",
            "selected_sort": sort_param or "",
            "page": page,
            "total_pages": total_pages,
            "total_coaches": total_coaches,
            "result_start": page_start + 1 if total_coaches else 0,
            "result_end": min(page_end, total_coaches),
            "has_previous": page > 1,
            "has_next": page < total_pages,
            "previous_page_params": {**query_params, "page": page - 1},
            "next_page_params": {**query_params, "page": page + 1},
            "year": datetime.now().year,
        },
    )


@app.get("/tiers", response_class=HTMLResponse)
async def coach_tiers(request: Request):
    """Render coach tier requirements and progression rules."""
    return templates.TemplateResponse(
        "tiers.html",
        {
            "request": request,
            "tiers": COACH_TIER_REQUIREMENTS,
            "year": datetime.now().year,
        },
    )


@app.get("/coaches/{coach_slug}", response_class=HTMLResponse)
async def coach_profile(request: Request, coach_slug: str):
    """Render a detailed marketplace profile for a coach."""
    coach = find_coach_by_slug(coach_slug)
    if not coach:
        return templates.TemplateResponse(
            "coach_profile.html",
            {
                "request": request,
                "error": "We could not find that coach. Please choose a coach from the list.",
                "year": datetime.now().year,
            },
            status_code=404,
        )

    return templates.TemplateResponse(
        "coach_profile.html",
        {
            "request": request,
            "coach": coach,
            "year": datetime.now().year,
        },
    )


@app.get("/book", response_class=HTMLResponse)
async def book_coach(request: Request, coach: str = ""):
    """Display a booking request form for an existing coach."""
    if not coach:
        return templates.TemplateResponse(
            "book.html",
            {
                "request": request,
                "coaches": coaches[:6],
                "year": datetime.now().year,
            },
        )

    selected_coach = find_coach_by_name(coach)
    if not selected_coach:
        return templates.TemplateResponse(
            "book.html",
            {
                "request": request,
                "coaches": coaches[:6],
                "error": "We could not find that coach. Please choose a coach from the list.",
                "year": datetime.now().year,
            },
            status_code=404,
        )

    csrf_token = generate_csrf_token()
    response = templates.TemplateResponse(
        "book.html",
        {
            "request": request,
            "coach": selected_coach,
            "csrf_token": csrf_token,
            "year": datetime.now().year,
        },
    )
    set_csrf_cookie(response, csrf_token)
    return response


@app.post("/book", response_class=HTMLResponse)
async def submit_booking(
    request: Request,
    coach: str = Form(...),
    name: str = Form(...),
    email: str = Form(...),
    preferred_date: str = Form(...),
    preferred_time: str = Form(...),
    preferred_slot: str = Form(""),
    message: str = Form(""),
):
    """Persist a pending booking request and show confirmation."""
    # Validate CSRF
    form_data = await request.form()
    form_csrf = form_data.get("csrf_token", "")
    cookie_csrf = request.cookies.get(CSRF_COOKIE, "")
    if not validate_csrf_token(cookie_csrf, form_csrf):
        selected_coach = find_coach_by_name(coach)
        csrf_token = generate_csrf_token()
        response = templates.TemplateResponse(
            "book.html",
            {
                "request": request,
                "coach": selected_coach,
                "error": "Invalid or expired form token. Please reload and try again.",
                "csrf_token": csrf_token,
                "year": datetime.now().year,
            },
            status_code=403,
        )
        set_csrf_cookie(response, csrf_token)
        return response

    selected_coach = find_coach_by_name(coach)
    if not selected_coach:
        return templates.TemplateResponse(
            "book.html",
            {
                "request": request,
                "error": "We could not find that coach. Please choose a coach from the list.",
                "year": datetime.now().year,
            },
            status_code=404,
        )

    field_error = validate_booking_fields(name, email, message, preferred_date, preferred_time)
    if not field_error:
        field_error = validate_booking_slot(selected_coach, preferred_slot)
    if field_error:
        csrf_token = generate_csrf_token()
        response = templates.TemplateResponse(
            "book.html",
            {
                "request": request,
                "coach": selected_coach,
                "error": field_error,
                "csrf_token": csrf_token,
                "year": datetime.now().year,
            },
            status_code=422,
        )
        set_csrf_cookie(response, csrf_token)
        return response

    booking = {
        "id": uuid.uuid4().hex,
        "timestamp": datetime.now().isoformat(),
        "status": "pending",
        "coach": selected_coach["name"],
        "name": name,
        "email": email,
        "preferred_date": preferred_date,
        "preferred_time": preferred_time,
        "preferred_slot": preferred_slot,
        "message": message,
    }

    append_booking(booking)

    logger.info("Booking created: id=%s coach=%s player=%s", booking["id"], selected_coach["name"], name)

    return templates.TemplateResponse(
        "book.html",
        {
            "request": request,
            "booking": booking,
            "coach": selected_coach,
            "year": datetime.now().year,
        },
    )


@app.get("/stats", response_class=HTMLResponse)
async def view_stats(request: Request):
    """Display statistics about all uploaded matches."""
    user = get_current_user(request)
    records = user_analysis_records(user, load_json_records("data.json"))
    bookings = user_booking_records(user, load_json_records("bookings.json"))
    total_matches = len(records)
    total_bookings = len(bookings)
    pending_bookings = sum(1 for booking in bookings if booking.get("status", "pending") == "pending")
    accepted_bookings = sum(1 for booking in bookings if booking.get("status") == "accepted")
    total_unforced_errors = sum(r["report"].get("unforced_errors", 0) for r in records)
    return templates.TemplateResponse(
        "stats.html",
        {
            "request": request,
            "total_matches": total_matches,
            "total_bookings": total_bookings,
            "pending_bookings": pending_bookings,
            "accepted_bookings": accepted_bookings,
            "total_unforced_errors": total_unforced_errors,
            "year": datetime.now().year,
        },
    )


@app.get("/bookings", response_class=HTMLResponse)
async def view_bookings(request: Request):
    """Display booking requests for coaches/admins."""
    key_param = request.query_params.get("key", "")
    user = get_current_user(request)
    if not key_param and not user:
        return bookings_forbidden_response(request)
    if key_param and not bookings_authorized(key_param):
        return bookings_forbidden_response(request)

    try:
        page = max(1, int(request.query_params.get("page", "1")))
    except ValueError:
        page = 1
    per_page = 50
    all_booking_records = load_json_records("bookings.json")
    if key_param and bookings_authorized(key_param):
        scoped_bookings = all_booking_records
    else:
        scoped_bookings = user_booking_records(user, all_booking_records)
    all_bookings = list(reversed(scoped_bookings))
    total_bookings = len(all_bookings)
    total_pages = max(1, math.ceil(total_bookings / per_page))
    page = min(page, total_pages)
    page_start = (page - 1) * per_page
    page_end = page_start + per_page
    bookings = all_bookings[page_start:page_end]
    csrf_token = generate_csrf_token()
    response = templates.TemplateResponse(
        "bookings.html",
        {
            "request": request,
            "bookings": bookings,
            "page": page,
            "total_pages": total_pages,
            "total_bookings": total_bookings,
            "result_start": page_start + 1 if total_bookings else 0,
            "result_end": min(page_end, total_bookings),
            "has_previous": page > 1,
            "has_next": page < total_pages,
            "previous_page_params": {"page": page - 1, **({"key": key_param} if key_param else {})},
            "next_page_params": {"page": page + 1, **({"key": key_param} if key_param else {})},
            "csrf_token": csrf_token,
            "key_param": key_param,
            "year": datetime.now().year,
        },
    )
    set_csrf_cookie(response, csrf_token)
    return response


@app.post("/bookings/{booking_id}/accept")
async def accept_booking(request: Request, booking_id: str):
    """Mark a booking request as accepted."""
    form_data = await request.form()
    form_csrf = form_data.get("csrf_token", "")
    cookie_csrf = request.cookies.get(CSRF_COOKIE, "")
    key_param = form_data.get("key_param", "")
    redirect_url = f"/bookings?key={key_param}" if key_param else "/bookings"
    booking = next(
        (item for item in load_json_records("bookings.json") if item.get("id") == booking_id),
        None,
    )
    if not can_manage_booking(request, booking, key_param):
        return bookings_forbidden_response(request)
    if not validate_csrf_token(cookie_csrf, form_csrf):
        logger.warning("CSRF validation failed on accept_booking id=%s", booking_id)
        return RedirectResponse(url=redirect_url, status_code=303)
    update_booking_status(booking_id, "accepted")
    logger.info("Booking accepted: id=%s", booking_id)
    return RedirectResponse(url=redirect_url, status_code=303)


@app.post("/bookings/{booking_id}/decline")
async def decline_booking(request: Request, booking_id: str):
    """Mark a booking request as declined."""
    form_data = await request.form()
    form_csrf = form_data.get("csrf_token", "")
    cookie_csrf = request.cookies.get(CSRF_COOKIE, "")
    key_param = form_data.get("key_param", "")
    redirect_url = f"/bookings?key={key_param}" if key_param else "/bookings"
    booking = next(
        (item for item in load_json_records("bookings.json") if item.get("id") == booking_id),
        None,
    )
    if not can_manage_booking(request, booking, key_param):
        return bookings_forbidden_response(request)
    if not validate_csrf_token(cookie_csrf, form_csrf):
        logger.warning("CSRF validation failed on decline_booking id=%s", booking_id)
        return RedirectResponse(url=redirect_url, status_code=303)
    update_booking_status(booking_id, "declined")
    logger.info("Booking declined: id=%s", booking_id)
    return RedirectResponse(url=redirect_url, status_code=303)


@app.get("/tips")
async def get_tips():
    """Return a list of pickleball tips and drills."""
    tips = [
        "Master the kitchen rule - stay behind the non-volley zone line",
        "Practice your dink shot for consistent net play",
        "Work on your third-shot drop to regain net control",
        "Develop a strong serve to start points with confidence",
        "Improve your footwork to reach more shots",
        "Learn to volley at the net for aggressive plays",
        "Practice soft hands for better ball control",
        "Develop patience and consistency in rallies",
    ]
    return {"tips": tips}


@app.get("/api/ml/models")
async def get_ml_models():
    """Describe the prototype model stack used by upload analysis."""
    return {
        "models": [
            {
                "name": "Skill classifier",
                "type": "Bayesian-style classifier",
                "input": ["stated skill", "tempo", "consistency", "pressure"],
            },
            {
                "name": "Play-style clustering",
                "type": "Nearest-centroid clustering",
                "input": ["tempo", "consistency", "pressure"],
            },
            {
                "name": "Shot-mix estimator",
                "type": "Normalized linear scoring",
                "input": ["tempo", "consistency", "pressure"],
            },
            {
                "name": "Training-focus ranker",
                "type": "Rules plus weighted ranking",
                "input": ["reported issues", "match metrics", "match type"],
            },
            {
                "name": "Learned CV scorer",
                "type": "Optional RandomForestRegressor artifact",
                "input": ["frame motion", "court segmentation", "activity zones"],
                "artifact_loaded": get_cv_artifact() is not None,
            },
            {
                "name": "Coach recommender",
                "type": "Content-based recommender",
                "input": ["focus areas", "location", "rating", "price"],
            },
        ]
    }


@app.get("/api/deep-analysis/{analysis_id}")
async def get_deep_analysis(analysis_id: str):
    """Return deep analysis results for a stored report by filename."""
    # This endpoint is intentionally conservative: deep reports can include
    # player-identifying details, so use the HTML report/history pages instead.
    return {"error": "Use the authenticated analysis history page."}
    records = load_json_records("data.json")
    for record in records:
        if record.get("details", {}).get("filename") == analysis_id:
            deep = record.get("report", {}).get("ml", {}).get("deep_analysis")
            if deep is None:
                return {"error": "No deep analysis for this record."}
            return {"analysis_id": analysis_id, "deep_analysis": deep}
    return {"error": "Analysis not found."}


@app.get("/api/coaches/{coach_name}")
async def get_coach_details(coach_name: str):
    """Get detailed information about a specific coach."""
    coach = find_coach_by_name(coach_name)
    if coach:
        return coach
    return {"error": "Coach not found"}


@app.get("/history", response_class=HTMLResponse)
async def view_history(request: Request):
    """Display recent match analysis history."""
    user = get_current_user(request)
    if not user:
        return bookings_forbidden_response(request)
    try:
        page = max(1, int(request.query_params.get("page", "1")))
    except ValueError:
        page = 1
    per_page = 50
    all_records = list(reversed(user_analysis_records(user, load_json_records("data.json"))))
    total_records = len(all_records)
    total_pages = max(1, math.ceil(total_records / per_page))
    page = min(page, total_pages)
    page_start = (page - 1) * per_page
    page_end = page_start + per_page
    records = all_records[page_start:page_end]

    return templates.TemplateResponse(
        "history.html",
        {
            "request": request,
            "records": records,
            "page": page,
            "total_pages": total_pages,
            "total_records": total_records,
            "result_start": page_start + 1 if total_records else 0,
            "result_end": min(page_end, total_records),
            "has_previous": page > 1,
            "has_next": page < total_pages,
            "previous_page_params": {"page": page - 1},
            "next_page_params": {"page": page + 1},
            "year": datetime.now().year,
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
