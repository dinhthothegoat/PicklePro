"""Smoke tests — verify core pages return 200 and contain expected content."""
from datetime import date, timedelta
import json
from pathlib import Path
import re
import uuid
import pytest
from httpx import AsyncClient, ASGITransport

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import main
from main import app


@pytest.fixture
def transport():
    return ASGITransport(app=app)


@pytest.mark.asyncio
async def test_home(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/")
    assert r.status_code == 200
    assert "PickleCoach" in r.text
    assert "Booking dashboard" not in r.text


@pytest.mark.asyncio
async def test_developer_tools_are_hidden_from_nav(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/")
    assert r.status_code == 200
    assert "Code Reviewer" not in r.text
    assert "Prompt Instructor" not in r.text


def test_video_report_uses_signal_confidence(monkeypatch):
    artifact_dir = Path(__file__).resolve().parents[1] / "test-artifacts"
    artifact_dir.mkdir(exist_ok=True)
    video_path = artifact_dir / f"confidence-{uuid.uuid4().hex}.mp4"
    video_path.write_bytes(b"fake video bytes")
    try:
        monkeypatch.setattr(
            main,
            "analyze_video_pixels",
            lambda _path, progress_callback=None: {
                "cv_model": "test model",
                "fps": 30,
                "frame_count": 9000,
                "width": 640,
                "height": 360,
                "duration_seconds": 300,
                "sampled_frames": 80,
                "sampling_stride": 112,
                "avg_motion": 0.02,
                "motion_variance": 0.0002,
                "motion_burst_threshold": 0.045,
                "motion_burst_rate": 0.08,
                "avg_brightness": 0.4,
                "avg_sharpness": 0.04,
                "court_presence": 0.08,
                "visual_confidence": 0.31,
                "signal_quality": "needs work",
                "quality_notes": ["Court-color signal is weak."],
                "dominant_court_color": "green",
                "court_color_presence": {"green": 0.08, "blue": 0.02, "tan": 0.01},
                "net_activity_ratio": 0.2,
                "baseline_activity_ratio": 0.6,
                "lateral_balance": 0.55,
                "activity_zones": {
                    "near_court": 0.02,
                    "mid_court": 0.01,
                    "far_court": 0.02,
                    "left_lane": 0.02,
                    "middle_lane": 0.01,
                    "right_lane": 0.005,
                },
            },
        )
        monkeypatch.setattr(main, "predict_learned_cv_scores", lambda _metrics: None)

        report = main.build_report_from_video_analysis(
            str(video_path),
            {
                "name": "Test Player",
                "email": "test@example.com",
                "location": "Gold Coast",
                "skill_level": "Intermediate",
                "match_type": "Singles",
                "opponent_level": "Intermediate",
                "issues": "",
            },
        )

        assert report["confidence"]["score"] == 0.31
        assert "frame-signal estimates" in report["metric_disclaimer"]
        assert report["ml"]["features"]["visual_confidence"] == 0.31
        assert report["ml"]["skill_prediction"]["confidence_note"]
    finally:
        if video_path.exists():
            os.remove(video_path)


@pytest.mark.asyncio
async def test_coaches_list(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/coaches")
    assert r.status_code == 200
    assert "Find Coaches" in r.text or "listing" in r.text.lower()


@pytest.mark.asyncio
async def test_coaches_filter_country(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/coaches?country=Australia")
    assert r.status_code == 200
    assert "Australia" in r.text


@pytest.mark.asyncio
async def test_coach_profile_found(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/coaches/sarah-johnson")
    assert r.status_code == 200
    assert "Sarah Johnson" in r.text


@pytest.mark.asyncio
async def test_coach_profile_not_found(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/coaches/this-coach-does-not-exist-xyz")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_upload_form(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/upload")
    assert r.status_code == 200
    assert "csrf_token" in r.text
    assert 'name="file"' in r.text
    assert "pthJ1IQPqGE" in r.text


@pytest.mark.asyncio
async def test_upload_rejects_bad_email(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # First get a valid CSRF token
        get_r = await client.get("/upload")
        match = re.search(r'name="csrf_token" value="([^"]+)"', get_r.text)
        csrf = match.group(1) if match else ""
        cookie = get_r.cookies.get("csrf_token", "")

        r = await client.post(
            "/upload",
            data={
                "name": "Test Player",
                "email": "not-an-email",
                "location": "Sydney",
                "skill_level": "Beginner",
                "match_type": "Singles",
                "opponent_level": "Beginner",
                "csrf_token": csrf,
            },
            files={"file": ("test.mp4", b"\x00" * 10, "video/mp4")},
            cookies={"csrf_token": cookie},
        )
    assert r.status_code == 422
    assert "valid email" in r.text


@pytest.mark.asyncio
async def test_upload_rejects_non_mp4(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        get_r = await client.get("/upload")
        match = re.search(r'name="csrf_token" value="([^"]+)"', get_r.text)
        csrf = match.group(1) if match else ""
        cookie = get_r.cookies.get("csrf_token", "")

        r = await client.post(
            "/upload",
            data={
                "name": "Test Player",
                "email": "test@example.com",
                "location": "Sydney",
                "skill_level": "Beginner",
                "match_type": "Singles",
                "opponent_level": "Beginner",
                "csrf_token": csrf,
            },
            files={"file": ("test.txt", b"not a video file", "text/plain")},
            cookies={"csrf_token": cookie},
        )
    assert r.status_code == 400
    assert "valid MP4" in r.text


@pytest.mark.asyncio
async def test_upload_queues_background_analysis(transport, monkeypatch):
    captured = {}

    def fake_create_job(job_id, filename, upload_path, details):
        captured["upload_path"] = upload_path
        return {
            "id": job_id,
            "status": "queued",
            "created_at": "2026-04-19T10:00:00",
            "updated_at": "2026-04-19T10:00:00",
            "filename": filename,
            "upload_path": upload_path,
            "details": details,
        }

    monkeypatch.setattr(main, "create_analysis_job", fake_create_job)
    monkeypatch.setattr(main, "process_upload_analysis_job", lambda job_id: None)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        get_r = await client.get("/upload")
        match = re.search(r'name="csrf_token" value="([^"]+)"', get_r.text)
        csrf = match.group(1) if match else ""
        client.cookies.set("csrf_token", get_r.cookies.get("csrf_token", ""))

        r = await client.post(
            "/upload",
            data={
                "name": "Test Player",
                "email": "test@example.com",
                "location": "Sydney",
                "skill_level": "Beginner",
                "match_type": "Singles",
                "opponent_level": "Beginner",
                "csrf_token": csrf,
            },
            files={"file": ("test.mp4", b"\x00\x00\x00\x18ftypmp42", "video/mp4")},
        )

    try:
        assert r.status_code == 202
        assert "Video Analysis Queued" in r.text
        assert "Analysis job" in r.text
    finally:
        upload_path = captured.get("upload_path")
        if upload_path and os.path.exists(upload_path):
            os.remove(upload_path)


@pytest.mark.asyncio
async def test_book_form_requires_valid_coach(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/book?coach=nonexistent-coach-xyz")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_book_page_without_coach_shows_picker(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/book")
    assert r.status_code == 200
    assert "Choose a coach" in r.text
    assert "Sarah Johnson" in r.text


@pytest.mark.asyncio
async def test_booking_rejects_unavailable_slot(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        page = await client.get("/book?coach=Sarah%20Johnson")
        match = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
        csrf = match.group(1) if match else ""
        client.cookies.set("csrf_token", page.cookies.get("csrf_token", ""))

        r = await client.post(
            "/book",
            data={
                "coach": "Sarah Johnson",
                "name": "Slot Tester",
                "email": "slot@example.com",
                "preferred_date": (date.today() + timedelta(days=7)).isoformat(),
                "preferred_time": "10:30",
                "preferred_slot": "Sunday 3:00 AM",
                "message": "Invalid slot test",
                "csrf_token": csrf,
            },
        )
    assert r.status_code == 422
    assert "Selected slot is not available" in r.text


@pytest.mark.asyncio
async def test_booking_rejects_past_date(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        page = await client.get("/book?coach=Sarah%20Johnson")
        match = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
        csrf = match.group(1) if match else ""
        client.cookies.set("csrf_token", page.cookies.get("csrf_token", ""))

        r = await client.post(
            "/book",
            data={
                "coach": "Sarah Johnson",
                "name": "Date Tester",
                "email": "date@example.com",
                "preferred_date": (date.today() - timedelta(days=1)).isoformat(),
                "preferred_time": "10:30",
                "preferred_slot": "Mon 5:00 PM",
                "message": "Past date test",
                "csrf_token": csrf,
            },
        )
    assert r.status_code == 422
    assert "Preferred date cannot be in the past" in r.text


@pytest.mark.asyncio
async def test_booking_uses_append_storage_path(transport, monkeypatch):
    captured = []
    monkeypatch.setattr(main, "append_booking", captured.append)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        page = await client.get("/book?coach=Sarah%20Johnson")
        match = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
        csrf = match.group(1) if match else ""
        client.cookies.set("csrf_token", page.cookies.get("csrf_token", ""))

        r = await client.post(
            "/book",
            data={
                "coach": "Sarah Johnson",
                "name": "Append Tester",
                "email": "append@example.com",
                "preferred_date": (date.today() + timedelta(days=7)).isoformat(),
                "preferred_time": "10:30",
                "preferred_slot": "Mon 5:00 PM",
                "message": "Append path test",
                "csrf_token": csrf,
            },
        )
    assert r.status_code == 200
    assert "Booking Request Sent" in r.text
    assert len(captured) == 1
    assert captured[0]["email"] == "append@example.com"


@pytest.mark.asyncio
async def test_stats_page(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/stats")
    assert r.status_code == 200
    assert "Stats" in r.text
    assert "<!DOCTYPE html>" in r.text  # must be a full HTML page, not a raw string


@pytest.mark.asyncio
async def test_bookings_no_password(transport):
    """Anonymous visitors cannot see booking records even when no admin key is set."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/bookings")
    assert r.status_code == 403
    assert "Sign in to see your own bookings" in r.text
    assert "?key=your-password" not in r.text


@pytest.mark.asyncio
async def test_signup_login_logout_flow(transport):
    unique_email = f"auth-{uuid.uuid4().hex}@example.com"
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        signup_page = await client.get("/signup")
        match = re.search(r'name="csrf_token" value="([^"]+)"', signup_page.text)
        csrf = match.group(1) if match else ""
        client.cookies.set("csrf_token", signup_page.cookies.get("csrf_token", ""))

        signup_r = await client.post(
            "/signup",
            data={
                "name": "Auth Coach",
                "email": unique_email,
                "password": "strong-password",
                "role": "coach",
                "csrf_token": csrf,
            },
            follow_redirects=False,
        )
        assert signup_r.status_code == 303

        home = await client.get("/")
        assert "Auth Coach" in home.text
        assert "Coach" in home.text

        logout_r = await client.post("/logout", follow_redirects=False)
        assert logout_r.status_code == 303

        login_page = await client.get("/login")
        match = re.search(r'name="csrf_token" value="([^"]+)"', login_page.text)
        csrf = match.group(1) if match else ""
        client.cookies.set("csrf_token", login_page.cookies.get("csrf_token", ""))

        login_r = await client.post(
            "/login",
            data={
                "email": unique_email,
                "password": "strong-password",
                "csrf_token": csrf,
            },
            follow_redirects=False,
        )
        assert login_r.status_code == 303


@pytest.mark.asyncio
async def test_signed_in_home_and_forms_are_personalized(transport):
    unique_email = f"personal-{uuid.uuid4().hex}@example.com"
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        signup_page = await client.get("/signup")
        match = re.search(r'name="csrf_token" value="([^"]+)"', signup_page.text)
        csrf = match.group(1) if match else ""
        await client.post(
            "/signup",
            data={
                "name": "Personal Player",
                "email": unique_email,
                "password": "strong-password",
                "role": "player",
                "csrf_token": csrf,
            },
            follow_redirects=False,
        )

        home = await client.get("/")
        upload = await client.get("/upload")
        book = await client.get("/book?coach=Sarah%20Johnson")

    assert "Welcome back, Personal." in home.text
    assert "Your activity" in home.text
    assert f'value="{unique_email}"' in upload.text
    assert 'value="Personal Player"' in book.text


@pytest.mark.asyncio
async def test_location_tracking_requires_login_and_saves_user_location(transport):
    unique_email = f"location-{uuid.uuid4().hex}@example.com"
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        anonymous = await client.get("/location")
        assert anonymous.status_code == 403

        signup_page = await client.get("/signup")
        match = re.search(r'name="csrf_token" value="([^"]+)"', signup_page.text)
        csrf = match.group(1) if match else ""
        await client.post(
            "/signup",
            data={
                "name": "Location Player",
                "email": unique_email,
                "password": "strong-password",
                "role": "player",
                "csrf_token": csrf,
            },
            follow_redirects=False,
        )

        page = await client.get("/location")
        assert page.status_code == 200
        assert "Location sharing" in page.text
        match = re.search(r'<script id="location-config" type="application/json">\s*(.*?)\s*</script>', page.text, re.S)
        location_csrf = json.loads(match.group(1))["csrfToken"] if match else ""

        saved = await client.post(
            "/api/location",
            headers={"X-CSRF-Token": location_csrf},
            json={"latitude": -33.8688, "longitude": 151.2093, "accuracy": 20},
        )
        assert saved.status_code == 200
        assert saved.json()["location"]["latitude"] == -33.8688

        current = await client.get("/api/location")
        assert current.status_code == 200
        assert current.json()["location"]["longitude"] == 151.2093

        cleared = await client.delete(
            "/api/location",
            headers={"X-CSRF-Token": location_csrf},
        )
        assert cleared.status_code == 200
        assert cleared.json()["location"] is None


@pytest.mark.asyncio
async def test_location_api_validates_coordinates(transport):
    unique_email = f"bad-location-{uuid.uuid4().hex}@example.com"
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        signup_page = await client.get("/signup")
        match = re.search(r'name="csrf_token" value="([^"]+)"', signup_page.text)
        csrf = match.group(1) if match else ""
        await client.post(
            "/signup",
            data={
                "name": "Bad Location Player",
                "email": unique_email,
                "password": "strong-password",
                "role": "player",
                "csrf_token": csrf,
            },
            follow_redirects=False,
        )

        page = await client.get("/location")
        match = re.search(r'<script id="location-config" type="application/json">\s*(.*?)\s*</script>', page.text, re.S)
        location_csrf = json.loads(match.group(1))["csrfToken"] if match else ""
        response = await client.post(
            "/api/location",
            headers={"X-CSRF-Token": location_csrf},
            json={"latitude": 120, "longitude": 151.2093, "accuracy": 20},
        )

    assert response.status_code == 422
    assert "Latitude" in response.json()["error"]


@pytest.mark.asyncio
async def test_history_and_bookings_are_scoped_to_signed_in_user(transport, monkeypatch):
    player_email = f"scope-{uuid.uuid4().hex}@example.com"
    other_email = f"other-{uuid.uuid4().hex}@example.com"
    monkeypatch.setattr(
        main,
        "load_json_records",
        lambda filename: [
            {
                "timestamp": "2026-04-19T10:00:00",
                "details": {
                    "filename": "mine.mp4",
                    "name": "Scoped Player",
                    "email": player_email,
                    "location": "Sydney",
                    "skill_level": "Intermediate",
                    "match_type": "Singles",
                },
                "report": {"rallies": 10, "winners": 3, "unforced_errors": 2},
            },
            {
                "timestamp": "2026-04-19T11:00:00",
                "details": {
                    "filename": "other.mp4",
                    "name": "Other Player",
                    "email": other_email,
                    "location": "Perth",
                    "skill_level": "Advanced",
                    "match_type": "Singles",
                },
                "report": {"rallies": 20, "winners": 6, "unforced_errors": 4},
            },
        ] if filename == "data.json" else [
            {
                "id": "mine-booking",
                "status": "pending",
                "coach": "Sarah Johnson",
                "name": "Scoped Player",
                "email": player_email,
                "preferred_date": "2026-05-01",
                "preferred_time": "10:30",
                "preferred_slot": "Mon 5:00 PM",
                "message": "mine",
            },
            {
                "id": "other-booking",
                "status": "pending",
                "coach": "Michael Lee",
                "name": "Other Player",
                "email": other_email,
                "preferred_date": "2026-05-02",
                "preferred_time": "11:30",
                "preferred_slot": "Tue 6:00 PM",
                "message": "other",
            },
        ],
    )

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        signup_page = await client.get("/signup")
        match = re.search(r'name="csrf_token" value="([^"]+)"', signup_page.text)
        csrf = match.group(1) if match else ""
        client.cookies.set("csrf_token", signup_page.cookies.get("csrf_token", ""))
        await client.post(
            "/signup",
            data={
                "name": "Scoped Player",
                "email": player_email,
                "password": "strong-password",
                "role": "player",
                "csrf_token": csrf,
            },
            follow_redirects=False,
        )

        history = await client.get("/history")
        bookings = await client.get("/bookings")
        stats = await client.get("/stats")

    assert history.status_code == 200
    assert "Scoped Player" in history.text
    assert "Other Player" not in history.text
    assert bookings.status_code == 200
    assert "Scoped Player" in bookings.text
    assert "Other Player" not in bookings.text
    assert "Sarah Johnson" in bookings.text
    assert "Michael Lee" not in bookings.text
    assert "Stats" in stats.text
    assert "<dd>1</dd>" in stats.text


@pytest.mark.asyncio
async def test_coach_login_can_access_password_protected_bookings(transport, monkeypatch):
    monkeypatch.setattr(main, "BOOKINGS_PASSWORD", "secret")
    unique_email = f"coach-{uuid.uuid4().hex}@example.com"
    monkeypatch.setattr(
        main,
        "load_json_records",
        lambda filename: [
            {
                "id": "coach-booking",
                "status": "pending",
                "coach": "Dashboard Coach",
                "name": "Player One",
                "email": "player1@example.com",
                "preferred_date": "2026-05-01",
                "preferred_time": "10:30",
                "preferred_slot": "Mon 5:00 PM",
                "message": "for this coach",
            },
            {
                "id": "other-coach-booking",
                "status": "pending",
                "coach": "Other Coach",
                "name": "Player Two",
                "email": "player2@example.com",
                "preferred_date": "2026-05-02",
                "preferred_time": "11:30",
                "preferred_slot": "Tue 6:00 PM",
                "message": "not for this coach",
            },
        ] if filename == "bookings.json" else [],
    )
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        signup_page = await client.get("/signup")
        match = re.search(r'name="csrf_token" value="([^"]+)"', signup_page.text)
        csrf = match.group(1) if match else ""
        client.cookies.set("csrf_token", signup_page.cookies.get("csrf_token", ""))
        await client.post(
            "/signup",
            data={
                "name": "Dashboard Coach",
                "email": unique_email,
                "password": "strong-password",
                "role": "coach",
                "csrf_token": csrf,
            },
            follow_redirects=False,
        )

        r = await client.get("/bookings")
    assert r.status_code == 200
    assert "Booking Requests" in r.text
    assert "Dashboard Coach" in r.text
    assert "Other Coach" not in r.text


@pytest.mark.asyncio
async def test_booking_status_update_requires_admin_key(transport, monkeypatch):
    monkeypatch.setattr(main, "BOOKINGS_PASSWORD", "secret")
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        get_r = await client.get("/upload")
        match = re.search(r'name="csrf_token" value="([^"]+)"', get_r.text)
        csrf = match.group(1) if match else ""
        client.cookies.set("csrf_token", get_r.cookies.get("csrf_token", ""))

        r = await client.post(
            "/bookings/example-booking-id/accept",
            data={"csrf_token": csrf, "key_param": ""},
        )

    assert r.status_code == 403


@pytest.mark.asyncio
async def test_404_page(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/this-route-absolutely-does-not-exist")
    assert r.status_code == 404
    assert "Not Found" in r.text


@pytest.mark.asyncio
async def test_tips_api(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/tips")
    assert r.status_code == 200
    data = r.json()
    assert "tips" in data
    assert len(data["tips"]) > 0


@pytest.mark.asyncio
async def test_ml_models_api(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/api/ml/models")
    assert r.status_code == 200
    data = r.json()
    assert "models" in data
    assert any(model["name"] == "Coach recommender" for model in data["models"])


def fake_youtube_analysis(url):
    details = {
        "name": "YouTube Pro Singles Example",
        "email": "example@picklecoach.local",
        "location": "Gold Coast",
        "skill_level": "Pro",
        "match_type": "Singles",
        "opponent_level": "Pro",
        "issues": "top men's pro singles",
        "filename": "youtube-pthJ1IQPqGE",
        "source": "YouTube example",
        "video_id": "pthJ1IQPqGE",
        "competition_level": "Pro",
        "video_metrics": {
            "duration_seconds": 120,
            "avg_motion": 0.12,
            "motion_variance": 0.002,
            "avg_sharpness": 0.5,
            "court_presence": 0.35,
        },
    }
    report = {
        "rallies": 24,
        "unforced_errors": 3,
        "winners": 9,
        "net_ratio": 0.68,
        "analysis_time": "2026-04-17 10:00",
        "recommendations": [],
        "source_url": url,
        "video_metrics": {
            "width": 1280,
            "height": 720,
            "duration_seconds": 120,
            "sampled_frames": 180,
            "avg_motion": 0.12,
            "court_presence": 0.35,
        },
    }
    report["ml"] = main.build_ml_report(48 * 1024 * 1024, details, report)
    report["ml"]["analysis_source"] = "Downloaded YouTube video frames"
    return details, report, details["filename"]


@pytest.mark.asyncio
async def test_youtube_example_analysis_page(transport, monkeypatch):
    queued = []
    monkeypatch.setattr(main, "process_youtube_analysis_job", queued.append)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/example/youtube")
    assert r.status_code == 202
    assert "youtube-pthJ1IQPqGE" in r.text
    assert "Video Analysis Queued" in r.text
    assert "full-video sampling and shot detection" in r.text
    assert len(queued) == 1


@pytest.mark.asyncio
async def test_youtube_post_uses_real_analyzer_path(transport, monkeypatch):
    queued = []
    monkeypatch.setattr(main, "process_youtube_analysis_job", queued.append)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        get_r = await client.get("/upload")
        match = re.search(r'name="csrf_token" value="([^"]+)"', get_r.text)
        csrf = match.group(1) if match else ""
        client.cookies.set("csrf_token", get_r.cookies.get("csrf_token", ""))

        r = await client.post(
            "/upload/youtube",
            data={
                "youtube_url": "https://youtu.be/pthJ1IQPqGE?si=eVw5judspJHy5Ku4",
                "csrf_token": csrf,
            },
        )

    assert r.status_code == 202
    assert "Video Analysis Queued" in r.text
    assert "full-video sampling and shot detection" in r.text
    assert len(queued) == 1


@pytest.mark.asyncio
async def test_analysis_job_requires_access_token(transport, monkeypatch):
    job = {
        "id": "private-job",
        "status": "queued",
        "created_at": "2026-04-19T10:00:00",
        "updated_at": "2026-04-19T10:00:00",
        "filename": "private.mp4",
        "upload_path": "",
        "access_token": "secret-token",
        "details": {"filename": "private.mp4"},
        "progress": {"percent": 25, "phase": "Processing", "message": "Working", "frames_analyzed": 10},
    }
    monkeypatch.setattr(main, "get_analysis_job", lambda job_id: job if job_id == "private-job" else None)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        denied = await client.get("/analysis/jobs/private-job")
        allowed = await client.get("/analysis/jobs/private-job?token=secret-token")
        api_denied = await client.get("/api/analysis/jobs/private-job")
        api_allowed = await client.get("/api/analysis/jobs/private-job?token=secret-token")

    assert denied.status_code == 403
    assert allowed.status_code == 202
    assert api_denied.json()["error"] == "Analysis job not found."
    assert api_allowed.json()["status"] == "queued"


def test_build_ml_report_returns_predictions():
    details = {
        "skill_level": "Intermediate",
        "match_type": "Doubles",
        "opponent_level": "Advanced",
        "issues": "late footwork and weak serve returns",
        "location": "Gold Coast",
    }
    report = {
        "rallies": 12,
        "unforced_errors": 6,
        "winners": 2,
        "net_ratio": 0.42,
    }

    ml_report = main.build_ml_report(8 * 1024 * 1024, details, report)

    assert ml_report["skill_prediction"]["label"] in {"Beginner", "Intermediate", "Advanced"}
    assert ml_report["play_style"]["label"]
    assert len(ml_report["focus_areas"]) == 3
    assert len(ml_report["coach_matches"]) == 3


def test_pro_context_overrides_synthetic_skill_bucket():
    details = {
        "skill_level": "Pro",
        "match_type": "Singles",
        "opponent_level": "Pro",
        "issues": "top men's pro singles",
        "location": "Gold Coast",
        "competition_level": "Pro",
        "video_id": "pthJ1IQPqGE",
    }
    report = {
        "rallies": 24,
        "unforced_errors": 3,
        "winners": 9,
        "net_ratio": 0.68,
    }

    ml_report = main.build_ml_report(48 * 1024 * 1024, details, report)

    assert ml_report["skill_prediction"]["label"] == "Pro / Elite"
    assert ml_report["deep_analysis"]["ensemble"]["label"] == "Pro / Elite"
    assert ml_report["skill_prediction"]["raw_label"] in {"Beginner", "Intermediate", "Advanced"}


def test_csrf_validation_requires_matching_tokens():
    first = main.generate_csrf_token()
    second = main.generate_csrf_token()

    assert main.validate_csrf_token(first, first)
    assert not main.validate_csrf_token(first, second)


@pytest.mark.asyncio
async def test_tiers_page(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/tiers")
    assert r.status_code == 200
    assert "Ultra" in r.text


def test_run_deep_analysis_returns_structure():
    import deep_analysis
    features = {
        "file_size_mb": 12.5, "estimated_duration_min": 8.0,
        "tempo_score": 0.52, "consistency_score": 0.68,
        "pressure_score": 0.41, "issue_complexity": 2,
    }
    details = {
        "skill_level": "Intermediate", "match_type": "Doubles",
        "opponent_level": "Advanced", "issues": "late footwork",
    }
    result = deep_analysis.run_deep_analysis(features, details)
    assert result["ensemble"]["label"] in {"Beginner", "Intermediate", "Advanced"}
    assert 0.0 <= result["ensemble"]["confidence"] <= 1.0
    assert 0.0 <= result["ensemble"]["agreement_pct"] <= 1.0
    assert len(result["feature_vector"]) == 8
    for key in ("random_forest", "svm", "neural_net", "kmeans_segment"):
        assert key in result["models"]


def test_build_ml_report_includes_deep_analysis():
    details = {
        "skill_level": "Beginner", "match_type": "Singles",
        "opponent_level": "Beginner", "issues": "", "location": "Sydney",
    }
    report = {"rallies": 8, "unforced_errors": 3, "winners": 1, "net_ratio": 0.3}
    ml_report = main.build_ml_report(5 * 1024 * 1024, details, report)
    assert "deep_analysis" in ml_report
    da = ml_report["deep_analysis"]
    assert da["ensemble"]["label"] in {"Beginner", "Intermediate", "Advanced"}
    assert isinstance(da["feature_vector"], list)


def test_cv_learning_artifact_round_trip():
    import cv_learning
    import os
    import uuid

    metrics = {
        "width": 1280,
        "height": 720,
        "duration_seconds": 180,
        "avg_motion": 0.08,
        "motion_variance": 0.002,
        "motion_burst_rate": 0.18,
        "avg_brightness": 0.45,
        "avg_sharpness": 0.62,
        "court_presence": 0.36,
        "net_activity_ratio": 0.34,
        "baseline_activity_ratio": 0.66,
        "lateral_balance": 0.82,
    }
    samples = [{"metrics": metrics}, {"metrics": {**metrics, "avg_motion": 0.12}}]
    artifact_dir = os.path.join(os.path.dirname(__file__), "..", "test-artifacts")
    os.makedirs(artifact_dir, exist_ok=True)
    artifact_path = os.path.join(artifact_dir, f"cv-model-{uuid.uuid4().hex}.pkl")

    try:
        cv_learning.train_cv_artifact(samples, artifact_path)
        artifact = cv_learning.load_cv_artifact(artifact_path)
        prediction = cv_learning.predict_with_artifact(metrics, artifact).as_dict()

        assert prediction["training_samples"] == 2
        assert 0.0 <= prediction["tempo_score"] <= 1.0
        assert 0.0 <= prediction["consistency_score"] <= 1.0
        assert 0.0 <= prediction["pressure_score"] <= 1.0
    finally:
        if os.path.exists(artifact_path):
            os.remove(artifact_path)


@pytest.mark.asyncio
async def test_deep_analysis_api_does_not_expose_private_records(transport, monkeypatch):
    details = {
        "skill_level": "Intermediate", "match_type": "Doubles",
        "opponent_level": "Advanced", "issues": "footwork",
        "location": "Gold Coast", "filename": "test-abc123.mp4",
    }
    report = {"rallies": 10, "unforced_errors": 4, "winners": 2, "net_ratio": 0.45}
    ml_report = main.build_ml_report(8 * 1024 * 1024, details, report)
    report["ml"] = ml_report
    monkeypatch.setattr(
        main, "load_json_records",
        lambda _: [{"timestamp": "2026-01-01T00:00:00", "details": details, "report": report}],
    )
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/api/deep-analysis/test-abc123.mp4")
    assert r.status_code == 200
    data = r.json()
    assert "deep_analysis" not in data
    assert data["error"] == "Use the authenticated analysis history page."
