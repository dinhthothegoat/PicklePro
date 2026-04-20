"""Seed demo-scale PickleCoach usage data.

This creates deterministic records so the app looks like thousands of players
have used match analysis and coach booking workflows.

Example:
    python backend/seed_demo_data.py --users 4000
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import random
from typing import Any

try:
    from .marketplace import coaches
    from .storage import replace_analysis_jobs, save_json_records
except ImportError:
    from marketplace import coaches
    from storage import replace_analysis_jobs, save_json_records


FIRST_NAMES = [
    "Ava", "Noah", "Mia", "Leo", "Grace", "Ethan", "Sofia", "Jack",
    "Lily", "Kai", "Zoe", "Hugo", "Ivy", "Mason", "Ruby", "Lucas",
]

LAST_NAMES = [
    "Nguyen", "Wilson", "Tran", "Taylor", "Pham", "Brown", "Le", "Chen",
    "Singh", "Roberts", "Kelly", "Martin", "Hoang", "Walker", "Vo", "Lee",
]

LOCATIONS = [
    "Gold Coast", "Brisbane", "Sydney", "Melbourne", "Perth", "Adelaide",
    "Ho Chi Minh City", "Hanoi", "Da Nang", "Nha Trang",
]

ISSUES = [
    "late footwork and missed serve returns",
    "soft game consistency near the kitchen",
    "weak third-shot drops under pressure",
    "volley timing and reset choices",
    "court positioning in doubles rallies",
    "baseline recovery after fast exchanges",
]

FOCUS_LABELS = [
    "Footwork", "Dinks", "Serve returns", "Volleys",
    "Third-shot drops", "Doubles strategy",
]


def build_player(index: int) -> dict[str, str]:
    first = FIRST_NAMES[index % len(FIRST_NAMES)]
    last = LAST_NAMES[(index * 5) % len(LAST_NAMES)]
    return {
        "name": f"{first} {last} {index:04d}",
        "email": f"player{index:04d}@example.com",
        "location": LOCATIONS[index % len(LOCATIONS)],
    }


def build_analysis(index: int, now: datetime, rng: random.Random) -> dict[str, Any]:
    player = build_player(index)
    timestamp = now - timedelta(hours=index * 2, minutes=index % 47)
    skill_level = ["Beginner", "Intermediate", "Advanced"][index % 3]
    match_type = "Doubles" if index % 2 == 0 else "Singles"
    opponent_level = ["Beginner", "Intermediate", "Advanced"][(index + 1) % 3]
    issue = ISSUES[index % len(ISSUES)]
    filename = f"demo-match-{index:04d}.mp4"

    tempo = round(0.35 + (index % 50) / 100, 2)
    consistency = round(0.48 + ((index * 3) % 42) / 100, 2)
    pressure = round(0.38 + ((index * 7) % 45) / 100, 2)
    unforced_errors = max(1, int((1 - consistency) * 18) + (index % 4))
    winners = max(1, int(tempo * 10 + pressure * 4))
    rallies = 8 + (index % 28)
    net_ratio = round(0.25 + pressure * 0.35, 2)
    focus = [
        {"label": FOCUS_LABELS[(index + offset) % len(FOCUS_LABELS)], "score": round(0.92 - offset * 0.11, 2)}
        for offset in range(3)
    ]
    coach = coaches[index % len(coaches)]

    details = {
        **player,
        "skill_level": skill_level,
        "match_type": match_type,
        "opponent_level": opponent_level,
        "issues": issue,
        "filename": filename,
    }
    report = {
        "rallies": rallies,
        "unforced_errors": unforced_errors,
        "winners": winners,
        "net_ratio": net_ratio,
        "analysis_time": timestamp.strftime("%Y-%m-%d %H:%M"),
        "recommendations": [f"Prioritize {item['label']} drills this week." for item in focus],
        "video_metrics": {
            "cv_model": "Demo historical frame model",
            "width": 1280,
            "height": 720,
            "duration_seconds": 180 + (index % 16) * 15,
            "sampled_frames": 180,
            "avg_motion": round(0.04 + (index % 30) / 1000, 4),
            "motion_burst_rate": round(0.08 + (index % 20) / 100, 4),
            "court_presence": round(0.34 + (index % 35) / 100, 4),
            "dominant_court_color": ["green", "blue", "tan"][index % 3],
            "net_activity_ratio": round(net_ratio, 4),
            "baseline_activity_ratio": round(1 - net_ratio, 4),
            "lateral_balance": round(0.68 + (index % 25) / 100, 4),
        },
        "ml": {
            "analysis_source": "Seeded demo historical match",
            "features": {
                "tempo_score": tempo,
                "consistency_score": consistency,
                "pressure_score": pressure,
                "estimated_duration_min": round((180 + (index % 16) * 15) / 60, 1),
                "issue_complexity": 2,
            },
            "skill_prediction": {
                "label": skill_level,
                "confidence": 0.72,
                "probabilities": {
                    "Beginner": 0.2 if skill_level != "Beginner" else 0.72,
                    "Intermediate": 0.2 if skill_level != "Intermediate" else 0.72,
                    "Advanced": 0.2 if skill_level != "Advanced" else 0.72,
                },
            },
            "play_style": {"label": ["Control Builder", "Balanced Rallyer", "Attack Finisher"][index % 3], "confidence": 0.76},
            "shot_mix": {"Dinks": 0.28, "Drives": 0.24, "Volleys": 0.2, "Third-shot drops": 0.16, "Serve returns": 0.12},
            "focus_areas": focus,
            "coach_matches": [
                {
                    "name": coach["name"],
                    "slug": coach["slug"],
                    "location": coach["location"],
                    "rating": coach["rating"],
                    "match_score": round(0.72 + rng.random() * 0.22, 2),
                    "matched_specialties": [focus[0]["label"]],
                }
            ],
        },
    }
    report["advanced"] = build_advanced_sections(report, report["ml"]["features"])
    return {"timestamp": timestamp.isoformat(), "details": details, "report": report}


def build_advanced_sections(report: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    duration = float(report["video_metrics"]["duration_seconds"])
    tempo = float(features["tempo_score"])
    consistency = float(features["consistency_score"])
    pressure = float(features["pressure_score"])
    net_ratio = float(report["net_ratio"])
    focus_label = report["ml"]["focus_areas"][0]["label"]
    timeline = [
        {
            "label": "Opening pattern",
            "start_seconds": 0,
            "end_seconds": round(duration * 0.33),
            "score": round(tempo * 0.82 + consistency * 0.18, 2),
            "summary": f"Opening exchanges point toward {focus_label.lower()} as the best early upgrade.",
        },
        {
            "label": "Mid-match pressure",
            "start_seconds": round(duration * 0.33),
            "end_seconds": round(duration * 0.66),
            "score": round(pressure * 0.78 + tempo * 0.22, 2),
            "summary": "Pressure rallies show where court position and reset choices matter most.",
        },
        {
            "label": "Closing execution",
            "start_seconds": round(duration * 0.66),
            "end_seconds": round(duration),
            "score": round(consistency * 0.72 + pressure * 0.28, 2),
            "summary": "Closing points need the same repeatable shape as the stronger blocks.",
        },
    ]
    rally_segments = []
    for offset in range(6):
        rally_segments.append({
            "label": f"Rally block {offset + 1}",
            "start_seconds": round(duration * offset / 6),
            "end_seconds": round(duration * (offset + 1) / 6),
            "rallies": max(1, report["rallies"] // 6 + (1 if offset < report["rallies"] % 6 else 0)),
            "pressure_score": round(max(0.0, min(1.0, pressure + ((offset % 3) - 1) * 0.06)), 2),
            "consistency_score": round(max(0.0, min(1.0, consistency - (offset % 2) * 0.04)), 2),
            "note": f"Use this block to reinforce {focus_label.lower()} under realistic point rhythm.",
        })
    return {
        "performance_bands": [
            {"label": "Tempo", "value": tempo, "status": score_status(tempo)},
            {"label": "Consistency", "value": consistency, "status": score_status(consistency)},
            {"label": "Pressure", "value": pressure, "status": score_status(pressure)},
            {"label": "Net Control", "value": net_ratio, "status": score_status(net_ratio)},
        ],
        "timeline": timeline,
        "rally_segments": rally_segments,
        "coaching_summary": {
            "strengths": [
                "Repeatable patterns are visible across multiple rally blocks.",
                "Shot mix gives the coach enough signal for targeted drills.",
            ],
            "risks": [
                "Error pressure can climb when the rally pace changes.",
                "Court position needs monitoring between baseline and kitchen phases.",
            ],
            "next_session_plan": [
                f"Warm up with {focus_label.lower()} pattern reps.",
                "Run pressure games to 7 with serve-return constraints.",
                "Finish with filmed points and compare rally-block consistency.",
            ],
        },
    }


def score_status(value: float) -> str:
    if value >= 0.7:
        return "strong"
    if value >= 0.5:
        return "steady"
    return "needs work"


def build_booking(index: int, analysis: dict[str, Any], now: datetime) -> dict[str, Any]:
    player = analysis["details"]
    coach = coaches[(index * 7) % len(coaches)]
    requested = now - timedelta(hours=index * 3, minutes=index % 53)
    status = ["pending", "accepted", "declined", "accepted"][index % 4]
    preferred_date = (now.date() + timedelta(days=1 + index % 45)).isoformat()
    slots = coach.get("availability") or ["Mon 5:00 PM"]
    return {
        "id": f"demo-booking-{index:04d}",
        "timestamp": requested.isoformat(),
        "status": status,
        "coach": coach["name"],
        "name": player["name"],
        "email": player["email"],
        "preferred_date": preferred_date,
        "preferred_time": f"{8 + index % 10:02d}:30",
        "preferred_slot": slots[index % len(slots)],
        "message": f"Demo booking after match analysis. Focus: {analysis['report']['ml']['focus_areas'][0]['label']}.",
        "updated_at": (requested + timedelta(hours=2)).isoformat() if status != "pending" else None,
    }


def build_job(index: int, analysis: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"demo-job-{index:04d}",
        "status": "complete",
        "created_at": analysis["timestamp"],
        "updated_at": analysis["timestamp"],
        "filename": analysis["details"]["filename"],
        "upload_path": f"seeded://{analysis['details']['filename']}",
        "details": analysis["details"],
        "result": analysis,
    }


def seed(users: int, seed_value: int) -> None:
    rng = random.Random(seed_value)
    now = datetime.now()
    analyses = [build_analysis(index, now, rng) for index in range(1, users + 1)]
    bookings = [build_booking(index, analysis, now) for index, analysis in enumerate(analyses, start=1)]
    jobs = [build_job(index, analysis) for index, analysis in enumerate(analyses, start=1)]

    save_json_records("data.json", analyses)
    save_json_records("bookings.json", bookings)
    replace_analysis_jobs(jobs)

    print(f"Seeded {len(analyses)} match analyses.")
    print(f"Seeded {len(bookings)} bookings.")
    print(f"Seeded {len(jobs)} completed analysis jobs.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed demo-scale PickleCoach usage data.")
    parser.add_argument("--users", type=int, default=4000, help="Number of demo users to seed.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic demo data.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed(args.users, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
