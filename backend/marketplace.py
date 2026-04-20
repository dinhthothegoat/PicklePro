"""Coach catalogue loading and prototype marketplace supply."""

from __future__ import annotations

import json
import logging

try:
    from .config import COACHES_PATH
except ImportError:
    from config import COACHES_PATH

logger = logging.getLogger(__name__)


COACH_TIERS = [
    "Super Saiyan Coach 1",
    "Super Saiyan Coach 2",
    "Super Saiyan Coach 3",
    "God Coach",
    "Blue Coach",
    "Blue Kaioken Coach",
    "Evolution Coach",
    "Ultra Coach",
]

COACH_TIER_REQUIREMENTS = [
    {
        "name": "Super Saiyan Coach 1",
        "class_name": "tier-ssj-1",
        "summary": "Reliable fundamentals coach for newer players.",
        "requirements": "Minimum 4.1 rating, 5 completed sessions, beginner lesson specialty, and a complete coach profile.",
        "how_to_achieve": "Run consistent intro sessions, collect player feedback, and keep availability current.",
    },
    {
        "name": "Super Saiyan Coach 2",
        "class_name": "tier-ssj-2",
        "summary": "Strong coach with repeat bookings and clear lesson structure.",
        "requirements": "Minimum 4.2 rating, 12 completed sessions, two specialties, and at least one accepted booking per week.",
        "how_to_achieve": "Build repeat player relationships and publish focused lesson types.",
    },
    {
        "name": "Super Saiyan Coach 3",
        "class_name": "tier-ssj-3",
        "summary": "Advanced coach for players improving match habits.",
        "requirements": "Minimum 4.3 rating, 25 completed sessions, three specialties, and strong booking response history.",
        "how_to_achieve": "Coach structured drills, respond quickly, and help players track progress.",
    },
    {
        "name": "God Coach",
        "class_name": "tier-god",
        "summary": "Elite regional coach with trusted marketplace activity.",
        "requirements": "Minimum 4.5 rating, 40 completed sessions, low decline rate, and verified lesson quality.",
        "how_to_achieve": "Maintain high ratings across multiple player levels and keep cancellations rare.",
    },
    {
        "name": "Blue Coach",
        "class_name": "tier-blue",
        "summary": "Premium coach for competitive and tactical development.",
        "requirements": "Minimum 4.6 rating, 60 completed sessions, advanced drills, and consistent availability.",
        "how_to_achieve": "Offer match-play coaching, video review, and specific tactical improvements.",
    },
    {
        "name": "Blue Kaioken Coach",
        "class_name": "tier-blue-kaioken",
        "summary": "High-demand coach with fast response and strong outcomes.",
        "requirements": "Minimum 4.7 rating, 85 completed sessions, high acceptance rate, and repeat booking history.",
        "how_to_achieve": "Reply quickly, create clear training plans, and convert sessions into repeat bookings.",
    },
    {
        "name": "Evolution Coach",
        "class_name": "tier-evolution",
        "summary": "Top-tier coach with broad player trust and deep specialization.",
        "requirements": "Minimum 4.8 rating, 120 completed sessions, multiple advanced specialties, and excellent player reviews.",
        "how_to_achieve": "Specialize deeply, coach measurable improvements, and keep profile quality sharp.",
    },
    {
        "name": "Ultra Coach",
        "class_name": "tier-ultra",
        "summary": "Highest marketplace tier for exceptional coaching quality.",
        "requirements": "Minimum 4.9 rating, 160 completed sessions, elite response history, and standout player outcomes.",
        "how_to_achieve": "Deliver premium sessions, maintain near-perfect feedback, and mentor across multiple skill levels.",
    },
]

TIER_CLASS_BY_NAME = {
    tier["name"]: tier["class_name"]
    for tier in COACH_TIER_REQUIREMENTS
}

COACHES_PER_PAGE = 24

AUSTRALIA_LOCATIONS = [
    "Gold Coast",
    "Brisbane",
    "Sunshine Coast",
    "Sydney",
    "Melbourne",
    "Perth",
    "Adelaide",
    "Canberra",
]

VIETNAM_LOCATIONS = [
    "Ho Chi Minh City",
    "Hanoi",
    "Da Nang",
    "Nha Trang",
    "Can Tho",
    "Hue",
    "Hoi An",
    "Vung Tau",
]

SPECIALTY_GROUPS = [
    ["Beginner lessons", "Dinks", "Footwork"],
    ["Serve returns", "Court positioning", "Doubles strategy"],
    ["Volleys", "Net play", "Match tempo"],
    ["Third-shot drops", "Resets", "Soft game"],
    ["Doubles basics", "Footwork", "Serve returns"],
    ["Advanced drills", "Net play", "Match tempo"],
]

LESSON_TYPE_GROUPS = [
    ["Private session", "Doubles basics"],
    ["Private session", "Strategy session"],
    ["Private session", "Advanced drills"],
    ["Private session", "Video review"],
    ["Private session", "Group clinic"],
]

FIRST_NAMES = [
    "Alex",
    "Jordan",
    "Taylor",
    "Casey",
    "Morgan",
    "Riley",
    "Sam",
    "Jamie",
    "Chris",
    "Nina",
    "Tuan",
    "An",
    "Huy",
    "Lan",
    "Quang",
    "Thao",
    "Grace",
    "Noah",
    "Mia",
    "Leo",
]

LAST_NAMES = [
    "Nguyen",
    "Tran",
    "Pham",
    "Le",
    "Hoang",
    "Vo",
    "Wilson",
    "Brown",
    "Taylor",
    "Martin",
    "Chen",
    "Singh",
    "Walker",
    "Kelly",
    "Roberts",
]


def slugify(value: str):
    """Create a simple URL slug."""
    return value.lower().replace(" ", "-")


def build_generated_coach(index: int):
    """Build deterministic fake marketplace supply for the prototype."""
    country = "Australia" if index % 2 == 0 else "Vietnam"
    locations = AUSTRALIA_LOCATIONS if country == "Australia" else VIETNAM_LOCATIONS
    first = FIRST_NAMES[index % len(FIRST_NAMES)]
    last = LAST_NAMES[(index * 3) % len(LAST_NAMES)]
    name = f"{first} {last} {index:03d}"
    tier = COACH_TIERS[index % len(COACH_TIERS)]
    rating = round(4.1 + ((index % 9) * 0.1), 1)
    price = 45 + (index % 7) * 5 if country == "Australia" else 25 + (index % 6) * 4
    location = locations[index % len(locations)]

    return {
        "name": name,
        "slug": slugify(name),
        "country": country,
        "location": location,
        "rating": rating,
        "price": price,
        "email": f"coach{index:03d}@example.com",
        "badge": tier,
        "image_url": "https://source.unsplash.com/900x650/?pickleball,coach",
        "bio": f"{tier} based in {location}, focused on practical drills, confident rallies, and match-ready habits.",
        "specialties": SPECIALTY_GROUPS[index % len(SPECIALTY_GROUPS)],
        "lesson_types": LESSON_TYPE_GROUPS[index % len(LESSON_TYPE_GROUPS)],
        "availability": [
            f"Mon {6 + (index % 4)}:00 PM",
            f"Wed {5 + (index % 4)}:30 PM",
            f"Sat {7 + (index % 5)}:00 AM",
        ],
    }


def expand_fake_coaches(seed_coaches, target_count=200):
    """Expand seed coaches to a larger deterministic fake marketplace."""
    expanded = list(seed_coaches)
    seen_slugs = {coach.get("slug") for coach in expanded}

    index = 1
    while len(expanded) < target_count:
        coach = build_generated_coach(index)
        if coach["slug"] not in seen_slugs:
            expanded.append(coach)
            seen_slugs.add(coach["slug"])
        index += 1

    return expanded


def load_coaches():
    try:
        with open(COACHES_PATH, "r", encoding="utf-8") as f:
            loaded_coaches = json.load(f)
    except FileNotFoundError:
        logger.error("coaches.json not found at %s — starting with empty coach list.", COACHES_PATH)
        loaded_coaches = []
    except json.JSONDecodeError as exc:
        logger.error("coaches.json is malformed (%s) — starting with empty coach list.", exc)
        loaded_coaches = []

    loaded_coaches = expand_fake_coaches(loaded_coaches)
    for coach in loaded_coaches:
        coach["tier_class"] = TIER_CLASS_BY_NAME.get(coach.get("badge", ""), "tier-default")
    return loaded_coaches


coaches = load_coaches()


def find_coach_by_name(coach_name: str):
    """Find a coach by name using a case-insensitive exact match."""
    return next(
        (c for c in coaches if c.get("name", "").lower() == coach_name.lower()),
        None,
    )


def find_coach_by_slug(slug: str):
    """Find a coach by URL slug."""
    return next((c for c in coaches if c.get("slug") == slug), None)
