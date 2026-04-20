# PickleCoach AI Prototype

PickleCoach AI is a local FastAPI prototype for pickleball video analysis and coach discovery. It combines a frame-sampling video pipeline, model-style match intelligence, coach recommendations, booking requests and SQLite-backed history.

## Features

* **Home page** with links to the core player and coach workflows.
* **Upload page** that accepts an MP4 match video, queues a background analysis job and returns a status page while processing runs.
* **YouTube demo analysis** for running the same pipeline against the bundled sample match URL.
* **Analysis page** that displays match metrics, measured video signals, AI match intelligence, training focus areas and coach recommendations.
* **Advanced coaching dashboard** with performance bands, timeline reads, rally blocks, strengths, risks and next-session planning.
* **Coach finder** with generated marketplace supply, filters, profile pages, tiers, specialties and booking links.
* **Booking dashboard** for reviewing, accepting and declining incoming coach session requests.
* **Player and coach accounts** with signup, login, logout, signed session cookies and hashed passwords.
* **SQLite persistence** for analysis jobs, saved analysis records and booking requests. The app creates `backend/picklecoach.sqlite3` at runtime, or you can override the path with `DATABASE_PATH`.

## Project Structure

```text
picklecoach-prototype/
├── backend/
│   ├── main.py                  # FastAPI app and route handlers
│   ├── config.py                # Environment and path configuration
│   ├── database.py              # SQLite schema and repositories
│   ├── storage.py               # Compatibility storage helpers
│   ├── marketplace.py           # Coach catalogue and tier data
│   ├── match_intelligence.py    # Match scoring and recommendations
│   ├── cv_learning.py           # Trainable CV scoring artifact helpers
│   ├── deep_analysis.py         # Ensemble-style synthetic ML analysis
│   ├── train_cv_model.py        # Batch training CLI
│   ├── coaches.json             # Seed coach data
│   ├── templates/               # Jinja2 HTML templates
│   ├── static/                  # CSS and static assets
│   └── tests/                   # Pytest suite
├── requirements.txt
└── README.md
```

## Running The Prototype

1. Install Python 3.9 or higher.
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the application:

   ```bash
   cd backend
   uvicorn main:app --reload
   ```

4. Visit `http://localhost:8000`.

## Persistence

Analysis jobs, analysis history and booking requests are stored in SQLite through `backend/database.py` and `backend/storage.py`. If older `backend/data.json` or `backend/bookings.json` files exist, they are imported into the database the first time the relevant table is empty.

Runtime database files are ignored by git:

```text
backend/picklecoach.sqlite3
backend/*.sqlite3-*
```

## Accounts

Players and coaches can create local accounts at `/signup` and log in at `/login`. Passwords are stored with PBKDF2 hashes and sessions are signed HTTP-only cookies.

The booking dashboard remains open during local development when `BOOKINGS_PASSWORD` is unset. If `BOOKINGS_PASSWORD` is set, `/bookings` accepts either the existing `?key=...` fallback or a logged-in coach/admin account.

## Load Testing

The repository includes an async real-user simulator in `backend/load_test.py`. Start the server first:

```bash
cd backend
uvicorn main:app --host 127.0.0.1 --port 8000
```

Then run a 1000-user read-heavy simulation from another terminal:

```bash
python backend/load_test.py --users 1000 --duration 60 --ramp-up 10 --base-url http://127.0.0.1:8000
```

The default mix simulates home/upload browsing, coach search, profile views, booking-form views, stats/history pages and API reads. To include booking POST traffic, add `--include-booking-writes`; that intentionally creates database rows.

For a quick app-only check without starting uvicorn:

```bash
python backend/load_test.py --in-process --users 50 --duration 10
```

## Demo-Scale Data

To make the app look like thousands of players have already used match analysis and bookings, seed deterministic demo history:

```bash
python backend/seed_demo_data.py --users 4000
```

This replaces demo analysis records, completed analysis jobs and booking requests in the SQLite database. History and bookings are paginated, and Stats will reflect the seeded totals.

## Training The Video Model

Uploaded videos are analyzed with a local computer-vision pipeline that samples frames, measures motion, estimates court color/coverage and tracks activity zones. You can add a learned scoring layer by training on a larger match library:

```bash
python backend/train_cv_model.py --videos path\to\pickleball\matches
```

For a labeled or URL-based dataset, create a CSV manifest with a `source` column and optional `tempo_score`, `consistency_score` and `pressure_score` columns:

```bash
python backend/train_cv_model.py --manifest training_manifest.csv
```

The trainer writes `backend/models/cv_model.pkl`. When that artifact exists, the app automatically blends the learned model scores into each upload analysis. To reach a 3000-match model, point the trainer at a directory or manifest containing 3000 successfully readable match videos.

## Customisation

* **Improving analysis**: extend the frame analysis pipeline with shot detection, rally segmentation or pose tracking models.
* **Adding coaches**: edit `coaches.json` to seed coach profiles, or migrate coach records into the database layer.
* **Styling**: update `static/style.css` to customise the app.
* **Deployment**: run FastAPI behind a reverse proxy, set production secrets, configure persistent storage and move long-running video analysis into background jobs.
