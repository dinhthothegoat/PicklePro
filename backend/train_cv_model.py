r"""Batch-train the PickleCoach vision scoring model from match videos.

Examples:
    python backend/train_cv_model.py --videos D:\pickleball\matches
    python backend/train_cv_model.py --manifest training_manifest.csv

Manifest columns:
    source                  required; local path or YouTube URL
    tempo_score             optional 0..1 human label
    consistency_score       optional 0..1 human label
    pressure_score          optional 0..1 human label
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

try:
    from backend.cv_learning import TARGET_NAMES, heuristic_targets, train_cv_artifact
    from backend.main import analyze_video_pixels
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backend.cv_learning import TARGET_NAMES, heuristic_targets, train_cv_artifact
    from backend.main import analyze_video_pixels

import yt_dlp


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def iter_video_paths(directory: Path):
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def read_manifest(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            source = (row.get("source") or "").strip()
            if not source:
                continue
            targets = []
            has_all_targets = True
            for name in TARGET_NAMES:
                value = (row.get(name) or "").strip()
                if not value:
                    has_all_targets = False
                    break
                targets.append(float(value))
            yield {
                "source": source,
                "targets": targets if has_all_targets else None,
            }


def download_url(url: str, work_dir: Path) -> Path:
    output_template = str(work_dir / "%(id)s.%(ext)s")
    options = {
        "format": "bv*[ext=mp4][height<=720]/b[ext=mp4][height<=720]/worst",
        "outtmpl": output_template,
        "quiet": True,
        "noplaylist": True,
        "merge_output_format": "mp4",
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=True)
    for candidate in work_dir.iterdir():
        if candidate.suffix.lower() in VIDEO_EXTENSIONS:
            return candidate
    raise RuntimeError(f"No downloaded video file found for {url} ({info.get('id', 'unknown id')}).")


def analyze_source(source: str, targets, keep_downloads: Path | None):
    cleanup_dir = None
    try:
        if is_url(source):
            cleanup_dir = Path(tempfile.mkdtemp(prefix="picklecoach-train-"))
            video_path = download_url(source, cleanup_dir)
            if keep_downloads:
                keep_downloads.mkdir(parents=True, exist_ok=True)
                kept_path = keep_downloads / video_path.name
                shutil.copy2(video_path, kept_path)
        else:
            video_path = Path(source)
            if not video_path.exists():
                raise FileNotFoundError(source)

        metrics = analyze_video_pixels(str(video_path))
        return {
            "source": source,
            "metrics": metrics,
            "targets": targets or heuristic_targets(metrics).tolist(),
        }
    finally:
        if cleanup_dir and cleanup_dir.exists():
            shutil.rmtree(cleanup_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Train PickleCoach's learned CV scoring model.")
    parser.add_argument("--videos", type=Path, help="Directory of local match videos.")
    parser.add_argument("--manifest", type=Path, help="CSV manifest of local video paths or URLs.")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "cv_model.pkl",
        help="Output model artifact path.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of videos to process.")
    parser.add_argument("--keep-downloads", type=Path, help="Optional folder for downloaded URL videos.")
    args = parser.parse_args()

    if not args.videos and not args.manifest:
        parser.error("Provide --videos, --manifest, or both.")

    sources = []
    if args.videos:
        sources.extend({"source": str(path), "targets": None} for path in iter_video_paths(args.videos))
    if args.manifest:
        sources.extend(read_manifest(args.manifest))
    if args.limit:
        sources = sources[: args.limit]

    samples = []
    failures = []
    total = len(sources)
    for index, item in enumerate(sources, start=1):
        source = item["source"]
        print(f"[{index}/{total}] analyzing {source}")
        try:
            samples.append(analyze_source(source, item.get("targets"), args.keep_downloads))
        except Exception as exc:
            failures.append({"source": source, "error": str(exc)})
            print(f"  skipped: {exc}")

    result = train_cv_artifact(samples, args.artifact)
    print(f"Trained {result['artifact_path']} on {result['training_samples']} videos.")
    if failures:
        print(f"Skipped {len(failures)} failed videos.")
        for failure in failures[:10]:
            print(f"  {failure['source']}: {failure['error']}")
    if result["training_samples"] < 3000:
        print(
            "Note: this model is trained, but it has not seen 3000 matches yet. "
            f"Add {3000 - result['training_samples']} more successful samples to hit that target."
        )


if __name__ == "__main__":
    main()
