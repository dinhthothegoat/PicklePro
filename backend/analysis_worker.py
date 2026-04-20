"""Resumable analysis worker for queued video jobs.

Run this in a separate terminal for heavier analysis sessions:

    python -m backend.analysis_worker
"""

from __future__ import annotations

import logging
import time

from .storage import list_analysis_jobs_by_status
from .main import process_upload_analysis_job, process_youtube_analysis_job


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_available_jobs() -> int:
    jobs = [
        job for job in list_analysis_jobs_by_status(["queued", "processing"])
        if job is not None
    ]
    for job in jobs:
        if job.get("details", {}).get("youtube_url"):
            logger.info("Worker processing YouTube job %s", job["id"])
            process_youtube_analysis_job(job["id"])
        else:
            logger.info("Worker processing upload job %s", job["id"])
            process_upload_analysis_job(job["id"])
    return len(jobs)


def main(poll_seconds: int = 5) -> None:
    logger.info("Analysis worker started.")
    while True:
        processed = process_available_jobs()
        if processed == 0:
            time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
