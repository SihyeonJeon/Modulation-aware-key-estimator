"""Audio download helpers."""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

from .utils import finalize_downloaded_wav


def download_youtube_audio(
    youtube_url: str,
    output_dir: str | Path,
    cookies_path: str | Path | None = None,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    unique_id = uuid.uuid4().hex[:8]
    output_template = str(output_path / f"%(title)s_{unique_id}.%(ext)s")
    command = [
        "yt-dlp",
        "-x",
        "--audio-format",
        "wav",
        "--output",
        output_template,
        "--encoding",
        "utf-8",
        "--geo-bypass",
        youtube_url,
    ]
    if cookies_path:
        command[1:1] = ["--cookies", str(cookies_path)]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(f"yt-dlp failed: {message}") from exc
    return Path(finalize_downloaded_wav(output_path))
