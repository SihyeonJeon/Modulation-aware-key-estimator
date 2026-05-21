from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from .keys import key_to_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mod-key-estimator",
        description="Estimate region-wise keys and optionally transpose a song to a target key.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wav", type=Path, help="Path to a local WAV/audio file.")
    source.add_argument("--youtube-url", help="YouTube URL to download with yt-dlp.")
    parser.add_argument("--target-key", default="c", help="Target key for region-wise pitch transfer.")
    parser.add_argument("--model", type=Path, default=None, help="Optional checkpoint path.")
    parser.add_argument("--output", type=Path, default=None, help="Output WAV path for the shifted audio.")
    parser.add_argument("--cookies", type=Path, default=None, help="Optional yt-dlp cookies.txt path.")
    parser.add_argument("--no-shift", action="store_true", help="Only estimate keys; do not write a shifted WAV.")
    parser.add_argument("--json", action="store_true", help="Print compact JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from .download import download_youtube_audio
    from .inference import run_inference
    from .model_loader import load_model

    if args.youtube_url:
        wav_path = download_youtube_audio(
            args.youtube_url,
            output_dir=Path(tempfile.gettempdir()) / "modulation-key-estimator",
            cookies_path=args.cookies,
        )
    else:
        wav_path = args.wav

    model = load_model(args.model)
    result = run_inference(
        wav_path,
        model,
        target_key_index=key_to_index(args.target_key),
        write_shifted=not args.no_shift,
        output_path=args.output,
    )

    indent = None if args.json else 2
    print(json.dumps(result, ensure_ascii=False, indent=indent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
