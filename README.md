# Modulation-Aware Key Estimator

[![CI](https://github.com/SihyeonJeon/Modulation-aware-key-estimator/actions/workflows/ci.yml/badge.svg)](https://github.com/SihyeonJeon/Modulation-aware-key-estimator/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Region-wise key estimation for songs that change key

Most key-estimation demos force one global label onto a whole track. This repo
keeps the section boundary visible: it estimates likely key regions, reports
candidate modulation points, and can pitch-shift each region toward a target
key.

```json
{
  "target_key_name": "C",
  "modulation_points": [{"time_sec": 74.24}],
  "region_infos": [
    {"start_time_sec": 0.0, "end_time_sec": 74.24, "key_name": "G", "confidence": 0.82},
    {"start_time_sec": 74.24, "end_time_sec": 181.76, "key_name": "A", "confidence": 0.77}
  ]
}
```

## What It Does

- extracts chroma and HPCP-style harmonic pitch-class features
- runs a two-stream Transformer checkpoint
- predicts 12 pitch-class keys per audio window
- groups windows into likely key regions
- exposes approximate modulation points
- serves local CLI and FastAPI inference
- downloads the release checkpoint with SHA-256 verification

## Install

```bash
git clone https://github.com/SihyeonJeon/Modulation-aware-key-estimator.git
cd Modulation-aware-key-estimator
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The checkpoint downloads on first use from the GitHub release and is cached
under `~/.cache/modulation-aware-key-estimator/`.

Use a local checkpoint instead:

```bash
MODEL_CHECKPOINT_PATH=/path/to/key_model.pt mod-key-estimator --wav song.wav --json
```

## CLI

Local file:

```bash
mod-key-estimator --wav song.wav --target-key c --json
```

YouTube URL through `yt-dlp`:

```bash
mod-key-estimator --youtube-url "https://www.youtube.com/watch?v=..." --target-key f#
```

If a video requires browser cookies, pass them explicitly:

```bash
mod-key-estimator --youtube-url "https://www.youtube.com/watch?v=..." --cookies ./cookies.txt
```

No cookies file is stored in this repository.

## API

```bash
uvicorn modulation_key_estimator.api:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/analyze-file \
  -F "file=@song.wav" \
  -F "target_key=c"
```

```bash
curl -X POST http://localhost:8000/analyze-youtube \
  -H "content-type: application/json" \
  -d '{"youtube_url":"https://www.youtube.com/watch?v=...","target_key":"c"}'
```

## Docker

```bash
docker build -t modulation-key-estimator .
docker run --rm -p 8000:8000 modulation-key-estimator
```

## Model Surface

| Item | Value |
| --- | --- |
| input | mono audio, resampled to 16 kHz |
| features | chroma + HPCP-style 12-bin harmonic features |
| architecture | two-stream Transformer encoder with attention pooling |
| output | 12 pitch-class probabilities per window |
| regioning | probability-shift grouping across neighboring windows |
| checkpoint | GitHub Release asset with SHA-256 verification |

See [docs/model-card.md](docs/model-card.md) for intended use, limitations, and
failure modes.

## Evaluation

Run a labeled manifest:

```bash
python scripts/evaluate_manifest.py examples/manifest.example.csv --json
```

Expected CSV columns:

```csv
path,expected_key
path/to/song.wav,c
```

The script reports exact pitch-class accuracy and per-file predictions. Replace
the example manifest with local labeled audio before reporting a benchmark
number.

## Boundary

This repo currently ships the inference package, model architecture, release
checkpoint, and manifest-based evaluation script. It does not yet ship the
original training code, training manifest, dataset list, or training logs.

The checkpoint predicts pitch class only: `C`, `C#`, ..., `B`. It does not
model major/minor, modal function, enharmonic spelling, or score-level harmonic
analysis.
