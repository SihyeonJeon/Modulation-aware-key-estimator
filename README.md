# Modulation-Aware Key Estimator

Estimate the key of songs that modulate.

This project detects likely key regions in an audio track, reports the key for
each region, and can transpose every region into one target key. It is built
around a two-stream Transformer that reads chroma and HPCP-style harmonic
features instead of treating the whole song as one static key.

## Why It Exists

Most key-estimation demos assume one song has one key. That breaks on music with
clear modulation, borrowed sections, or long bridge changes. This repo keeps the
modulation points visible:

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

## Install

```bash
git clone https://github.com/SihyeonJeon/Modulation-aware-key-estimator.git
cd Modulation-aware-key-estimator
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The trained checkpoint is included at
`modulation_key_estimator/assets/key_model.pt`.

## CLI

Analyze a local file:

```bash
mod-key-estimator --wav song.wav --target-key c --json
```

Analyze a YouTube URL through `yt-dlp`:

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
curl -X POST http://localhost:8000/analyze-youtube \
  -H "content-type: application/json" \
  -d '{"youtube_url":"https://www.youtube.com/watch?v=...","target_key":"c"}'
```

For local files:

```bash
curl -X POST http://localhost:8000/analyze-file \
  -F "file=@song.wav" \
  -F "target_key=c"
```

## Docker

```bash
docker build -t modulation-key-estimator .
docker run --rm -p 8000:8000 modulation-key-estimator
```

## Model

- Input: mono audio, resampled to 16 kHz
- Features: chroma + HPCP-style 12-bin harmonic features
- Window: 1024 frames with 64-frame stride
- Architecture: two-stream Transformer encoder with attention pooling
- Output: 12-class key probability per window
- Regioning: detects probability shifts across neighboring windows, then
  re-estimates a key per region

See [docs/model-card.md](docs/model-card.md) for intended use, limitations, and
failure modes.

## Evaluation

Run a labeled manifest:

```bash
python scripts/evaluate_manifest.py examples/manifest.example.csv --json
```

Replace the example path with local labeled audio files before running.

Expected CSV columns:

```csv
path,expected_key
path/to/song.wav,c
```

The script reports exact-key accuracy and per-file predictions. It intentionally
does not ship a claimed benchmark number without the source manifest.
