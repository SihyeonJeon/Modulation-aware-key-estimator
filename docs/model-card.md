# Model Card

## Summary

`Modulation-aware-key-estimator` is a music-information-retrieval model for
estimating region-wise musical keys. It is designed for songs where a single
global key label loses useful information.

## Intended Use

- Inspect likely key regions in a WAV/audio file.
- Detect approximate modulation points.
- Transpose detected regions into a requested target key for practice or
  prototyping.
- Serve inference through a small FastAPI app or CLI.

## Not Intended For

- Copyright circumvention or redistribution of downloaded audio.
- Music-theory grading where enharmonic spelling, mode, and harmonic context
  must be exact.
- Separating major/minor or modal labels. This checkpoint predicts 12 pitch
  classes, not full tonal function.

## Architecture

The model uses two feature streams:

- chroma features from `librosa`
- HPCP-style harmonic pitch-class features from `essentia` when available, with
  a `librosa` fallback

Each stream is projected into a Transformer encoder. The streams are fused,
pooled with attention, normalized, and classified into 12 key classes.

## Output Schema

The inference result contains:

- `predicted_region_keys`: key indices for detected regions
- `region_infos`: start/end times, key names, confidence, and class
  probabilities
- `modulation_points`: detected boundary frames and timestamps
- `shifted_wav_path`: present only when pitch-shift output is enabled

## Known Limitations

- The checkpoint predicts pitch class only: `C`, `C#`, ..., `B`.
- Major/minor ambiguity is not modeled.
- Very short clips may be rejected by the windowing configuration.
- Dense chromatic music, noisy live audio, and non-Western tuning systems can
  reduce reliability.
- Region boundaries are approximate and should be treated as candidate
  modulation points, not score-level annotations.

## Data And Reproducibility

The original training notebook is retained for historical context. Public
portfolio use should rely on the package, CLI, API, and manifest-based
evaluation script rather than notebook state.

Use `scripts/evaluate_manifest.py` with a labeled CSV to produce a reproducible
accuracy report.

