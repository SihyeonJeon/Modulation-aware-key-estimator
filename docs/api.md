# API

Start the server:

```bash
uvicorn modulation_key_estimator.api:app --reload
```

## `GET /health`

Returns checkpoint path and existence.

## `GET /keys`

Returns accepted target keys:

```json
{"keys":["c","c#","d","d#","e","f","f#","g","g#","a","a#","b"]}
```

## `POST /analyze-youtube`

```json
{
  "youtube_url": "https://www.youtube.com/watch?v=...",
  "target_key": "c",
  "write_shifted": true
}
```

Use `YTDLP_COOKIES_PATH=/path/to/cookies.txt` only when a source requires it.

## `POST /analyze-file`

Multipart form fields:

- `file`: WAV/audio file
- `target_key`: optional, default `c`
- `write_shifted`: optional, default `true`

