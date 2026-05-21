from __future__ import annotations

import os
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .download import download_youtube_audio
from .inference import run_inference
from .keys import KEY_NAMES, key_to_index
from .model_loader import default_checkpoint_path, load_model

app = FastAPI(
    title="Modulation-Aware Key Estimator",
    version="0.1.0",
    description="Estimate region-wise musical keys and transpose modulating songs to a target key.",
)


class YoutubeRequest(BaseModel):
    youtube_url: str = Field(..., examples=["https://www.youtube.com/watch?v=..."])
    target_key: str = "c"
    write_shifted: bool = True


@lru_cache(maxsize=1)
def get_model():
    return load_model()


def _data_dir() -> Path:
    root = os.environ.get("MOD_KEY_DATA_DIR", tempfile.gettempdir())
    path = Path(root) / "modulation-key-estimator"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cookies_path() -> Path | None:
    value = os.environ.get("YTDLP_COOKIES_PATH")
    return Path(value).expanduser().resolve() if value else None


def _target_key_index(target_key: str) -> int:
    try:
        return key_to_index(target_key)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/health")
def health():
    checkpoint = default_checkpoint_path()
    return {
        "status": "ok",
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.exists(),
    }


@app.get("/keys")
def keys():
    return {"keys": list(KEY_NAMES)}


@app.post("/analyze")
async def analyze_audio_legacy(request: YoutubeRequest):
    return await analyze_youtube(request)


@app.post("/analyze-youtube")
async def analyze_youtube(request: YoutubeRequest):
    target_index = _target_key_index(request.target_key)
    try:
        wav_path = download_youtube_audio(
            request.youtube_url,
            output_dir=_data_dir(),
            cookies_path=_cookies_path(),
        )
        result = run_inference(
            wav_path,
            get_model(),
            target_key_index=target_index,
            write_shifted=request.write_shifted,
        )
        return {"result": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-file")
async def analyze_file(
    file: UploadFile = File(...),
    target_key: str = Form("c"),
    write_shifted: bool = Form(True),
):
    target_index = _target_key_index(target_key)
    suffix = Path(file.filename or "input.wav").suffix or ".wav"
    input_path = _data_dir() / f"upload-{uuid.uuid4().hex}{suffix}"
    try:
        input_path.write_bytes(await file.read())
        result = run_inference(
            input_path,
            get_model(),
            target_key_index=target_index,
            write_shifted=write_shifted,
        )
        return {"result": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
