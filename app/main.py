from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import subprocess
import uuid
from model_loader import load_model
from inference import run_inference
from utils import finalize_downloaded_wav

app = FastAPI()
model_checkpoint_path = "model_keepgoing.pt"
model = load_model(model_checkpoint_path)  # ✅ FastAPI 실행 시 한 번만 로드

class YoutubeRequest(BaseModel):
    youtube_url: str
    target_key: str = "c"

@app.post("/analyze")
async def analyze_audio(request: YoutubeRequest):
    try:
        output_dir = Path("/app/data")
        output_dir.mkdir(parents=True, exist_ok=True)

        unique_id = uuid.uuid4().hex[:8]
        output_template = f"{output_dir}/%(title)s_{unique_id}.%(ext)s"
        download_command = [
            "yt-dlp",
            "-x", "--audio-format", "wav",
            "--output", output_template,
            "--encoding", "utf-8",
            "--cookies", "/app/cookies.txt",
            "--geo-bypass",
            request.youtube_url
        ]
        subprocess.run(download_command, check=True)

        wav_path = finalize_downloaded_wav(output_dir)

        target_key_index = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b'].index(request.target_key.lower())
        result = run_inference(wav_path, model, target_key_index=target_key_index)

        return {"result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
