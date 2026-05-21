from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
import torch.nn.functional as F

from .keys import KEY_NAMES, key_name
from .utils import compute_chromagram, compute_hpcp, pitch_shift_segments, preprocess_waveform


@dataclass(frozen=True)
class InferenceConfig:
    sample_rate: int = 16000
    window_size: int = 1024
    stride: int = 64
    hop_length: int = 512
    modulation_margin_windows: int = 16
    probability_delta_threshold: float = 0.35
    min_modulation_distance_windows: int = 12


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_inference(
    wav_path: str | Path,
    model,
    target_key_index: int = 0,
    write_shifted: bool = True,
    output_path: str | Path | None = None,
    config: InferenceConfig | None = None,
):
    config = config or InferenceConfig()
    sr = config.sample_rate
    window_size = config.window_size
    stride = config.stride
    hop_length = config.hop_length
    device = _model_device(model)

    def frame_to_time(frame_index):
        return frame_index * hop_length / sr

    waveform_origin, original_sr = torchaudio.load(wav_path)
    if waveform_origin.shape[0] > 1:
        waveform_origin = waveform_origin.mean(dim=0, keepdim=True)
    if original_sr != sr:
        waveform_origin = torchaudio.transforms.Resample(original_sr, sr)(waveform_origin)
    waveform = preprocess_waveform(waveform_origin, sr=sr)

    chroma = compute_chromagram(waveform, sr=sr)
    hpcp = compute_hpcp(waveform, sr=sr)
    min_len = min(chroma.shape[0], hpcp.shape[0])
    feats_full = torch.cat([chroma[:min_len], hpcp[:min_len]], dim=1)

    input_batch = []
    windows = []

    for start in range(0, feats_full.shape[0] - window_size + 1, stride):
        window = feats_full[start : start + window_size]
        input_batch.append(window)
        windows.append(start)

    if not input_batch:
        raise ValueError("Audio is too short to analyze with the current window size.")

    input_tensor = torch.stack(input_batch).to(device)

    with torch.inference_mode():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=-1)

    logits_tensor = probs.cpu()
    mod_point_candidates = []
    margin = config.modulation_margin_windows
    for i in range(len(windows) - margin):
        left_probs = logits_tensor[i]
        right_probs = logits_tensor[i + margin]
        left_key = left_probs.argmax().item()
        right_key = right_probs.argmax().item()

        if left_key != right_key:
            prob_diff = torch.abs(left_probs - right_probs).sum().item()
            if prob_diff >= config.probability_delta_threshold:
                mod_point_candidates.append((i + margin, prob_diff))

    mod_point_candidates.sort(key=lambda x: -x[1])
    selected_mod_frames = []
    for idx, _ in mod_point_candidates:
        if all(abs(idx - sel) >= config.min_modulation_distance_windows for sel in selected_mod_frames):
            selected_mod_frames.append(idx)

    mod_frames = [windows[i] for i in selected_mod_frames]
    region_boundaries = [0] + sorted(mod_frames) + [feats_full.shape[0]]
    region_keys = []
    region_infos = []

    for i in range(len(region_boundaries) - 1):
        start_frame = region_boundaries[i]
        end_frame = region_boundaries[i + 1]

        if end_frame - start_frame < window_size:
            window_feats = feats_full[start_frame:end_frame]
            pad_len = window_size - (end_frame - start_frame)
            window_feats = torch.nn.functional.pad(window_feats, (0,0,0,pad_len))
            region_windows = [window_feats]
        else:
            region_windows = []
            for start in range(start_frame, end_frame - window_size + 1, stride):
                window_feats = feats_full[start:start+window_size]
                region_windows.append(window_feats)

        if not region_windows:
            continue

        region_tensor = torch.stack(region_windows).to(device)
        with torch.inference_mode():
            region_logits = F.softmax(model(region_tensor), dim=-1).cpu()
        mean_probs = region_logits.mean(dim=0)
        region_key = mean_probs.argmax().item()
        region_keys.append(region_key)

        region_infos.append({
            "region_index": i,
            "start_time_sec": frame_to_time(start_frame),
            "end_time_sec": frame_to_time(end_frame),
            "key_index": region_key,
            "key_name": key_name(region_key).upper(),
            "confidence": float(mean_probs.max().item()),
            "probabilities": {
                key_name(index).upper(): float(value)
                for index, value in enumerate(mean_probs.tolist())
            },
        })

    result = {
        "predicted_region_keys": region_keys,
        "region_infos": region_infos,
        "target_key_index": target_key_index,
        "target_key_name": KEY_NAMES[target_key_index].upper(),
        "modulation_points": [
            {
                "frame": int(frame),
                "time_sec": frame_to_time(frame),
            }
            for frame in sorted(mod_frames)
        ],
    }

    if write_shifted:
        shifted_waveform = pitch_shift_segments(
            waveform_origin,
            sr,
            region_boundaries,
            region_keys,
            target_key_index=target_key_index,
            keys_linear=list(KEY_NAMES),
        )
        if output_path is None:
            output_path = os.path.splitext(str(wav_path))[0] + f"_shifted_{KEY_NAMES[target_key_index]}.wav"
        torchaudio.save(str(output_path), shifted_waveform, sr)
        result["shifted_wav_path"] = str(output_path)

    return result
