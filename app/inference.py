import torch
import torchaudio
import os
import numpy as np
import torch.nn.functional as F
from utils import preprocess_waveform, compute_chromagram, compute_hpcp, pitch_shift_segments
from model_loader import load_model

def run_inference(wav_path, model, target_key_index=0):
    sr = 16000
    window_size = 1024
    stride = 64
    hop_length = 512
    device = "cuda" if torch.cuda.is_available() else "cpu"
    keys_linear = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']
    prob_threshold = 0.32

    def frame_to_time(frame_index):
        return frame_index * hop_length / sr

    # Load waveform
    waveform, original_sr = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if original_sr != sr:
        waveform = torchaudio.transforms.Resample(original_sr, sr)(waveform)
    processed_waveform = preprocess_waveform(waveform, sr=sr)

    # Feature extraction
    chroma = compute_chromagram(processed_waveform, sr=sr)
    hpcp = compute_hpcp(processed_waveform, sr=sr)
    min_len = min(chroma.shape[0], hpcp.shape[0])
    feats_full = torch.cat([chroma[:min_len], hpcp[:min_len]], dim=1)

    # Sliding window inference
    windows = []
    for start in range(0, feats_full.shape[0] - window_size + 1, stride):
        windows.append(start)

    logits_list = []
    with torch.no_grad():
        for start in windows:
            window_feats = feats_full[start:start+window_size]
            feats = window_feats.unsqueeze(0).to(device)
            logits = model(feats)
            probs = F.softmax(logits, dim=-1)
            logits_list.append(probs.cpu())
    logits_tensor = torch.cat(logits_list, dim=0)  # [N, 12]

    # Modulation Point Detection
    max_prob_diff = 0
    mod_point_index = -1

    for i in range(len(windows) - 16):
        left_probs = logits_tensor[i]
        right_probs = logits_tensor[i + 16]
        left_key = left_probs.argmax().item()
        right_key = right_probs.argmax().item()

        if left_key != right_key:
            prob_diff = torch.abs(left_probs - right_probs).sum().item()
            if prob_diff >= prob_threshold and prob_diff > max_prob_diff:
                max_prob_diff = prob_diff
                mod_point_index = i + 16

    if mod_point_index >= 0:
        mod_frames = [windows[mod_point_index]]
    else:
        mod_frames = []

    print("Detected Modulation Point (frames):", mod_frames)

    # Region Split & Key Assignment
    region_boundaries = [0] + mod_frames + [feats_full.shape[0]]
    region_boundaries = sorted(region_boundaries)
    region_keys = []

    print("===== Region Split & Key Assignment =====")
    for i in range(len(region_boundaries) - 1):
        start_frame = region_boundaries[i]
        end_frame = region_boundaries[i + 1]
        if end_frame - start_frame < window_size:
            print(f"⚠️ Skipping short region: [{start_frame}:{end_frame}] (too short)")
            continue

        region_windows = []
        for start in range(start_frame, end_frame - window_size + 1, stride):
            window_feats = feats_full[start:start+window_size]
            region_windows.append(window_feats)

        if not region_windows:
            print(f"⚠️ Region [{start_frame}:{end_frame}] too short for windowing.")
            continue

        region_logits = []
        with torch.no_grad():
            for window_feats in region_windows:
                feats = window_feats.unsqueeze(0).to(device)
                logits = model(feats)
                probs = F.softmax(logits, dim=-1)
                region_logits.append(probs.cpu())
        region_logits = torch.cat(region_logits, dim=0)
        mean_probs = region_logits.mean(dim=0)
        region_key = mean_probs.argmax().item()
        start_time = start_frame * hop_length / sr
        end_time = end_frame * hop_length / sr
        start_str = f"{int(start_time // 60):02}:{int(start_time % 60):02}"
        end_str = f"{int(end_time // 60):02}:{int(end_time % 60):02}"

        print(f"Region {i+1}: [{start_str} - {end_str}]")
        print(f"  Mean Probabilities: {mean_probs.tolist()}")
        print(f"  Assigned Key Index: {region_key} ({keys_linear[region_key].upper()})\n")

        region_keys.append(region_key)

    # Pitch shift (region-based)
    shifted_waveform = pitch_shift_segments(
        waveform,
        sr,
        region_boundaries,
        region_keys,
        target_key_index=target_key_index,
        keys_linear=keys_linear
    )

    # Save shifted waveform
    output_path = os.path.splitext(wav_path)[0] + f"_shifted_{keys_linear[target_key_index]}.wav"
    torchaudio.save(output_path, shifted_waveform, sr)
    print(f"Pitch-shifted file saved at: {output_path}")

    # 🔥 Build region info
    region_infos = []
    for i, (start_frame, key_index) in enumerate(zip(region_boundaries[:-1], region_keys)):
        region_infos.append({
            "region_index": i,
            "start_time_sec": frame_to_time(start_frame),
            "key_index": key_index,
            "key_name": keys_linear[key_index].upper()
        })

    return {
        "predicted_region_keys": region_keys,
        "region_infos": region_infos,
        "target_key_index": target_key_index,
        "shifted_wav_path": output_path
    }
