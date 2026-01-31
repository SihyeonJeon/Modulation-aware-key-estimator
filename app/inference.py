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
    prob_threshold = 0.35
    min_distance = 12  # 🔥 최소 간격 설정

    def frame_to_time(frame_index):
        return frame_index * hop_length / sr

    # Load waveform
    waveform_origin, original_sr = torchaudio.load(wav_path)
    if waveform_origin.shape[0] > 1:
        waveform_origin = waveform_origin.mean(dim=0, keepdim=True)
    if original_sr != sr:
        waveform_origin = torchaudio.transforms.Resample(original_sr, sr)(waveform_origin)
    waveform = preprocess_waveform(waveform_origin, sr=sr)

    # Feature extraction
    chroma = compute_chromagram(waveform, sr=sr)
    hpcp = compute_hpcp(waveform, sr=sr)
    min_len = min(chroma.shape[0], hpcp.shape[0])
    feats_full = torch.cat([chroma[:min_len], hpcp[:min_len]], dim=1)

    # Sliding window inference
    input_batch = []
    windows = []  # start 인덱스를 저장할 리스트

    # stride: 64는 겹치는 구간이 많아 정확도는 높지만 연산량이 많습니다. 
    # 속도가 중요하다면 128이나 256으로 늘려도 무방합니다.
    for start in range(0, feats_full.shape[0] - window_size + 1, stride):
        window = feats_full[start : start + window_size]
        input_batch.append(window)
        windows.append(start)  # [중요] 나중에 시간 계산을 위해 인덱스 저장 필수

    # 예외 처리: 곡이 너무 짧아서 윈도우가 하나도 안 만들어진 경우
    if not input_batch:
        print("Audio is too short to analyze.")
        return None # 또는 적절한 에러 처리

    # 리스트를 하나의 큰 텐서로 변환 [N, Window_Size(1024), Feat_Dim(24)]
    input_tensor = torch.stack(input_batch).to(device)

    # ---------------------------------------------------------
    # 2. Real Batch Inference (여기가 핵심)
    # ---------------------------------------------------------
    # for문 없이 한 번에 모델에 넣습니다.
    # N이 약 100~200개 정도이므로(4분 곡 기준), 미니배치 없이 한 번에 넣어도 메모리 충분합니다.
    with torch.no_grad():
        logits = model(input_tensor)    # [N, 12] 결과가 한 번에 나옴
        probs = F.softmax(logits, dim=-1)
    
    # 결과를 CPU로 가져옴
    logits_tensor = probs.cpu()  # [N, 12]

    # ---------------------------------------------------------
    # 3. Modulation Point Detection (기존 로직 유지)
    # ---------------------------------------------------------
    mod_point_candidates = []
    
    # 벡터화된 연산으로 속도 더 높이기 (옵션)
    # 파이썬 for문도 N이 작아서 문제없으나, 아래처럼 짜면 더 깔끔합니다.
    # N = logits_tensor.shape[0]
    # margin = 16
    # if N > margin:
    #     left_probs = logits_tensor[:-margin]
    #     right_probs = logits_tensor[margin:]
    #     ... (이후 로직은 동일)

    # 기존 for문 로직 사용 시:
    for i in range(len(windows) - 16): 
        # windows 리스트가 채워져 있어야 이 루프가 돕니다.
        left_probs = logits_tensor[i]
        right_probs = logits_tensor[i + 16]
        left_key = left_probs.argmax().item()
        right_key = right_probs.argmax().item()

        if left_key != right_key:
            prob_diff = torch.abs(left_probs - right_probs).sum().item()
            if prob_diff >= prob_threshold:
                mod_point_candidates.append((i + 16, prob_diff))

    # 정렬 및 min_distance 필터링
    mod_point_candidates.sort(key=lambda x: -x[1])  # prob_diff 내림차순
    selected_mod_frames = []
    for idx, _ in mod_point_candidates:
        if all(abs(idx - sel) >= min_distance for sel in selected_mod_frames):
            selected_mod_frames.append(idx)

    mod_frames = [windows[i] for i in selected_mod_frames]
    print("Detected Modulation Points (frames):", mod_frames)

    # Region Split & Key Assignment
    region_boundaries = [0] + sorted(mod_frames) + [feats_full.shape[0]]
    region_keys = []

    print("===== Region Split & Key Assignment =====")
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
            print(f"⚠️ Region [{start_frame}:{end_frame}] too short for windowing.")
            region_keys.append(-1)  # dummy
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

        start_time = frame_to_time(start_frame)
        end_time = frame_to_time(end_frame)

        start_min = int(start_time // 60)
        start_sec = start_time % 60
        end_min = int(end_time // 60)
        end_sec = end_time % 60

        print(f"Region {i+1}: [{start_min:02d}:{start_sec:05.2f} - {end_min:02d}:{end_sec:05.2f}]")
        print(f"  Mean Probabilities: {mean_probs.tolist()}")
        print(f"  Assigned Key Index: {region_key} ({keys_linear[region_key].upper()})\n")

        region_keys.append(region_key)

    # Pitch shift (region-based)
    shifted_waveform = pitch_shift_segments(
        waveform_origin,
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
