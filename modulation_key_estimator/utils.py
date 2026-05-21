import os
import re
import glob

import librosa
import numpy as np
import torch
import torchaudio

from .keys import shortest_semitone_shift


def clean_filename(name):
    name_wo_ext = os.path.splitext(name)[0]
    name_ascii = re.sub(r"[^A-Za-z0-9_]", "_", name_wo_ext)
    name_ascii = re.sub(r"_+", "_", name_ascii)
    return name_ascii.lower()


def finalize_downloaded_wav(output_dir):
    wav_files = glob.glob(os.path.join(str(output_dir), "*.wav"))
    if not wav_files:
        raise FileNotFoundError("No .wav files found in the directory.")

    latest_file = max(wav_files, key=os.path.getctime)
    base_filename = clean_filename(os.path.basename(latest_file))

    dst = os.path.join(output_dir, f"{base_filename}.wav")

    count = 1
    while os.path.exists(dst):
        dst = os.path.join(output_dir, f"{base_filename}_{count}.wav")
        count += 1

    os.rename(latest_file, dst)
    return dst

def apply_eq_filter(waveform, sr=16000, low_cutoff=100, high_cutoff=8000, q=0.707):
    """Apply highpass and lowpass filtering."""
    waveform = torchaudio.functional.highpass_biquad(waveform, sr, low_cutoff, Q=q)
    if torch.isnan(waveform).any() or torch.isinf(waveform).any():
        waveform = torch.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
    waveform = torchaudio.functional.lowpass_biquad(waveform, sr, high_cutoff, Q=q)
    if torch.isnan(waveform).any() or torch.isinf(waveform).any():
        waveform = torch.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
    return waveform

def preprocess_waveform(waveform, sr=16000, target_rms=0.1, noise_reduce=True, eq_filter=True, low_cutoff=100, high_cutoff=8000):
    """Apply EQ filtering, RMS normalization, and optional noise reduction."""
    if eq_filter:
        waveform = apply_eq_filter(waveform, sr, low_cutoff, high_cutoff)
    
    waveform = waveform / (waveform.abs().max() + 1e-7)
    
    rms = waveform.pow(2).mean().sqrt()
    waveform = waveform * (target_rms / (rms + 1e-7))
    
    if noise_reduce:
        try:
            import noisereduce as nr

            y_np = waveform.squeeze().cpu().numpy()
            y_denoised = nr.reduce_noise(y_np, sr=sr)
            waveform = torch.tensor(y_denoised, dtype=torch.float32).unsqueeze(0)
        except Exception:
            pass
    
    return waveform

def compute_chromagram(waveform, sr=16000, n_fft=4096, hop_length=512):
    """Compute a 12-bin chromagram with librosa."""
    y = waveform.squeeze().cpu().numpy()
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    chroma = torch.tensor(chroma.T, dtype=torch.float32)
    return chroma

def compute_hpcp(waveform, sr=16000, frame_size=4096, hop_size=512):
    """Compute 12-bin HPCP features, with a librosa fallback."""
    y = waveform.squeeze().cpu().numpy()
    try:
        from essentia.standard import FrameGenerator, HPCP, Spectrum, Windowing

        window = Windowing(type="hann")
        spectrum = Spectrum(size=frame_size)
        hpcp_extractor = HPCP(size=12, sampleRate=sr)

        hpcp_frames = []
        for frame in FrameGenerator(y, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
            w = window(frame)
            mag = spectrum(w)
            freqs = np.linspace(0, sr / 2, len(mag))
            hpcp_frames.append(hpcp_extractor.compute(mag, freqs))

        if not hpcp_frames:
            raise ValueError("No HPCP frames produced.")
        return torch.tensor(np.stack(hpcp_frames), dtype=torch.float32)
    except Exception:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_size, n_chroma=12)
        return torch.tensor(chroma.T, dtype=torch.float32)

def pitch_shift_segments(
    waveform, 
    sr, 
    region_boundaries, 
    region_keys, 
    target_key_index=0,
    keys_linear=None,
):
    """Pitch-shift each detected region toward the target key."""
    hop_length = 512
    segments = []

    if keys_linear is None:
        keys_linear = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']

    for i in range(len(region_boundaries) - 1):
        start_frame = region_boundaries[i]
        end_frame = region_boundaries[i+1]

        start_sample = start_frame * hop_length
        end_sample = end_frame * hop_length

        if end_sample > waveform.shape[1]:
            end_sample = waveform.shape[1]

        segment_waveform = waveform[:, start_sample:end_sample].cpu().numpy().squeeze()

        region_key_index = region_keys[i]
        shift = shortest_semitone_shift(region_key_index, target_key_index)

        try:
            shifted_segment = librosa.effects.pitch_shift(
                segment_waveform, sr=sr, n_steps=shift
            )
        except Exception:
            shifted_segment = segment_waveform

        shifted_segment = np.expand_dims(shifted_segment, axis=0)
        segments.append(torch.tensor(shifted_segment, dtype=torch.float32, device=waveform.device))

    if segments:
        shifted_waveform = torch.cat(segments, dim=1)
    else:
        shifted_waveform = waveform

    return shifted_waveform
