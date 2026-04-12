import librosa
import numpy as np


def load_audio(file_path, target_sr=16000):
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    print(f"Loaded: {file_path} | Duration: {len(audio)/sr:.2f}s | SR: {sr}")
    return audio, sr


def normalize_audio(audio):
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    return audio / peak


def reduce_noise(audio, sr, noise_frame_count=10):
    stft = librosa.stft(audio)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # use first few frames as noise estimate
    noise_profile = np.mean(magnitude[:, :noise_frame_count], axis=1, keepdims=True)
    cleaned = np.maximum(magnitude - noise_profile, 0)

    cleaned_stft = cleaned * np.exp(1j * phase)
    return librosa.istft(cleaned_stft, length=len(audio))


def preprocess(file_path, target_sr=16000):
    audio, sr = load_audio(file_path, target_sr)
    audio = normalize_audio(audio)
    audio = reduce_noise(audio, sr)
    audio = normalize_audio(audio)  # renormalize after noise reduction
    return audio, sr


def print_results(file_path, pause_segments, total_pause, repetition_info):
    print("\n=== SPEECH ANALYSIS RESULTS ===")
    print(f"\nFile: {file_path}")

    print("\n Pause Detection:")
    if pause_segments:
        for s, e in pause_segments:
            print(f"  [{s:.2f}s - {e:.2f}s]")
    else:
        print("  No pauses detected")
    print(f"  Total pause time: {total_pause:.2f}s")

    print("\n Repetition Detection:")
    count = repetition_info.get("count", 0)
    positions = repetition_info.get("positions", [])
    examples = repetition_info.get("examples", [])

    if count > 0:
        print(f"  Repetitions found: {count}")
        print(f"  At positions: {', '.join(f'{p:.2f}s' for p in positions)}")
        for e in examples:
            print(f"  {e}")
    else:
        print("  No repetitions detected")

    print("\n================================\n")