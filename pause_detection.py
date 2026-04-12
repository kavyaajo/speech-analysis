#detect pauses in audio based on low energy(RMS) segments.
import numpy as np
import librosa


def compute_rms(audio, frame_length=512, hop_length=256):
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    return rms


def detect_pauses(audio, sr, frame_length=512, hop_length=256,
                  energy_threshold=0.02, min_pause_duration=0.20):

    rms = compute_rms(audio, frame_length=frame_length, hop_length=hop_length)
    silent_mask = rms < energy_threshold

    pause_segments = []
    in_pause = False
    start_frame = 0

    for i, is_silent in enumerate(silent_mask):
        if is_silent and not in_pause:
            in_pause = True
            start_frame = i
        elif not is_silent and in_pause:
            in_pause = False
            pause_segments.append((start_frame, i)) # end of a pause segment

    if in_pause:
        pause_segments.append((start_frame, len(silent_mask) - 1))

    pause_times = []
    for sf, ef in pause_segments:
        t_start = librosa.frames_to_time(sf, sr=sr, hop_length=hop_length)
        t_end = librosa.frames_to_time(ef, sr=sr, hop_length=hop_length)
        if (t_end - t_start) >= min_pause_duration:
            pause_times.append((round(t_start, 3), round(t_end, 3)))

    total_pause_dur = sum(e - s for s, e in pause_times)
    return pause_times, round(total_pause_dur, 3)