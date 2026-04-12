#repetition detection using MFCC similarity + cosine similarity to find repeated speech patterns in audio.
import numpy as np
import librosa
from scipy.spatial.distance import cosine


def extract_mfcc_windows(audio, sr, window_duration=0.20, hop_duration=0.10, n_mfcc=13):
    window_samples = int(window_duration * sr)
    hop_samples = int(hop_duration * sr)

    mfcc_matrix = []
    window_times = []

    start = 0
    while start + window_samples <= len(audio):
        segment = audio[start : start + window_samples]
        mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
        mfcc_matrix.append(np.mean(mfccs, axis=1))
        window_times.append(start / sr)
        start += hop_samples

    return np.array(mfcc_matrix), np.array(window_times)


def cosine_similarity(vec_a, vec_b):
    if np.all(vec_a == 0) or np.all(vec_b == 0):
        return 0.0
    return 1.0 - cosine(vec_a, vec_b)


def compute_similarity_profile(mfcc_matrix):
    n = len(mfcc_matrix)
    sim_profile = np.zeros(n - 1)
    for i in range(n - 1):
        #detect repeated speech patterns using MFCC similarity
        sim_profile[i] = cosine_similarity(mfcc_matrix[i], mfcc_matrix[i + 1])
    return sim_profile


def detect_repetitions(audio, sr, window_duration=0.20, hop_duration=0.10,
                        similarity_threshold=0.92, min_repeat_count=2):

    mfcc_matrix, window_times = extract_mfcc_windows(audio, sr, window_duration, hop_duration)

    if len(mfcc_matrix) < 3:
        return {"count": 0, "positions": [], "examples": []}
    sim_profile = compute_similarity_profile(mfcc_matrix)
    # mark segments where similarity is high (possible repetitions)
    similar_mask = sim_profile >= similarity_threshold

    events = []
    in_event = False
    event_start = 0

    for i, is_similar in enumerate(similar_mask):
        if is_similar and not in_event:
            in_event = True
            event_start = i
        elif not is_similar and in_event:
            in_event = False
            events.append((event_start, i))

    if in_event:
        events.append((event_start, len(similar_mask)))

    positions = []
    examples = []

    for es, ee in events:
        repeat_count = ee - es + 1
        if repeat_count < min_repeat_count:
            continue
        start_time = window_times[es]
        end_time = window_times[min(ee, len(window_times) - 1)]
        positions.append(round(start_time, 2))
        examples.append(f"Repeat @ {start_time:.2f}s-{end_time:.2f}s ({repeat_count} similar windows)")

    return {"count": len(positions), "positions": positions, "examples": examples}
