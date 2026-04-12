#main script to run speech analysis for pause and repetition detection.
import argparse
import sys
import os

from utils import preprocess, load_audio, normalize_audio, print_results
from pause_detection import detect_pauses
from repetition_detection import detect_repetitions


def parse_args():
    parser = argparse.ArgumentParser(description="Speech Analysis - Pause & Repetition Detection")

    parser.add_argument("audio_file", help="path to .wav file")
    parser.add_argument("--threshold", "-t", type=float, default=0.02, help="energy threshold for pause detection")
    parser.add_argument("--min-pause", type=float, default=0.20, help="minimum pause duration in seconds")
    parser.add_argument("--sim", "-s", type=float, default=0.92, help="similarity threshold for repetition detection")
    parser.add_argument("--window", type=float, default=0.20, help="mfcc window size in seconds")
    parser.add_argument("--hop", type=float, default=0.10, help="mfcc hop size in seconds")
    parser.add_argument("--no-noise-reduction", action="store_true", default=False)
    parser.add_argument("--sr", type=int, default=16000, help="sample rate")

    return parser.parse_args()


def run_analysis(audio_file, threshold=0.02, min_pause=0.20, sim=0.92,
                 window=0.20, hop=0.10, no_noise_reduction=False, sr=16000):

    print("\nStep 1: Preprocessing...")
    if no_noise_reduction:
        audio, sr = load_audio(audio_file, target_sr=sr)
        audio = normalize_audio(audio)
    else:
        audio, sr = preprocess(audio_file, target_sr=sr)

    print("\nStep 2: Detecting pauses...")
    pause_segments, total_pause = detect_pauses(
        audio, sr=sr, energy_threshold=threshold, min_pause_duration=min_pause)
    print(f"Detected {len(pause_segments)} pauses")

    print("\nStep 3: Detecting repetitions...")
    repetition_info = detect_repetitions(
        audio, sr=sr, window_duration=window, hop_duration=hop, similarity_threshold=sim)
    print(f"Detected {repetition_info['count']} repetitions")

    print_results(audio_file, pause_segments, total_pause, repetition_info)

    return {
        "pause_segments": pause_segments,
        "total_pause": total_pause,
        "repetition_info": repetition_info,
    }


if __name__ == "__main__":
    args = parse_args()

    if not os.path.isfile(args.audio_file):
        print(f"File not found: {args.audio_file}")
        sys.exit(1)

    run_analysis(
        audio_file=args.audio_file,
        threshold=args.threshold,
        min_pause=args.min_pause,
        sim=args.sim,
        window=args.window,
        hop=args.hop,
        no_noise_reduction=args.no_noise_reduction,
        sr=args.sr,
    )