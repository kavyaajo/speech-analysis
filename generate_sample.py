import numpy as np
import os
import scipy.io.wavfile as wav

SR = 16000

def make_tone(freq, duration, amplitude=0.5):
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)

def make_silence(duration):
    return np.zeros(int(SR * duration), dtype=np.float32)

#create a sample audio file with pauses and some repeated segments.
def build_sample():
    parts = [
        make_tone(300, 0.30),  # word 1
        make_silence(0.50),    # pause 1
        make_tone(400, 0.20),  # ba (repeat 1)
        make_tone(400, 0.20),  # ba (repeat 2)
        make_tone(400, 0.20),  # ba (repeat 3)
        make_tone(250, 0.30),  # ball
        make_silence(0.60),    # pause 2
        make_tone(350, 0.40),  # word 2
        make_tone(450, 0.30),  # word 3
    ]

    audio = np.concatenate(parts)

    # adding a small noise to simulate real-world audio
    noise = np.random.normal(0, 0.005, size=len(audio)).astype(np.float32)
    audio = audio + noise
    audio = np.clip(audio, -1.0, 1.0)

    os.makedirs("sample_audio", exist_ok=True)
    out_path = os.path.join("sample_audio", "sample.wav")
    wav.write(out_path, SR, (audio * 32767).astype(np.int16))

    print(f"saved to {out_path}")
    print(f"Audio duration: {len(audio)/SR:.2f} seconds")

if __name__ == "__main__":
    build_sample()