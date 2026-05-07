# Extract loudness / RMS windows from audio for Friends stimuli (requires pip install .[av]).

import librosa
import numpy as np
import pyloudnorm as pyln

from naturalistic_encoding.config import STIMULI_ROOT, data_path

data_path("features").mkdir(parents=True, exist_ok=True)

for episode in range(2, 8):
    for variant in ["a", "b"]:
        stim = f"friends_s01e{episode:02d}{variant}"
        print(stim)
        wav_path = STIMULI_ROOT / f"s1{stim}.wav"
        if not wav_path.is_file():
            wav_path = STIMULI_ROOT / f"friends_s01" / f"s1{stim}.wav"
        data, rate = librosa.load(str(wav_path), sr=20000)
        data = data / np.max(np.abs(data))

        window_size = int(1.49 * rate)
        meter = pyln.Meter(rate)
        loudness_values = []
        rms_values = []

        for i in range(0, len(data), window_size):
            chunk = data[i : i + window_size]
            if len(chunk) < window_size:
                break
            rms = np.sqrt(np.mean(chunk**2))
            rms_values.append(rms)
            loudness = meter.integrated_loudness(chunk)
            loudness_values.append(loudness)

        loudness_array = np.array(loudness_values)
        rms_array = np.array(rms_values)
        np.save(data_path("features", f"{stim}_lla_rms"), rms_array)
        np.save(data_path("features", f"{stim}_lla_lufs"), loudness_array)
