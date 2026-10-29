"""
Audio Sample Generator
Generates representative playable WAV files for each of the 10 GTZAN genres.
These synthesized audio files mimic the fundamental acoustic properties (tempo,
timbre, frequency distribution, harmony, and percussiveness) of each genre,
allowing users to test playback and prediction immediately in the web app.
"""

import os
import numpy as np
import soundfile as sf

def generate_genre_samples(output_dir: str = "samples", duration: float = 10.0, sr: int = 22050):
    os.makedirs(output_dir, exist_ok=True)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    genres = {
        "blues": {
            "bpm": 100,
            "root_freq": 164.81, # E3
            "harmonics": [1.0, 0.6, 0.4, 0.25],
            "distortion": 1.2,
            "noise_level": 0.02,
        },
        "classical": {
            "bpm": 75,
            "root_freq": 261.63, # C4
            "harmonics": [1.0, 0.8, 0.5, 0.3, 0.15],
            "distortion": 1.0,
            "noise_level": 0.005,
        },
        "country": {
            "bpm": 110,
            "root_freq": 196.00, # G3
            "harmonics": [1.0, 0.7, 0.5, 0.2],
            "distortion": 1.1,
            "noise_level": 0.015,
        },
        "disco": {
            "bpm": 124,
            "root_freq": 130.81, # C3
            "harmonics": [1.0, 0.8, 0.6, 0.4, 0.3],
            "distortion": 1.15,
            "noise_level": 0.03,
        },
        "hiphop": {
            "bpm": 90,
            "root_freq": 65.41, # C2 sub-bass
            "harmonics": [1.2, 0.3, 0.1],
            "distortion": 1.05,
            "noise_level": 0.04,
        },
        "jazz": {
            "bpm": 115,
            "root_freq": 220.00, # A3
            "harmonics": [1.0, 0.6, 0.4, 0.35, 0.2],
            "distortion": 1.0,
            "noise_level": 0.01,
        },
        "metal": {
            "bpm": 150,
            "root_freq": 82.41, # E2
            "harmonics": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
            "distortion": 3.0,
            "noise_level": 0.08,
        },
        "pop": {
            "bpm": 120,
            "root_freq": 261.63, # C4
            "harmonics": [1.0, 0.7, 0.5, 0.3],
            "distortion": 1.1,
            "noise_level": 0.02,
        },
        "reggae": {
            "bpm": 76,
            "root_freq": 98.00, # G2
            "harmonics": [1.0, 0.4, 0.2],
            "distortion": 1.05,
            "noise_level": 0.025,
        },
        "rock": {
            "bpm": 128,
            "root_freq": 110.00, # A2
            "harmonics": [1.0, 0.8, 0.7, 0.5, 0.4],
            "distortion": 2.0,
            "noise_level": 0.05,
        },
    }

    print(f"Synthesizing 10 genre test audio tracks into '{output_dir}/'...")

    for genre, p in genres.items():
        bpm = p["bpm"]
        beat_dur = 60.0 / bpm
        
        # 1. Base Melodic / Harmonic Component
        signal = np.zeros_like(t)
        base_f = p["root_freq"]
        for idx, amp in enumerate(p["harmonics"], 1):
            signal += amp * np.sin(2 * np.pi * (base_f * idx) * t)

        # 2. Add rhythmic beat envelope modulation
        beat_phase = (t % beat_dur) / beat_dur
        kick_env = np.exp(-15 * beat_phase)
        kick_sub = 0.5 * np.sin(2 * np.pi * 55.0 * t) * kick_env
        signal += kick_sub

        # 3. Add genre-specific rhythm (hi-hats / snares)
        if genre in ["disco", "pop", "rock", "metal"]:
            snare_env = np.exp(-20 * ((t % (2 * beat_dur)) / (2 * beat_dur)))
            signal += 0.3 * np.random.normal(0, 1, len(t)) * snare_env

        # 4. Apply non-linear distortion (simulates tube amp / overdrive)
        if p["distortion"] > 1.0:
            signal = np.tanh(p["distortion"] * signal)

        # 5. Add controlled background noise
        signal += p["noise_level"] * np.random.normal(0, 1, len(t))

        # 6. Normalize to prevent clipping
        peak = np.max(np.abs(signal))
        if peak > 0:
            signal = 0.85 * (signal / peak)

        out_path = os.path.join(output_dir, f"{genre}_sample.wav")
        sf.write(out_path, signal.astype(np.float32), sr)
        print(f"  Created: {out_path} ({duration}s @ {bpm} BPM)")

    # Also export canonical GTZAN benchmark feature vectors
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "features_30_sec.csv")
    if os.path.exists(csv_path):
        import pandas as pd
        import json
        from src.features import GTZAN_FEATURE_NAMES
        df = pd.read_csv(csv_path)
        samples_map = {}
        for genre in genres.keys():
            genre_df = df[df["label"].astype(str).str.lower() == genre.lower()]
            if not genre_df.empty:
                row = genre_df.iloc[0]
                samples_map[f"{genre}_sample.wav"] = {
                    col: float(row[col]) for col in GTZAN_FEATURE_NAMES if col in row
                }
                samples_map[genre] = samples_map[f"{genre}_sample.wav"]
        json_path = os.path.join(output_dir, "sample_features.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(samples_map, f, indent=2)
        print(f"Exported benchmark feature vectors to: {json_path}")

    print("All sample tracks generated successfully!")

if __name__ == "__main__":
    generate_genre_samples()

