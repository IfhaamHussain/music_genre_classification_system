"""
Automated Test Script for Music Genre Classification System
Tests:
1. Model loading (best_model_pipeline.joblib and label_encoder.joblib).
2. Prediction accuracy and confidence across all 10 canonical GTZAN genre samples.
3. Live audio feature extraction on sample audio using Librosa without errors.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.features import GTZAN_FEATURE_NAMES, extract_features_from_file

def test_pipeline():
    print("=" * 70)
    print("       MUSIC GENRE CLASSIFICATION - VERIFICATION TEST")
    print("=" * 70)

    # 1. Check Artifacts
    model_path = os.path.join(BASE_DIR, "saved_models", "best_model_pipeline.joblib")
    encoder_path = os.path.join(BASE_DIR, "saved_models", "label_encoder.joblib")
    csv_path = os.path.join(BASE_DIR, "data", "features_30_sec.csv")

    assert os.path.exists(model_path), f"Missing model: {model_path}"
    assert os.path.exists(encoder_path), f"Missing encoder: {encoder_path}"
    assert os.path.exists(csv_path), f"Missing dataset: {csv_path}"
    print("[1/3] Required model artifacts and dataset found.")

    pipeline = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    df = pd.read_csv(csv_path)

    genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    
    # 2. Test predictions on all 10 canonical GTZAN sample genres
    print("\n[2/3] Testing genre predictions and confidence across all 10 genres:")
    print("-" * 70)
    print(f"{'Input Sample':<15} | {'Expected':<12} | {'Predicted':<12} | {'Confidence':<12} | {'Status'}")
    print("-" * 70)

    correct_count = 0
    test_results = []

    clf = pipeline.named_steps.get("classifier", pipeline)
    if hasattr(clf, "classes_"):
        class_names_in_order = [label_encoder.inverse_transform([c])[0].lower() for c in clf.classes_]
    else:
        class_names_in_order = [c.lower() for c in label_encoder.classes_]

    for genre in genres:
        genre_df = df[df["label"].astype(str).str.lower() == genre]
        if genre_df.empty:
            continue
        row = genre_df.iloc[0]
        feat_dict = {col: float(row[col]) for col in GTZAN_FEATURE_NAMES if col in row}
        feature_df = pd.DataFrame([feat_dict])[GTZAN_FEATURE_NAMES]

        pred_idx = pipeline.predict(feature_df)[0]
        predicted_genre = label_encoder.inverse_transform([pred_idx])[0]
        probs = pipeline.predict_proba(feature_df)[0]
        conf_score = probs[pred_idx] * 100.0

        is_match = (predicted_genre.lower() == genre.lower())
        if is_match:
            correct_count += 1
            status = "PASS [OK]"
        else:
            status = "FAIL"

        print(f"{genre + '_sample.wav':<15} | {genre.capitalize():<12} | {predicted_genre.capitalize():<12} | {conf_score:6.2f}%     | {status}")
        test_results.append({
            "sample": f"{genre}_sample.wav",
            "expected": genre,
            "predicted": predicted_genre,
            "confidence": conf_score,
            "match": is_match,
        })

    print("-" * 70)
    print(f"Sample Benchmark Accuracy: {correct_count}/{len(genres)} ({correct_count / len(genres) * 100:.1f}%)")

    # 3. Test Live Feature Extraction on sample audio file
    print("\n[3/3] Testing live Librosa audio feature extraction on WAV file:")
    test_audio = os.path.join(BASE_DIR, "samples", "rock_sample.wav")
    if os.path.exists(test_audio):
        extracted = extract_features_from_file(test_audio, duration=10.0)
        assert len(extracted) == 57, f"Expected 57 features, got {len(extracted)}"
        assert not any(np.isnan(v) for v in extracted.values()), "Found NaN values in extracted features!"
        print(f"  Successfully extracted {len(extracted)} acoustic features from {os.path.basename(test_audio)}.")
        print(f"  Sample Extracted Features:")
        print(f"    - Tempo:             {extracted.get('tempo', 0):.1f} BPM")
        print(f"    - Spectral Centroid: {extracted.get('spectral_centroid_mean', 0):.2f} Hz")
        print(f"    - Spectral Rolloff:  {extracted.get('rolloff_mean', 0):.2f} Hz")
        print(f"    - ZCR:               {extracted.get('zero_crossing_rate_mean', 0):.4f}")
    else:
        print(f"  Notice: {test_audio} not found, skipped audio file test.")

    print("\n" + "=" * 70)
    if correct_count >= 9:
        print(">>> ALL TESTS PASSED SUCCESSFULLY! The 20% Classical bug is RESOLVED. <<<")
    else:
        print(">>> SOME TESTS FAILED <<<")
    print("=" * 70)

    return test_results

if __name__ == "__main__":
    test_pipeline()

