"""
End-to-End Model Training, Tuning, and Evaluation Pipeline
Executes:
1. Data loading and validation from GTZAN dataset.
2. Stratified train/test splitting (80/20) with random seed reproducibility.
3. Hyperparameter tuning using 5-fold Stratified Cross-Validation on training set:
   - Random Forest Classifier
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)
   - Logistic Regression
4. Data leakage prevention via Pipeline + StandardScaler.
5. Best model selection based on CV validation score.
6. Comprehensive test set evaluation:
   - Accuracy, Precision, Recall, F1-Score
   - Confusion matrix
   - Classification report
7. Visualization generation:
   - Model accuracy comparison
   - Confusion matrix heatmap
   - Per-genre Precision/Recall/F1
   - Feature importances
8. Model and artifact persistence using Joblib.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

# Ensure project root is in Python sys.path regardless of execution directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.dataset import load_and_preprocess_data, scan_and_validate_directory
from src.models import train_and_compare_all_models, save_pipeline
from src.evaluate import (
    evaluate_predictions,
    analyze_confusions,
    plot_model_comparison,
    plot_confusion_matrix,
    plot_per_genre_metrics,
    plot_feature_importances,
)


def run_pipeline(
    csv_path: str = None,
    test_size: float = 0.20,
    random_state: int = 42,
    cv_splits: int = 5,
):
    print("=" * 65)
    print("   MUSIC GENRE CLASSIFICATION - GTZAN ML PIPELINE")
    print("=" * 65)

    if csv_path is None:
        csv_path = os.path.join(BASE_DIR, "data", "features_30_sec.csv")

    saved_models_dir = os.path.join(BASE_DIR, "saved_models")
    reports_dir = os.path.join(BASE_DIR, "reports")
    os.makedirs(saved_models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Check directory health if raw audio exists
    raw_audio_dirs = [
        os.path.join(BASE_DIR, "data", "genres_original"),
        os.path.join(BASE_DIR, "data", "genres"),
        os.path.join(BASE_DIR, "genres"),
    ]
    for rad in raw_audio_dirs:
        if os.path.isdir(rad):
            print(f"\n[Validation] Checking raw audio directory: {rad}")
            report = scan_and_validate_directory(rad)
            print(f"  Genres found: {len(report['genres_found'])}/10")
            print(f"  Valid audio files: {report['valid_files_count']}")
            if report['corrupted_files']:
                print(f"  Corrupted files identified: {len(report['corrupted_files'])}")
                for cf in report['corrupted_files'][:5]:
                    print(f"    - {cf['file']} ({cf['genre']}): {cf['reason']}")
            break

    # 2. Load and Preprocess Dataset
    print(f"\n[Step 1] Loading preprocessed GTZAN features from: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"The feature dataset was not found at '{csv_path}'.\n"
            f"Please run 'python download_gtzan.py' or verify that 'data/features_30_sec.csv' exists."
        )

    X_train, X_test, y_train, y_test, label_encoder, feature_names = load_and_preprocess_data(
        csv_path=csv_path,
        test_size=test_size,
        random_state=random_state,
    )

    class_names = list(label_encoder.classes_)
    print(f"  Training samples: {len(X_train)} (80%)")
    print(f"  Testing samples:  {len(X_test)} (20%)")
    print(f"  Number of features: {len(feature_names)}")
    print(f"  Classes ({len(class_names)}): {', '.join(class_names)}")

    # 3. Model Training & Comparison across all 4 algorithms
    print("\n[Step 2] Training & Hyperparameter Tuning via 5-Fold Stratified CV...")
    comparison_results = train_and_compare_all_models(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        cv_splits=cv_splits,
        random_state=random_state,
    )

    best_name = comparison_results["best_model_name"]
    best_pipeline = comparison_results["best_pipeline"]
    summary = comparison_results["models_summary"]

    # 4. Evaluation on Held-Out Test Set
    print(f"\n[Step 3] Evaluating Selected Best Model ({best_name}) on Test Set...")
    y_pred = best_pipeline.predict(X_test)
    eval_metrics = evaluate_predictions(y_test, y_pred, class_names=class_names)

    print("\n" + "=" * 50)
    print(f"      FINAL EVALUATION RESULTS ({best_name})")
    print("=" * 50)
    print(f"Overall Accuracy:        {eval_metrics['accuracy'] * 100:.2f}%")
    print(f"Macro Precision:         {eval_metrics['precision_macro'] * 100:.2f}%")
    print(f"Macro Recall:            {eval_metrics['recall_macro'] * 100:.2f}%")
    print(f"Macro F1-Score:          {eval_metrics['f1_macro'] * 100:.2f}%")
    print(f"Weighted F1-Score:       {eval_metrics['f1_weighted'] * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(eval_metrics["classification_report_text"])

    # 5. Confusion Analysis
    print("\n[Step 4] Analyzing Most Frequently Confused Genres...")
    top_confusions = analyze_confusions(eval_metrics["confusion_matrix"], class_names, top_n=5)
    for idx, conf in enumerate(top_confusions, 1):
        print(f"\n{idx}. Misclassification: True '{conf['true_genre']}' predicted as '{conf['predicted_genre']}' ({conf['count']} tracks, {conf['percentage']:.1f}%)")
        print(f"   Acoustic Reason: {conf['explanation']}")

    # 6. Generate and Save Visualizations
    print("\n[Step 5] Generating Visualizations...")
    plot_model_comparison(summary, output_path=os.path.join(reports_dir, "model_comparison.png"))
    plot_confusion_matrix(
        eval_metrics["confusion_matrix"],
        class_names,
        output_path=os.path.join(reports_dir, "confusion_matrix.png"),
        title=f"Confusion Matrix - {best_name} (GTZAN Test Set)"
    )
    plot_per_genre_metrics(eval_metrics["per_genre_df"], output_path=os.path.join(reports_dir, "per_genre_metrics.png"))

    # Generate feature importances from Random Forest model
    rf_pipeline = summary["Random Forest"]["best_estimator"]
    plot_feature_importances(rf_pipeline, feature_names, output_path=os.path.join(reports_dir, "feature_importances.png"))

    # 7. Persist Models and Metadata
    print("\n[Step 6] Persisting Best Model Pipeline and Metadata...")
    model_save_path = os.path.join(saved_models_dir, "best_model_pipeline.joblib")
    encoder_save_path = os.path.join(saved_models_dir, "label_encoder.joblib")
    metadata_save_path = os.path.join(saved_models_dir, "model_metadata.json")

    save_pipeline(best_pipeline, model_save_path)
    joblib.dump(label_encoder, encoder_save_path)
    print(f"Saved Label Encoder to: {encoder_save_path}")

    # Prepare serializable metadata
    serializable_summary = {}
    for mname, mdata in summary.items():
        serializable_summary[mname] = {
            "cv_accuracy_mean": float(mdata["cv_accuracy_mean"]),
            "cv_accuracy_std": float(mdata["cv_accuracy_std"]),
            "test_accuracy": float(mdata["test_accuracy"]),
            "best_params": {k: str(v) for k, v in mdata["best_params"].items()},
            "training_time_sec": float(mdata["training_time_sec"]),
        }

    metadata = {
        "best_model_name": best_name,
        "best_cv_accuracy": float(comparison_results["best_cv_score"]),
        "test_accuracy": float(eval_metrics["accuracy"]),
        "macro_precision": float(eval_metrics["precision_macro"]),
        "macro_recall": float(eval_metrics["recall_macro"]),
        "macro_f1": float(eval_metrics["f1_macro"]),
        "models_summary": serializable_summary,
        "class_names": class_names,
        "feature_names": feature_names,
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "random_state": random_state,
        "top_confusions": top_confusions,
    }

    with open(metadata_save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata to: {metadata_save_path}")

    print("\n" + "=" * 65)
    print("Pipeline Execution Completed Successfully!")
    print("=" * 65)
    return metadata


if __name__ == "__main__":
    run_pipeline()
