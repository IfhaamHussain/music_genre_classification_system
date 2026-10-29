# 🎵 Music Genre Classification System Using GTZAN Dataset

An end-to-end Machine Learning and Audio Processing system built with **Python**, **Librosa**, **Scikit-Learn**, and **Streamlit**. Designed to classify audio tracks into **10 musical genres** with high accuracy and explainability, ideal for academic demonstrations, BCA/B.Tech projects, and audio data science portfolios.

---

## 📌 1. Project Overview

Music Genre Classification is a core problem in **Music Information Retrieval (MIR)**. Given an audio recording (WAV or MP3), the system:
1. Decodes and standardizes the audio signal (resampling to 22,050 Hz mono).
2. Extracts **57 discriminative acoustic features** covering timbre, rhythm, frequency distribution, and harmony.
3. Passes the normalized feature vector through a trained Machine Learning pipeline.
4. Predicts the genre among 10 target classes and provides a calibrated probability score across all categories.

### Target Classes (10 Genres)
- **Blues**
- **Classical**
- **Country**
- **Disco**
- **Hip-Hop**
- **Jazz**
- **Metal**
- **Pop**
- **Reggae**
- **Rock**

---

## 📂 2. Dataset Information: GTZAN Genre Collection

The **GTZAN dataset** is the international benchmark dataset for music genre recognition, compiled by George Tzanetakis and Perry Cook (2002).

- **Total Tracks:** 1,000 audio files (100 tracks per genre).
- **Track Duration:** 30 seconds per track.
- **Audio Format:** 22,050 Hz, 16-bit Mono WAV.
- **Preprocessed Feature Table:** `data/features_30_sec.csv` (1,000 rows × 60 columns).

### Robust Handling of Corrupted/Missing Files
In standard GTZAN distributions, certain tracks (most notably `jazz.00054.wav`) are corrupted, truncated, or contain 0 bytes. Our project includes an automated integrity checker (`src/dataset.py` and `download_gtzan.py`) that:
- Inspects audio file sizes and container headers using `soundfile` and `librosa`.
- Catches EOF errors and skips corrupted tracks gracefully without crashing the pipeline.
- Operates out-of-the-box using the pre-extracted `data/features_30_sec.csv` dataset, while allowing users to download or inspect the raw audio files at any time.

---

## 🛠️ 3. Technology Stack

| Layer | Library / Tool | Purpose |
| :--- | :--- | :--- |
| **Audio Processing** | `Librosa`, `SoundFile` | Audio loading, mono downmixing, resampling, feature extraction |
| **Data Manipulation** | `NumPy`, `Pandas` | Feature vectors, tabular dataset operations, matrix statistics |
| **Machine Learning** | `Scikit-Learn` | Pipelines, StandardScaler, Hyperparameter Tuning, Classifiers |
| **Visualization** | `Matplotlib`, `Seaborn` | Confusion matrices, performance charts, waveforms, spectrograms |
| **Model Persistence**| `Joblib` | Serializing trained pipelines and label encoders |
| **Web Interface** | `Streamlit` | Interactive web dashboard with audio playback and real-time inference |

---

## 🎼 4. Audio Feature Extraction (All 8 Core Categories)

A raw 30-second audio track sampled at 22,050 Hz contains $661,500$ discrete amplitude samples. To make this data usable for traditional machine learning algorithms, we transform it into a fixed-length vector of **summary statistics** (mean and variance across time frames).

### Feature Details & Musical Intuition

| # | Feature | Summary Stats | What It Represents | Musical / Genre Significance |
|---|---|---|---|---|
| 1 | **MFCC (1 to 20)** | 20 means + 20 vars (40 features) | Timbre and spectral envelope mapped to the logarithmic Mel scale | Distinguishes vocal texture and instrument timbre (e.g. distorted electric guitars in Metal vs. acoustic brass/woodwinds in Classical/Jazz). |
| 2 | **Spectral Centroid** | Mean + Variance | "Center of mass" / brightness of frequency spectrum | Metal and Disco have bright, high-frequency energy (high centroid); Jazz and Blues have warmer, low-frequency presence. |
| 3 | **Spectral Bandwidth** | Mean + Variance | Width/spread of frequencies around the centroid | Full band arrangements with distorted saturation (Rock/Pop) have broad bandwidth compared to solo instruments. |
| 4 | **Spectral Rolloff** | Mean + Variance | Frequency below which 85% of total spectral energy lies | Separates genres with high-frequency percussion/cymbals (Metal/Disco) from lower-mid heavy music (Blues). |
| 5 | **Zero-Crossing Rate (ZCR)** | Mean + Variance | Rate at which the waveform changes sign (+/-) per frame | Distorted guitars, harsh vocals, and snares (Rock/Metal) produce high ZCR; smooth melodic notes (Country/Classical) produce low ZCR. |
| 6 | **Chroma Features** | Mean + Variance | Projection of energy onto 12 musical semitones (C, C#, ..., B) | Captures harmonic chord progressions and tonality. Distinguishes complex jazz chords from simple 3-chord pop loops. |
| 7 | **RMS Energy** | Mean + Variance | Instantaneous loudness and power across time frames | Measures dynamic range. Modern Pop/Metal has heavy dynamic compression (high, flat RMS); Classical features quiet pianissimos and loud fortissimos (high variance). |
| 8 | **Tempo (BPM)** | Scalar value | Underlying speed of the rhythmic pulse (beats per minute) | Disco (~124 BPM), Hip-Hop (~90 BPM), Reggae (~76 BPM), Metal (140-180+ BPM). |

Total feature vector length: **57 features** per audio track.

---

## ⚙️ 5. Data Preprocessing & Leakage Prevention

1. **Mono Downmixing & Resampling:** User-uploaded files are converted to mono at a standardized $22,050\text{ Hz}$.
2. **Label Encoding:** Genre strings are mapped into categorical integers using `LabelEncoder`.
3. **Stratified Splitting:** 
   - 80% Training Set ($800$ tracks, exactly 80 per genre)
   - 20% Held-Out Testing Set ($200$ tracks, exactly 20 per genre)
   - `random_state = 42` for exact reproducibility.
4. **Data Leakage Elimination:** Feature scaling (`StandardScaler`) is packaged directly inside `sklearn.pipeline.Pipeline`. The scaler is fitted exclusively on training folds during cross-validation, never seeing test set values before evaluation.

---

## 🔬 6. Model Training, Tuning & Comparison

Four traditional classification algorithms were trained and evaluated using **5-Fold Stratified Cross-Validation** on the 800 training tracks:

1. **Support Vector Machine (SVM):** Tuned across $C \in \{0.5, 1.0, 5.0, 10.0\}$, kernels (`rbf`, `linear`), and $\gamma$ (`scale`, `auto`).
2. **Random Forest Classifier:** Tuned across `n_estimators` $\in \{100, 200\}$, `max_depth` $\in \{\text{None}, 12, 20\}$, and `min_samples_split`.
3. **Logistic Regression:** Tuned across regularization strength $C \in \{0.05, 0.2, 1.0, 5.0\}$ with `lbfgs`.
4. **K-Nearest Neighbors (KNN):** Tuned across $k \in \{3, 5, 7, 9\}$, weights (`uniform`, `distance`), and distance metrics (`euclidean`, `manhattan`).

---

## 📈 7. Actual Experimental Results

> **Evaluation Setup:** Stratified 80/20 split ($800$ train / $200$ test). All metrics calculated on the unseen held-out test set.

### Algorithm Performance Comparison

| Model | 5-Fold CV Accuracy (Train) | Held-Out Test Accuracy | Training Time | Selected Status |
| :--- | :---: | :---: | :---: | :---: |
| **Support Vector Machine (SVM)** | **75.50% ± 1.91%** | **76.50%** | **3.98s** | 🏆 **BEST MODEL** |
| **Logistic Regression** | 72.12% ± 3.34% | 71.00% | 0.62s | Runner-up |
| **Random Forest** | 70.38% ± 2.87% | 68.50% | 20.47s | Evaluated |
| **K-Nearest Neighbors (KNN)** | 69.25% ± 1.83% | 68.50% | 0.98s | Evaluated |

*Model selection is based strictly on cross-validation validation performance on the training set.*

### Best Model (SVM with RBF Kernel, $C=5.0$) Classification Report

```text
              precision    recall  f1-score   support

       blues       0.77      0.85      0.81        20
   classical       0.90      0.90      0.90        20
     country       0.85      0.85      0.85        20
       disco       0.71      0.60      0.65        20
      hiphop       0.76      0.80      0.78        20
        jazz       0.71      0.85      0.77        20
       metal       1.00      0.75      0.86        20
         pop       0.78      0.70      0.74        20
      reggae       0.61      0.70      0.65        20
        rock       0.65      0.65      0.65        20

    accuracy                           0.77       200
   macro avg       0.77      0.77      0.77       200
weighted avg       0.77      0.77      0.77       200
```

### Visual Reports
The pipeline automatically generated high-resolution visualizations in `reports/`:
- `reports/model_comparison.png` — CV vs. Test Accuracy bar chart
- `reports/confusion_matrix.png` — 10x10 Annotated Confusion Matrix Heatmap
- `reports/per_genre_metrics.png` — Per-genre Precision, Recall, and F1-score chart
- `reports/feature_importances.png` — Top discriminative acoustic features

---

## 🔍 8. Error Analysis: Frequently Confused Genres

Analysis of the test confusion matrix reveals recurring acoustic and musical patterns:

1. **Metal confused with Blues (15%):**
   - *Reason:* Electric blues and early heavy metal share heavy minor pentatonic scales, tube-overdriven guitar amps, and similar rhythmic cadence.
2. **Pop confused with Reggae (15%):**
   - *Reason:* Contemporary pop frequently borrows syncopated dancehall/reggae drum rhythms, upbeat off-beat tempos, and prominent basslines.
3. **Classical confused with Jazz (10%):**
   - *Reason:* Both genres utilize purely acoustic instruments (piano, strings, brass), uncompressed dynamic ranges (varying RMS energy), and complex chord structures.
4. **Disco confused with Pop (10%):**
   - *Reason:* Both feature four-on-the-floor kick drums, high dance tempos (~120-125 BPM), bright synthesizer leads, and high spectral rolloff.
5. **Disco confused with Hip-Hop (10%):**
   - *Reason:* Hip-hop originally evolved by sampling disco drum breaks; both emphasize strong low-end grooves and syncopated percussion.

---

## 💻 9. Streamlit Web Application

The interactive web application provides:
- **Audio File Upload:** Supports `.wav`, `.mp3`, `.ogg`, and `.flac`.
- **Pre-loaded Audio Samples:** 10 pre-synthesized tracks in `samples/` for instant one-click testing without uploading files.
- **In-Browser Audio Playback:** Listen to the audio before classifying.
- **Prediction Card:** Displays predicted genre with color-coded confidence score.
- **Probability Breakdown:** Horizontal bar chart showing calibrated probabilities across all 10 classes.
- **Acoustic Metrics Summary:** Displays estimated Tempo (BPM), Spectral Centroid (Hz), Spectral Rolloff (Hz), and Zero-Crossing Rate.
- **Waveform Visualizer:** Displays the normalized mono audio waveform.
- **Educational Tabs:** In-depth guides on audio features, model comparison charts, and dataset methodology.

---

## 🚀 10. How to Run the Project

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: (Optional) Re-train the Machine Learning Models
```bash
python train.py
```
*This loads `data/features_30_sec.csv`, performs cross-validation, trains all 4 models, generates plots in `reports/`, and saves the best model to `saved_models/`.*

### Step 3: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Step 4: (Optional) Download Full GTZAN Raw Audio
```bash
python download_gtzan.py
```
To only verify directory integrity:
```bash
python download_gtzan.py --verify-only
```

---

## 📁 Project Directory Structure

```text
├── data/
│   └── features_30_sec.csv          # GTZAN preprocessed feature dataset (1,000 tracks)
├── reports/
│   ├── model_comparison.png        # Bar chart comparing all 4 algorithms
│   ├── confusion_matrix.png        # 10x10 Heatmap of test predictions
│   ├── per_genre_metrics.png       # Grouped bar chart of Precision, Recall, F1
│   └── feature_importances.png     # Top discriminative acoustic features
├── saved_models/
│   ├── best_model_pipeline.joblib   # Trained SVM pipeline with StandardScaler
│   ├── label_encoder.joblib        # Scikit-learn LabelEncoder mapping strings to IDs
│   └── model_metadata.json         # Evaluation metrics, parameters, and metadata
├── samples/                        # Pre-synthesized playable test audio files
│   ├── blues_sample.wav
│   ├── classical_sample.wav
│   ├── country_sample.wav
│   ├── disco_sample.wav
│   ├── hiphop_sample.wav
│   ├── jazz_sample.wav
│   ├── metal_sample.wav
│   ├── pop_sample.wav
│   ├── reggae_sample.wav
│   └── rock_sample.wav
├── src/
│   ├── __init__.py
│   ├── dataset.py                  # Dataset validation, loader, and stratified splitter
│   ├── features.py                 # Librosa acoustic feature extraction (57 features)
│   ├── models.py                   # Model pipelines, grid search CV tuning & comparison
│   ├── evaluate.py                 # Evaluation metrics, confusion analysis, and plotting
│   └── utils.py                    # Audio loaders, feature explanations, JSON helpers
├── app.py                          # Streamlit interactive web application
├── download_gtzan.py               # Dataset downloader and directory integrity checker
├── generate_samples.py             # Audio sample synthesis script
├── requirements.txt                # Project dependencies
├── train.py                        # Complete end-to-end training and evaluation script
└── README.md                       # Comprehensive project documentation
```

