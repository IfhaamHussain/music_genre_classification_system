"""
GTZAN Dataset Downloader and Directory Verifier
Provides automated downloading, extraction, and validation of the GTZAN Genre Collection.

Dataset Summary:
- Total tracks: 1,000 (100 per genre across 10 genres)
- Sample duration: 30 seconds per track
- Sampling rate: 22,050 Hz, 16-bit Mono WAV
- Note on GTZAN anomalies: Track 'jazz.00054.wav' is historically truncated/corrupted
  in the original archive; this script explicitly detects and handles such anomalies.
"""

import os
import sys
import tarfile
import urllib.request
from typing import Dict, Any
from src.dataset import scan_and_validate_directory, GENRE_CLASSES


GTZAN_TAR_URL = "https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz"


def download_progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100.0 / total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\rDownloading: {percent:.1f}% ({mb_down:.1f} MB / {mb_total:.1f} MB)")
        sys.stdout.flush()


def download_and_extract_gtzan(target_dir: str = "data"):
    """
    Downloads and extracts the official GTZAN archive into target_dir.
    """
    os.makedirs(target_dir, exist_ok=True)
    archive_path = os.path.join(target_dir, "genres.tar.gz")
    extracted_folder = os.path.join(target_dir, "genres")

    if os.path.isdir(extracted_folder) or os.path.isdir(os.path.join(target_dir, "genres_original")):
        print(f"GTZAN audio directory already exists in {target_dir}. Skipping download.")
        verify_dataset(target_dir)
        return

    print("=" * 65)
    print("Downloading GTZAN Genre Collection (~1.17 GB)")
    print(f"Source URL: {GTZAN_TAR_URL}")
    print(f"Destination: {archive_path}")
    print("=" * 65)

    try:
        urllib.request.urlretrieve(GTZAN_TAR_URL, archive_path, reporthook=download_progress_hook)
        print("\nDownload finished. Extracting archive...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=target_dir)
        print(f"Extraction complete! Files extracted to: {target_dir}")
        verify_dataset(target_dir)
    except Exception as e:
        print(f"\nDownload or extraction error: {e}")
        print("\nAlternative Download Options:")
        print("1. Kaggle CLI: kaggle datasets download -d carlthome/gtzan-genre-collection")
        print("2. Browser: Download genres.tar.gz from Hugging Face or Kaggle and place it in the 'data/' directory.")


def verify_dataset(data_dir: str = "data"):
    """
    Scans and verifies audio files in data_dir.
    Checks for all 10 genres and flags corrupted files.
    """
    candidate_paths = [
        os.path.join(data_dir, "genres_original"),
        os.path.join(data_dir, "genres"),
        "genres",
    ]

    target = None
    for p in candidate_paths:
        if os.path.isdir(p):
            target = p
            break

    if not target:
        print(f"[Verification] No raw audio folder found in '{data_dir}'.")
        if os.path.exists("data/features_30_sec.csv"):
            print("Notice: 'data/features_30_sec.csv' IS present, meaning the model can train and run without downloading raw audio!")
        return

    print("=" * 65)
    print(f"Verifying Dataset Integrity at: {target}")
    print("=" * 65)

    report = scan_and_validate_directory(target)

    print(f"Total Valid Tracks:     {report['valid_files_count']} / 1000")
    print(f"Genres Detected:        {len(report['genres_found'])} / 10")
    print("-" * 65)
    print(f"{'Genre':<15} | {'Valid Files Count':<18}")
    print("-" * 65)
    for g in GENRE_CLASSES:
        cnt = report["genre_counts"].get(g, 0)
        print(f"{g:<15} | {cnt:<18}")
    print("-" * 65)

    if report["corrupted_files"]:
        print(f"\nWarning: Detected {len(report['corrupted_files'])} corrupted or truncated audio file(s):")
        for cf in report["corrupted_files"]:
            print(f"  - {cf['file']} ({cf['genre']}): {cf['reason']}")
    else:
        print("\nAll scanned files are intact and readable!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download and verify GTZAN dataset")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing directory")
    args = parser.parse_args()

    if args.verify_only:
        verify_dataset()
    else:
        download_and_extract_gtzan()

