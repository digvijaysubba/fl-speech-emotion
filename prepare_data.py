"""Step 0: turn RAVDESS .wav files into MFCC features (run once).

    python prepare_data.py [--ravdess-dir PATH]

Saves data/features.npz with X (n, 173, 13), y (n,), speaker (n,), filename (n,).
Byte-identical files (e.g. "...-10(1).wav" copies) are kept only once,
otherwise the same recording could land in both train and test.
"""
import argparse
import hashlib
from pathlib import Path

import librosa
import numpy as np

import config


def extract_mfcc(path):
    """Same feature as the original project: 13 MFCCs, padded/cut to 173 frames."""
    y, sr = librosa.load(path, sr=config.SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.N_MFCC)
    if mfcc.shape[1] < config.MAX_FRAMES:
        mfcc = np.pad(mfcc, ((0, 0), (0, config.MAX_FRAMES - mfcc.shape[1])))
    return mfcc[:, :config.MAX_FRAMES].T  # (frames, coefficients)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ravdess-dir", type=Path, default=config.RAVDESS_DIR)
    args = parser.parse_args()

    # Sorted so the result does not depend on the file system's listing order.
    # Files without "(" first, so the original name is the one we keep.
    files = sorted(args.ravdess_dir.glob("Actor_*/*.wav"), key=lambda p: ("(" in p.name, str(p)))
    seen_hashes, X, y, speaker, names = set(), [], [], [], []
    n_dupes = n_calm = 0
    for path in files:
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest in seen_hashes:
            n_dupes += 1
            continue
        seen_hashes.add(digest)
        # File name: modality-channel-EMOTION-intensity-statement-repetition-ACTOR.wav
        parts = path.stem.split("(")[0].split("-")
        if parts[2] not in config.EMOTIONS:
            n_calm += 1
            continue
        X.append(extract_mfcc(path))
        y.append(config.CLASS_NAMES.index(config.EMOTIONS[parts[2]]))
        speaker.append(int(parts[6]))
        names.append(path.name)

    X = np.asarray(X, dtype=np.float32)
    config.FEATURES_FILE.parent.mkdir(exist_ok=True)
    np.savez_compressed(config.FEATURES_FILE, X=X, y=np.array(y), speaker=np.array(speaker), filename=np.array(names))
    print(f"Found {len(files)} wav files, dropped {n_dupes} duplicates and {n_calm} 'calm' clips.")
    print(f"Saved {len(X)} recordings from {len(set(speaker))} speakers to {config.FEATURES_FILE}")
    print("Class counts:", dict(zip(config.CLASS_NAMES, np.bincount(y).tolist())))


if __name__ == "__main__":
    main()
