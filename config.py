"""Shared settings for every experiment. Change things here, not in the scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAVDESS_DIR = ROOT.parent / "CS Seminar FInal Project" / "datasets" / "ravdess"  # override with --ravdess-dir
FEATURES_FILE = ROOT / "data" / "features.npz"
RESULTS_DIR = ROOT / "results"

# --- Features (same as the original project) ---
SAMPLE_RATE = 22050
N_MFCC = 13
MAX_FRAMES = 173  # ~4 s of audio; shorter clips are zero-padded, longer ones cut

# RAVDESS emotion codes -> class index. "calm" (02) is left out, as in the original project.
EMOTIONS = {"05": "angry", "07": "disgust", "06": "fear", "03": "happy",
            "01": "neutral", "04": "sad", "08": "surprise"}
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(CLASS_NAMES)

# --- Speaker-independent test set: 2 male (odd ids) + 2 female (even ids) actors.
# Drawn once, before any training, with np.random.default_rng(0) choosing 2 odd
# and 2 even ids. Fixed so every experiment is scored on the same recordings.
TEST_SPEAKERS = [6, 8, 15, 19]

# Validation speakers for the early-stopped centralized baseline only: 1 male + 1 female
# from the 20 training speakers, drawn once with np.random.default_rng(1). They are used
# to choose the number of epochs, then put back into training.
VAL_SPEAKERS = [9, 16]

# --- Training budget ---
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_CLIENTS = 10
NUM_ROUNDS = 30
LOCAL_EPOCHS = 2
# Centralized training gets the same number of passes over the data as FL:
CENTRAL_EPOCHS = NUM_ROUNDS * LOCAL_EPOCHS  # = 60

SEEDS = [0, 1, 2]
