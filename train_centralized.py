"""Centralized baselines: all training data on one machine, normal Keras training.

    python train_centralized.py --split speaker --seed 0              # plain, 60 epochs
    python train_centralized.py --split speaker --seed 0 --val-stop   # early-stopped
    python train_centralized.py --split random  --seed 0              # random 80/20 split

Plain: trains for a fixed budget (CENTRAL_EPOCHS = rounds x local epochs, the same
number of passes over the data as federated training); the final model is reported.

--val-stop (stronger baseline, never looks at the test set):
  1. hold out the 2 validation speakers (config.VAL_SPEAKERS) from the 20 training speakers,
  2. train on the other 18 for up to 60 epochs, pick the epoch with the best
     validation macro-F1,
  3. retrain from scratch on all 20 speakers for that many epochs and report the test score.

Test scores are logged each epoch only for the plots; they never choose anything.
"""
import argparse
import json

import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow import keras

import config
import data
from model import build_model, evaluate


def train(seed, X_tr, y_tr, epochs, X_eval, y_eval, tag):
    """Train a fresh model for `epochs`, scoring (X_eval, y_eval) after every epoch."""
    keras.utils.set_random_seed(seed)  # seeds Python, NumPy and TensorFlow
    model = build_model()
    history = []
    for epoch in range(1, epochs + 1):
        fit = model.fit(X_tr, y_tr, epochs=1, batch_size=config.BATCH_SIZE, verbose=0)
        m = evaluate(model, X_eval, y_eval)
        history.append({"epoch": epoch, "train_loss": fit.history["loss"][0], **m})
        print(f"[{tag}] epoch {epoch:2d}  loss={history[-1]['train_loss']:.3f}  "
              f"acc={m['accuracy']:.4f}  macro-F1={m['macro_f1']:.4f}")
    return model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["speaker", "random"], default="speaker")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=config.CENTRAL_EPOCHS)
    parser.add_argument("--val-stop", action="store_true",
                        help="choose the number of epochs on validation speakers (speaker split only)")
    args = parser.parse_args()
    if args.val_stop and args.split != "speaker":
        parser.error("--val-stop needs --split speaker")

    X, y, speaker = data.load_features()
    train_idx, test_idx = data.train_test_indices(args.split)
    extra = {}

    epochs = args.epochs
    if args.val_stop:
        # Step 1-2: pick the epoch count on validation speakers (the test set is not used).
        is_val = np.isin(speaker[train_idx], config.VAL_SPEAKERS)
        fit_idx, val_idx = train_idx[~is_val], train_idx[is_val]
        norm18 = data.normalizer(fit_idx)
        _, val_hist = train(args.seed, norm18(X[fit_idx]), y[fit_idx], args.epochs,
                            norm18(X[val_idx]), y[val_idx], "validation")
        epochs = int(np.argmax([h["macro_f1"] for h in val_hist])) + 1  # first best epoch
        print(f"Best validation macro-F1 at epoch {epochs}; retraining on all 20 speakers.")
        extra = {"val_speakers": config.VAL_SPEAKERS, "selected_epoch": epochs, "val_history": val_hist}

    # Train on the full training pool and score the test set.
    norm = data.normalizer(train_idx)
    model, history = train(args.seed, norm(X[train_idx]), y[train_idx], epochs,
                           norm(X[test_idx]), y[test_idx], "test")
    pred = np.argmax(model.predict(norm(X[test_idx]), verbose=0), axis=1)

    setting = "centralized_valstop" if args.val_stop else f"centralized_{args.split}"
    out = {"setting": setting, "seed": args.seed, "epochs": epochs, **extra,
           "final": history[-1], "history": history,
           "confusion_matrix": confusion_matrix(y[test_idx], pred).tolist(), "classes": config.CLASS_NAMES}
    config.RESULTS_DIR.mkdir(exist_ok=True)
    path = config.RESULTS_DIR / f"{setting}_seed{args.seed}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\nFinal: acc={out['final']['accuracy']:.4f} macro-F1={out['final']['macro_f1']:.4f} -> {path}")


if __name__ == "__main__":
    main()
