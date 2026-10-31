"""Loading features, train/test splits, and splitting training data across FL clients."""
from functools import lru_cache

import numpy as np
from sklearn.model_selection import train_test_split

import config


@lru_cache(maxsize=1)
def load_features():
    d = np.load(config.FEATURES_FILE)
    return d["X"], d["y"], d["speaker"]


def train_test_indices(split="speaker"):
    """Indices of the training pool and the shared test set.

    speaker: test = recordings of the 4 held-out speakers (nobody trains on their voices).
    random:  stratified 80/20 split of all recordings, like the original project
             (speakers appear in both train and test). Only used for one baseline.
    """
    _, y, speaker = load_features()
    idx = np.arange(len(y))
    if split == "speaker":
        is_test = np.isin(speaker, config.TEST_SPEAKERS)
        return idx[~is_test], idx[is_test]
    if split == "random":
        return train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
    raise ValueError(split)


def normalizer(train_idx):
    """Per-MFCC-coefficient mean/std from the training pool, as in the original project.

    In real FL the server cannot see the data, but these statistics could be computed
    exactly from per-client sums and counts, so we treat them as known to everyone.
    """
    X, _, _ = load_features()
    mean = X[train_idx].mean(axis=(0, 1), keepdims=True)
    std = X[train_idx].std(axis=(0, 1), keepdims=True) + 1e-6
    return lambda x: (x - mean) / std


def partition(train_idx, scheme, num_clients=config.NUM_CLIENTS, seed=0, alpha=0.5):
    """Split the training pool into one index array per client.

    iid:     shuffle all training recordings and deal them out evenly, so every
             client sees a similar mix of speakers and emotions.
    speaker: each client owns all recordings of its own 2 speakers and nothing else,
             like phones that each keep their owner's voice data locally.
             (Speakers are paired in id order, so the split is fixed across seeds.)
    dirichlet: label skew. For each emotion, client shares are drawn from
             Dirichlet(alpha, ..., alpha); small alpha -> each client is dominated by
             a few emotions (speakers are mixed). Redrawn until every client has
             at least 10 recordings. Client sizes end up unequal.
    """
    _, y, speaker = load_features()
    if scheme == "dirichlet":
        rng = np.random.default_rng(seed)
        while True:
            parts = [[] for _ in range(num_clients)]
            for c in range(config.NUM_CLASSES):
                idx_c = rng.permutation(train_idx[y[train_idx] == c])
                shares = rng.dirichlet([alpha] * num_clients)
                cuts = (np.cumsum(shares)[:-1] * len(idx_c)).astype(int)
                for cid, chunk in enumerate(np.split(idx_c, cuts)):
                    parts[cid].extend(chunk.tolist())
            if min(len(p) for p in parts) >= 10:
                return [np.sort(np.array(p)) for p in parts]
    if scheme == "iid":
        rng = np.random.default_rng(seed)
        return np.array_split(rng.permutation(train_idx), num_clients)
    if scheme == "speaker":
        speakers = np.sort(np.unique(speaker[train_idx]))
        groups = np.array_split(speakers, num_clients)
        return [train_idx[np.isin(speaker[train_idx], g)] for g in groups]
    raise ValueError(scheme)


def describe_partition(parts):
    """Print what each client holds; useful to see the heterogeneity."""
    _, y, speaker = load_features()
    for cid, p in enumerate(parts):
        counts = np.bincount(y[p], minlength=config.NUM_CLASSES)
        spk = sorted(set(speaker[p].tolist()))
        spk_text = spk if len(spk) <= 4 else f"{len(spk)} different"
        print(f"client {cid}: n={len(p):3d}  speakers={spk_text}  class counts={counts.tolist()}")
