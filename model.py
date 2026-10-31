"""The 1D CNN from the original project, unchanged. Used by every experiment."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tensorflow import keras
from tensorflow.keras import layers

import config


def build_model():
    model = keras.Sequential([
        keras.Input(shape=(config.MAX_FRAMES, config.N_MFCC)),
        layers.Conv1D(64, 3, activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(config.NUM_CLASSES, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(config.LEARNING_RATE),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def evaluate(model, X, y):
    """Accuracy and macro-F1 (macro = every emotion counts equally, despite imbalance)."""
    pred = np.argmax(model.predict(X, batch_size=256, verbose=0), axis=1)
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro"))}
