"""Split federated learning (SFL-V2 style), simulated in plain TensorFlow.

    python split_learning.py --partition speaker --seed 0
    python split_learning.py --partition dirichlet --alpha 0.5 --seed 0

The same 1D CNN is cut after its first block:
    client part: Conv1D(64) -> BatchNorm -> MaxPool        (stays on the device)
    server part: Conv1D(128) -> BatchNorm -> MaxPool -> Dense -> Dropout -> Dense

One round:
  1. Every client starts from the current global client part.
  2. Clients take turns (random order). For each mini-batch the client runs its part
     and sends the cut-layer activations ("smashed data") + labels to the server.
     The server finishes the forward pass, computes the loss, updates ITS part,
     and sends back d(loss)/d(activations). The client uses that gradient to
     update its own part. Raw features never leave the client.
  3. The server keeps ONE server part, updated by every client in turn.
  4. At the end of the round, the client parts are averaged, weighted by number
     of recordings (FedAvg on the client side only).
Flower has no built-in split learning, so this script does not use it; the
data, splits, model, test set and budget are the same as in federated.py.
"""
import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow import keras

import config
import data
from model import build_model, evaluate

CUT = 3  # layers [0, 3) run on the client


def weighted_average(weight_lists, sizes):
    total = sum(sizes)
    return [sum(w[i] * (n / total) for w, n in zip(weight_lists, sizes)) for i in range(len(weight_lists[0]))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=["iid", "speaker", "dirichlet"], required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=config.NUM_ROUNDS)
    parser.add_argument("--local-epochs", type=int, default=config.LOCAL_EPOCHS)
    args = parser.parse_args()

    keras.utils.set_random_seed(args.seed)
    full = build_model()  # same initial weights as the other experiments with this seed
    client_part = keras.Sequential(full.layers[:CUT])  # shares layer objects with `full`
    server_part = keras.Sequential(full.layers[CUT:])
    server_opt = keras.optimizers.Adam(config.LEARNING_RATE)  # one server, one optimizer
    loss_fn = keras.losses.SparseCategoricalCrossentropy()

    X, y, _ = data.load_features()
    train_idx, test_idx = data.train_test_indices("speaker")
    norm = data.normalizer(train_idx)
    X_test, y_test = norm(X[test_idx]), y[test_idx]
    parts = data.partition(train_idx, args.partition, seed=args.seed, alpha=args.alpha)
    rng = np.random.default_rng(args.seed)

    global_client_weights = client_part.get_weights()
    floats_sent = 0  # activations up + gradients down, counted exactly
    history = [{"round": 0, "train_loss": None, **evaluate(full, X_test, y_test)}]
    print(f"[round  0] test acc={history[0]['accuracy']:.4f} macro-F1={history[0]['macro_f1']:.4f}")

    for rnd in range(1, args.rounds + 1):
        new_client_weights, sizes, losses = [], [], []
        for cid in rng.permutation(len(parts)):  # clients take turns
            client_part.set_weights(global_client_weights)
            client_opt = keras.optimizers.Adam(config.LEARNING_RATE)  # fresh each round, as in FedAvg
            Xc, yc = norm(X[parts[cid]]), y[parts[cid]]
            for _ in range(args.local_epochs):
                order = rng.permutation(len(Xc))
                for batch in np.array_split(order, int(np.ceil(len(Xc) / config.BATCH_SIZE))):
                    # --- client: forward through its part ---
                    with tf.GradientTape() as client_tape:
                        smashed = client_part(Xc[batch], training=True)
                    # --- server: receives `smashed` + labels, finishes forward/backward ---
                    with tf.GradientTape() as server_tape:
                        server_tape.watch(smashed)
                        loss = loss_fn(yc[batch], server_part(smashed, training=True))
                    s_grads, grad_smashed = server_tape.gradient(
                        loss, [server_part.trainable_weights, smashed])
                    server_opt.apply_gradients(zip(s_grads, server_part.trainable_weights))
                    # --- client: receives grad_smashed, finishes backprop on its part ---
                    c_grads = client_tape.gradient(smashed, client_part.trainable_weights,
                                                   output_gradients=grad_smashed)
                    client_opt.apply_gradients(zip(c_grads, client_part.trainable_weights))
                    floats_sent += 2 * int(np.prod(smashed.shape))
                    losses.append(float(loss))
            new_client_weights.append(client_part.get_weights())
            sizes.append(len(Xc))
        # FedAvg over the client parts only (the server part is already shared).
        global_client_weights = weighted_average(new_client_weights, sizes)
        client_part.set_weights(global_client_weights)
        m = evaluate(full, X_test, y_test)
        history.append({"round": rnd, "train_loss": float(np.mean(losses)), **m})
        print(f"[round {rnd:2d}] test acc={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f}")

    n_client = sum(int(np.prod(w.shape)) for w in client_part.weights)
    n_total = sum(int(np.prod(w.shape)) for w in full.weights)
    pname = f"dirichlet{args.alpha}" if args.partition == "dirichlet" else args.partition
    setting = f"split_{pname}"
    out = {"setting": setting, "seed": args.seed, "rounds": args.rounds,
           "local_epochs": args.local_epochs, "num_clients": len(parts),
           "client_params": n_client, "total_params": n_total,
           "activation_floats_per_round": floats_sent // args.rounds,
           "final": history[-1], "history": history}
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / f"{setting}_seed{args.seed}.json").write_text(json.dumps(out, indent=1))
    print(f"\nclient part: {n_client:,} of {n_total:,} parameters; "
          f"activations+gradients sent per round: {floats_sent // args.rounds:,} floats")
    print(f"Final: acc={out['final']['accuracy']:.4f} macro-F1={out['final']['macro_f1']:.4f}"
          f" -> results/{setting}_seed{args.seed}.json")


if __name__ == "__main__":
    main()
