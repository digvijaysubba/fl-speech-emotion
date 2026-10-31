"""Federated training with Flower (simulation mode, Flower 1.42 Message API).

    python federated.py --partition iid     --seed 0
    python federated.py --partition speaker --seed 0
    python federated.py --partition dirichlet --alpha 0.5 --seed 0
    python federated.py --partition dirichlet --alpha 0.5 --seed 0 --mu 0.01   # FedProx

How one experiment runs:
  1. The server builds the 1D CNN and keeps its weights as the "global model".
  2. Every round, the server sends the global weights to all 10 clients.
  3. Each client copies those weights into its own model, trains for LOCAL_EPOCHS
     on ITS OWN recordings only, and sends back the new weights + how many
     recordings it trained on. Raw audio/features never leave the client.
  4. The server aggregates: new global = weighted average of client weights,
     weighted by number of recordings (this is FedAvg).
  5. The server evaluates the new global model on the shared held-out-speaker
     test set (the same 208 recordings as the centralized baseline).

FedProx (--mu M; --mu 0 is a sanity check that should behave like FedAvg) changes only step 3:
each client minimizes  local_loss(w) + mu/2 * ||w - w_global||^2.  The extra term
pulls the client's weights back toward the global model, limiting "client drift"
when clients' data differ. The server-side averaging is the same as FedAvg.
"""
import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx
from flwr.simulation import run_simulation
from tensorflow import keras

import config
import data
from model import build_model, evaluate

# ---------------------------------------------------------------------------
# Client side
# ---------------------------------------------------------------------------
client_app = ClientApp()


def train_fedprox(model, X, y, epochs, mu, seed):
    """Local training with the FedProx proximal term (plain mini-batch loop, Adam).

    With mu = 0 this is ordinary training, just written out by hand.
    """
    global_weights = [tf.constant(w.numpy()) for w in model.trainable_weights]  # w_global, frozen
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        losses = []
        for batch in np.array_split(rng.permutation(len(X)), int(np.ceil(len(X) / config.BATCH_SIZE))):
            with tf.GradientTape() as tape:
                loss = loss_fn(y[batch], model(X[batch], training=True))
                prox = tf.add_n([tf.reduce_sum(tf.square(w - g))
                                 for w, g in zip(model.trainable_weights, global_weights)])
                total = loss + (mu / 2) * prox
            grads = tape.gradient(total, model.trainable_weights)
            model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
            losses.append(float(loss))
    return float(np.mean(losses))  # task loss (without the prox term) in the last epoch


@client_app.train()
def client_train(msg: Message, context: Context) -> Message:
    """One client's work in one round: local training on local data only."""
    cfg = msg.content["config"]  # settings the server sent this round
    cid = int(context.node_config["partition-id"])  # which simulated client am I (0..9)?

    # Each client gets its own seed per round, so runs are reproducible.
    keras.backend.clear_session()
    client_seed = int(cfg["seed"]) * 10_000 + int(cfg["server-round"]) * 100 + cid
    keras.utils.set_random_seed(client_seed)

    # This client's slice of the training pool (decided by the partition scheme).
    X, y, _ = data.load_features()
    train_idx, _ = data.train_test_indices("speaker")
    my_idx = data.partition(train_idx, cfg["partition"], seed=int(cfg["seed"]),
                            alpha=float(cfg["alpha"]))[cid]
    norm = data.normalizer(train_idx)

    # Start from the current global model and train locally.
    model = build_model()  # fresh Adam optimizer each round
    model.set_weights(msg.content["arrays"].to_numpy_ndarrays())
    if "proximal-mu" in cfg:  # the FedProx strategy puts mu into the config
        train_loss = train_fedprox(model, norm(X[my_idx]), y[my_idx], int(cfg["local-epochs"]),
                                   float(cfg["proximal-mu"]), client_seed)
    else:
        fit = model.fit(norm(X[my_idx]), y[my_idx], epochs=int(cfg["local-epochs"]),
                        batch_size=config.BATCH_SIZE, verbose=0)
        train_loss = float(fit.history["loss"][-1])

    # Reply with updated weights and the number of examples (FedAvg uses it as the weight).
    content = RecordDict({
        "arrays": ArrayRecord(model.get_weights()),
        "metrics": MetricRecord({"train_loss": train_loss,
                                 "num-examples": len(my_idx)}),
        "client-info": ConfigRecord({"partition-id": cid}),  # lets the server sort replies
    })
    return Message(content=content, reply_to=msg)


# ---------------------------------------------------------------------------
# Server side
# ---------------------------------------------------------------------------
class SortedReplies:
    """Average client replies in client-id order.

    Clients finish in a random order, and floating-point sums depend on order,
    so without this two runs with the same seed drift apart over the rounds.
    The average itself is mathematically unchanged.
    """

    def aggregate_train(self, server_round, replies):
        replies = sorted(replies, key=lambda m: int(m.content["client-info"]["partition-id"]))
        return super().aggregate_train(server_round, replies)


class OrderedFedAvg(SortedReplies, FedAvg):
    pass


class OrderedFedProx(SortedReplies, FedProx):
    """Flower's FedProx only sends mu to the clients; the clients apply it (train_fedprox)."""


def make_server_app(args) -> ServerApp:
    server_app = ServerApp()

    @server_app.main()
    def main(grid: Grid, context: Context) -> None:
        keras.utils.set_random_seed(args.seed)
        global_model = build_model()  # same initial weights as the centralized run with this seed

        X, y, _ = data.load_features()
        train_idx, test_idx = data.train_test_indices("speaker")
        X_test, y_test = data.normalizer(train_idx)(X[test_idx]), y[test_idx]

        def evaluate_global(server_round: int, arrays: ArrayRecord) -> MetricRecord:
            """Called by the strategy after each round: score the aggregated model."""
            global_model.set_weights(arrays.to_numpy_ndarrays())
            m = evaluate(global_model, X_test, y_test)
            print(f"[round {server_round:2d}] global model: test acc={m['accuracy']:.4f} "
                  f"macro-F1={m['macro_f1']:.4f}")
            return MetricRecord(m)

        # FedAvg: aggregated weights = sum_k (n_k / n) * w_k over the clients k.
        # Every client trains every round; clients do not evaluate (the server does).
        common = dict(fraction_train=1.0, fraction_evaluate=0.0,
                      min_train_nodes=config.NUM_CLIENTS, min_available_nodes=config.NUM_CLIENTS)
        if args.mu is None:
            strategy = OrderedFedAvg(**common)
        else:
            strategy = OrderedFedProx(**common, proximal_mu=args.mu)
        result = strategy.start(
            grid=grid,
            initial_arrays=ArrayRecord(global_model.get_weights()),
            num_rounds=args.rounds,
            train_config=ConfigRecord({"partition": args.partition, "seed": args.seed,
                                       "alpha": args.alpha, "local-epochs": args.local_epochs}),
            evaluate_fn=evaluate_global,
        )
        save_results(args, result)

    return server_app


def partition_name(args):
    return f"dirichlet{args.alpha}" if args.partition == "dirichlet" else args.partition


def save_results(args, result):
    train_loss = {r: float(m["train_loss"]) for r, m in result.train_metrics_clientapp.items()}
    history = [{"round": int(r), "train_loss": train_loss.get(r), "accuracy": float(m["accuracy"]),
                "macro_f1": float(m["macro_f1"])}
               for r, m in sorted(result.evaluate_metrics_serverapp.items())]
    algorithm = "fedavg" if args.mu is None else f"fedprox{args.mu:g}"
    setting = f"{algorithm}_{partition_name(args)}"
    name = f"{setting}_seed{args.seed}"
    out = {"setting": setting, "seed": args.seed, "rounds": args.rounds, "mu": args.mu,
           "local_epochs": args.local_epochs, "num_clients": config.NUM_CLIENTS,
           "final": history[-1], "history": history}
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / f"{name}.json").write_text(json.dumps(out, indent=1))
    print(f"\nFinal: acc={out['final']['accuracy']:.4f} macro-F1={out['final']['macro_f1']:.4f}"
          f" -> results/{name}.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=["iid", "speaker", "dirichlet"], required=True)
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration")
    parser.add_argument("--mu", type=float, default=None,
                        help="use FedProx with this proximal weight (omit for FedAvg)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=config.NUM_ROUNDS)
    parser.add_argument("--local-epochs", type=int, default=config.LOCAL_EPOCHS)
    args = parser.parse_args()

    train_idx, _ = data.train_test_indices("speaker")
    print(f"Partition '{args.partition}' ({config.NUM_CLIENTS} clients):")
    data.describe_partition(data.partition(train_idx, args.partition, seed=args.seed, alpha=args.alpha))

    # Simulation: Flower starts 10 virtual clients on this machine (Ray backend).
    # Each one gets 2 CPU cores, so several clients train in parallel.
    run_simulation(server_app=make_server_app(args), client_app=client_app,
                   num_supernodes=config.NUM_CLIENTS,
                   backend_config={"client_resources": {"num_cpus": 2, "num_gpus": 0.0}})


if __name__ == "__main__":
    main()
