| Method | Data split | Accuracy, final (mean ± sd) | Macro-F1, final (mean ± sd) | Macro-F1, avg of last 5 rounds (mean ± sd) | Macro-F1 per seed, final |
|---|---|---|---|---|---|
| Centralized, plain 60 epochs | Random 80/20 split ⚠ different test set: speakers appear in train and test | 66.7% ± 1.5 | 0.660 ± 0.012 | 0.636 ± 0.014 | 0.650, 0.673, 0.655 |
| Centralized, plain 60 epochs | All 20 training speakers on one machine | 53.0% ± 3.1 | 0.504 ± 0.030 | 0.508 ± 0.014 | 0.487, 0.486, 0.538 |
| Centralized, epochs chosen on 2 validation speakers | All 20 training speakers on one machine | 51.9% ± 3.4 | 0.503 ± 0.017 | 0.480 ± 0.048 | 0.517, 0.484, 0.507 |
| FedAvg | IID | 55.9% ± 1.1 | 0.551 ± 0.015 | 0.559 ± 0.019 | 0.538, 0.548, 0.567 |
| FedAvg | 2 speakers per client | 54.5% ± 1.5 | 0.538 ± 0.015 | 0.542 ± 0.030 | 0.551, 0.521, 0.541 |
| FedAvg | Label skew (Dir. α=0.5) | 52.7% ± 3.3 | 0.512 ± 0.035 | 0.504 ± 0.044 | 0.519, 0.474, 0.543 |
| FedProx μ=0 | Label skew (Dir. α=0.5) | 50.8% ± 1.4 | 0.496 ± 0.016 | 0.496 ± 0.009 | 0.485, 0.488, 0.514 |
| FedProx μ=0.001 | Label skew (Dir. α=0.5) | 50.8% ± 1.8 | 0.499 ± 0.024 | 0.501 ± 0.023 | 0.485, 0.485, 0.527 |
| FedProx μ=0.01 | Label skew (Dir. α=0.5) | 51.6% ± 2.2 | 0.499 ± 0.037 | 0.495 ± 0.037 | 0.507, 0.458, 0.531 |
| FedProx μ=0.1 | Label skew (Dir. α=0.5) | 49.0% ± 3.8 | 0.477 ± 0.036 | 0.500 ± 0.019 | 0.514, 0.442, 0.476 |
| Split learning (SFL-V2) | 2 speakers per client | 48.6% ± 3.5 | 0.471 ± 0.045 | 0.476 ± 0.009 | 0.471, 0.427, 0.516 |
| Split learning (SFL-V2) | Label skew (Dir. α=0.5) | 45.0% ± 4.7 | 0.418 ± 0.053 | 0.461 ± 0.020 | 0.477, 0.406, 0.373 |

Seeds 0, 1, 2 for every row. Test set: the 208 recordings of held-out speakers [6, 8, 15, 19], except the random-split row. For centralized rows, 'rounds' means epochs (plain: final = epoch 60, last 5 = epochs 56-60; early-stopped: final = the chosen epoch, last 5 = the 5 epochs up to it).
