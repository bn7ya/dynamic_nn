from __future__ import annotations

import gzip
from urllib.request import urlopen

import torch

from elasticneuralnetwork.benchmarks._common import (
    BenchmarkRow,
    CACHE_DIR,
    make_argparser,
    run_enn,
    set_seed,
    write_curves_png,
    write_rows_csv,
)
from elasticneuralnetwork.benchmarks._tabular_common import (
    load_synthetic_classification,
    split_train_test,
    standardize,
)
from elasticneuralnetwork.benchmarks.baselines import (
    LightGBMBaseline,
    PyTorchMLPBaseline,
)


DATASET = "tabular_higgs"
URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00280/HIGGS.csv.gz")


def load_higgs(seed: int, n_rows: int = 100_000):
    import numpy as np
    import pandas as pd
    dest = CACHE_DIR / "HIGGS.csv.gz"
    try:
        if not dest.exists():
            with urlopen(URL, timeout=180) as resp:
                dest.write_bytes(resp.read())
        with gzip.open(dest, "rt") as f:
            df = pd.read_csv(f, header=None, nrows=n_rows)
        X = df.iloc[:, 1:].to_numpy(dtype=np.float32)
        y = df.iloc[:, 0].to_numpy(dtype=np.int64)
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)
        num_classes = 2
    except Exception as exc:
        print(f"[fallback] higgs download failed ({exc}); "
                f"using synthetic data")
        return load_synthetic_classification(DATASET, seed,
                                              n_classes=2)
    Xtr, ytr, Xte, yte = split_train_test(X_t, y_t, 0.8, seed)
    Xtr, Xte = standardize(Xtr, Xte)
    return Xtr, ytr, Xte, yte, num_classes


def main():
    args = make_argparser().parse_args()
    set_seed(args.seed)
    Xtr, ytr, Xte, yte, num_classes = load_higgs(args.seed)
    if args.limit_samples > 0:
        Xtr = Xtr[: args.limit_samples]
        ytr = ytr[: args.limit_samples]
    ytr_oh = torch.eye(num_classes)[ytr]
    yte_oh = torch.eye(num_classes)[yte]

    rows: list[BenchmarkRow] = []
    enn = run_enn(Xtr, ytr_oh, Xte, yte_oh, args.seed,
                    input_shape=(Xtr.shape[1],),
                    output_size=num_classes, device=args.device)
    rows.append(BenchmarkRow(
        dataset=DATASET, method="elasticneuralnetwork", seed=args.seed,
        device=args.device, accuracy=enn["accuracy"],
        parameter_count=enn["parameter_count"],
        train_seconds=enn["train_seconds"],
        topology_version_final=enn["topology_version_final"],
        epochs_completed=enn["epochs_completed"]))

    epochs = args.epochs_cap if args.epochs_cap > 0 else 40
    mlp = PyTorchMLPBaseline.train(Xtr, ytr_oh, Xte, yte_oh,
                                    target_param_count=enn[
                                        "parameter_count"],
                                    epochs=epochs, device=args.device)
    rows.append(BenchmarkRow(
        dataset=DATASET, method="pytorch_mlp", seed=args.seed,
        device=args.device, accuracy=mlp.final_metric,
        parameter_count=mlp.parameter_count,
        train_seconds=mlp.train_seconds))
    try:
        gbm = LightGBMBaseline.train(Xtr, ytr_oh, Xte, yte_oh,
                                       n_estimators=100)
        rows.append(BenchmarkRow(
            dataset=DATASET, method="lightgbm", seed=args.seed,
            device="cpu", accuracy=gbm.final_metric,
            parameter_count=gbm.parameter_count,
            train_seconds=gbm.train_seconds))
    except Exception as exc:
        print(f"[skip] lightgbm baseline: {exc}")

    csv_path = write_rows_csv(DATASET, args.seed, rows)
    png_path = write_curves_png(DATASET, args.seed,
                                  enn["cost_trajectory"],
                                  enn["utilization_trajectory"])
    print(f"wrote {csv_path} and {png_path}")
    for r in rows:
        print(f"  {r.method}: acc={r.accuracy:.4f} "
                f"params={r.parameter_count} time={r.train_seconds:.1f}s")


if __name__ == "__main__":
    main()
