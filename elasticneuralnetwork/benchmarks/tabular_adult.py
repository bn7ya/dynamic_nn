from __future__ import annotations

import io
from pathlib import Path
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
    _categorize_columns,
    _to_tensors,
    load_synthetic_classification,
    split_train_test,
    standardize,
)
from elasticneuralnetwork.benchmarks.baselines import (
    LightGBMBaseline,
    PyTorchMLPBaseline,
)


DATASET = "tabular_adult"
URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "adult/adult.data")
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]


def load_adult(seed: int):
    import pandas as pd
    dest = CACHE_DIR / "adult.data"
    try:
        if not dest.exists():
            with urlopen(URL, timeout=30) as resp:
                dest.write_bytes(resp.read())
        df = pd.read_csv(dest, names=COLUMNS, skipinitialspace=True,
                          na_values="?")
        df = df.dropna()
        df, cols = _categorize_columns(df, "income")
        X, y, num_classes = _to_tensors(df, cols, "income")
    except Exception as exc:
        print(f"[fallback] adult download failed ({exc}); using "
                f"synthetic data")
        Xtr, ytr, Xte, yte, num_classes = load_synthetic_classification(
            DATASET, seed)
        return Xtr, ytr, Xte, yte, num_classes
    Xtr, ytr, Xte, yte = split_train_test(X, y, 0.8, seed)
    Xtr, Xte = standardize(Xtr, Xte)
    return Xtr, ytr, Xte, yte, num_classes


def main():
    args = make_argparser().parse_args()
    set_seed(args.seed)
    Xtr, ytr, Xte, yte, num_classes = load_adult(args.seed)
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
