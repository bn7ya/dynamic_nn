from __future__ import annotations

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
from elasticneuralnetwork.benchmarks.baselines import (
    PyTorchCNNBaseline,
    PyTorchMLPBaseline,
)


DATASET = "cifar10"


def load_cifar10():
    from torchvision import datasets, transforms
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.CIFAR10(str(CACHE_DIR), train=True, download=True,
                              transform=tfm)
    test = datasets.CIFAR10(str(CACHE_DIR), train=False, download=True,
                              transform=tfm)
    Xtr = torch.stack([train[i][0] for i in range(len(train))]).float()
    ytr = torch.tensor([train[i][1] for i in range(len(train))],
                        dtype=torch.long)
    Xte = torch.stack([test[i][0] for i in range(len(test))]).float()
    yte = torch.tensor([test[i][1] for i in range(len(test))],
                        dtype=torch.long)
    return Xtr, ytr, Xte, yte


def main():
    args = make_argparser().parse_args()
    set_seed(args.seed)
    Xtr, ytr, Xte, yte = load_cifar10()
    if args.limit_samples > 0:
        Xtr = Xtr[: args.limit_samples]
        ytr = ytr[: args.limit_samples]
    num_classes = int(ytr.max().item()) + 1
    ytr_onehot = torch.eye(num_classes)[ytr]
    yte_onehot = torch.eye(num_classes)[yte]

    Xtr_flat = Xtr.reshape(Xtr.shape[0], -1)
    Xte_flat = Xte.reshape(Xte.shape[0], -1)
    rows: list[BenchmarkRow] = []

    enn = run_enn(Xtr_flat, ytr_onehot, Xte_flat, yte_onehot,
                    args.seed, input_shape=(Xtr_flat.shape[1],),
                    output_size=num_classes, device=args.device)
    rows.append(BenchmarkRow(
        dataset=DATASET, method="elasticneuralnetwork", seed=args.seed,
        device=args.device, accuracy=enn["accuracy"],
        parameter_count=enn["parameter_count"],
        train_seconds=enn["train_seconds"],
        topology_version_final=enn["topology_version_final"],
        epochs_completed=enn["epochs_completed"]))

    epochs = args.epochs_cap if args.epochs_cap > 0 else 25
    cnn = PyTorchCNNBaseline.train(Xtr, ytr_onehot, Xte, yte_onehot,
                                    target_param_count=enn[
                                        "parameter_count"],
                                    epochs=epochs, device=args.device)
    rows.append(BenchmarkRow(
        dataset=DATASET, method="pytorch_cnn", seed=args.seed,
        device=args.device, accuracy=cnn.final_metric,
        parameter_count=cnn.parameter_count,
        train_seconds=cnn.train_seconds))

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
