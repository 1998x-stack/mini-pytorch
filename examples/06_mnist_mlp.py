"""Train a small MLP on MNIST with mini-pytorch.

Outputs under --output-dir/<timestamp>/:
  train.log, metrics.csv, loss_curve.png, accuracy_curve.png, best_model.npz
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from mini_pytorch import nn, no_grad, optim
from mini_pytorch.data import DataLoader, MNIST


def build_model(hidden_size: int) -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 10),
    )


def accuracy(logits, targets) -> tuple[int, int]:
    predictions = np.argmax(logits.data, axis=1)
    labels = targets.data.reshape(-1).astype(np.int64)
    return int((predictions == labels).sum()), len(labels)


def train_one_epoch(model, loader, criterion, optimizer, epoch: int, log_interval: int):
    model.train()
    loss_sum = 0.0
    correct = 0
    seen = 0

    for batch_idx, (images, targets) in enumerate(loader, start=1):
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        loss_sum += loss.item() * batch_size
        batch_correct, batch_seen = accuracy(logits, targets)
        correct += batch_correct
        seen += batch_seen

        if log_interval > 0 and batch_idx % log_interval == 0:
            logger.info(
                "epoch={} batch={}/{} loss={:.4f} running_acc={:.2%}",
                epoch,
                batch_idx,
                len(loader),
                loss.item(),
                correct / seen,
            )

    return loss_sum / seen, correct / seen


def evaluate(model, loader, criterion):
    model.eval()
    loss_sum = 0.0
    correct = 0
    seen = 0

    with no_grad():
        for images, targets in loader:
            logits = model(images)
            loss = criterion(logits, targets)
            batch_size = images.shape[0]
            loss_sum += loss.item() * batch_size
            batch_correct, batch_seen = accuracy(logits, targets)
            correct += batch_correct
            seen += batch_seen

    return loss_sum / seen, correct / seen


def save_model(model, path: Path) -> None:
    np.savez(path, **model.state_dict())


def write_metrics(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_history(rows: list[dict], output_dir: Path) -> None:
    epochs = [row["epoch"] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_loss"] for row in rows], marker="o", label="train")
    plt.plot(epochs, [row["test_loss"] for row in rows], marker="o", label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("MNIST loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_acc"] for row in rows], marker="o", label="train")
    plt.plot(epochs, [row["test_acc"] for row in rows], marker="o", label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("MNIST accuracy")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=160)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train mini-pytorch MLP on MNIST")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./runs/mnist"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")

    np.random.seed(args.seed)
    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(lambda message: print(message, end=""), level="INFO")
    logger.add(run_dir / "train.log", rotation="10 MB", retention=5, level="DEBUG")

    config = vars(args).copy()
    config["data_dir"] = str(config["data_dir"])
    config["output_dir"] = str(config["output_dir"])
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    logger.info("run_dir={}", run_dir)
    logger.info("loading MNIST from {}", args.data_dir)
    train_dataset = MNIST(
        args.data_dir,
        train=True,
        download=args.download,
        normalize=args.normalize,
        limit=args.train_limit,
    )
    test_dataset = MNIST(
        args.data_dir,
        train=False,
        download=args.download,
        normalize=args.normalize,
        limit=args.test_limit,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(args.hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.lr * 0.05,
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    logger.info("model={} parameters={:,}", model, parameter_count)
    logger.info("train_samples={} test_samples={}", len(train_dataset), len(test_dataset))

    history: list[dict] = []
    best_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch, args.log_interval
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        current_lr = optimizer.learning_rates[0]

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        history.append(row)
        write_metrics(run_dir / "metrics.csv", history)
        plot_history(history, run_dir)

        logger.info(
            "epoch={}/{} lr={:.6g} train_loss={:.4f} train_acc={:.2%} "
            "test_loss={:.4f} test_acc={:.2%}",
            epoch,
            args.epochs,
            current_lr,
            train_loss,
            train_acc,
            test_loss,
            test_acc,
        )

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            save_model(model, run_dir / "best_model.npz")
            logger.info("saved new best model: test_acc={:.2%}", best_accuracy)

        scheduler.step()

    save_model(model, run_dir / "last_model.npz")
    logger.info("training complete best_test_acc={:.2%}", best_accuracy)
    logger.info("loss plot={}", run_dir / "loss_curve.png")
    logger.info("metrics={}", run_dir / "metrics.csv")


if __name__ == "__main__":
    main()
