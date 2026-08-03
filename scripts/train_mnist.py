import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.bidirectional_rnn import BidirectionalRNN
from models.common import get_device
from models.modular_rnn import ModularRNN
from models.simple_rnn import SimpleRNN
from scripts.hf_optimizer import HFOptimizer
from scripts.live_plot import LiveTrainingPlot
from scripts.run_artifacts import make_run_dir, save_activity_snapshot, setup_logger

logger = logging.getLogger(__name__)

# hidden_size must be divisible by 3 for modular_rnn (input/intermediate/output modules)
MODEL_DEFAULTS = {
    "simple_rnn": {"cls": SimpleRNN, "hidden_size": 64},
    "bidirectional_rnn": {"cls": BidirectionalRNN, "hidden_size": 64},
    "modular_rnn": {"cls": ModularRNN, "hidden_size": 63},
}


def build_model(model_name: str, hidden_size: int):
    cls = MODEL_DEFAULTS[model_name]["cls"]
    return cls(input_size=28, hidden_size=hidden_size, output_size=10, output_mode="last")


def module_bounds_for(model_name: str, hidden_size: int) -> list[int]:
    if model_name == "modular_rnn":
        third = hidden_size // 3
        return [third, 2 * third]
    return [hidden_size]


def load_data(data_root: str, batch_size: int = 128):
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def to_sequence(images: torch.Tensor) -> torch.Tensor:
    # images: (batch, 1, 28, 28) -> (batch, 28, 28), each image read as 28 rows of 28 pixels
    return images.squeeze(1)


def save_results(model, history, results_path: Path, model_path: Path) -> None:
    with open(results_path, "w") as f:
        json.dump(history, f, indent=2)
    torch.save(model.state_dict(), model_path)
    logger.info(f"saved {len(history)} epoch(s) of history to {results_path}, model weights to {model_path}")


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = to_sequence(images).to(device)
        labels = labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def train_adam(
    model,
    train_loader,
    test_loader,
    device,
    epochs: int,
    run_dir: Path,
    lr: float = 1e-3,
    live_plot=None,
    module_bounds: list[int] | None = None,
) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    accuracy = 0.0
    history = []
    activity_dir = run_dir / "activity"
    sample_images, _ = next(iter(test_loader))
    activity_sample = to_sequence(sample_images[:1]).to(device)
    try:
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for images, labels in train_loader:
                images = to_sequence(images).to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            accuracy = evaluate(model, test_loader, device)
            logger.info(f"epoch {epoch + 1}/{epochs} loss {avg_loss:.4f} accuracy {accuracy:.4f}")
            history.append({"epoch": epoch + 1, "loss": avg_loss, "accuracy": accuracy})
            if live_plot is not None:
                live_plot.update(epoch + 1, avg_loss, accuracy)

            model.eval()
            with torch.no_grad():
                _, hidden = model(activity_sample, return_hidden=True)
            save_activity_snapshot(hidden[0], activity_dir, f"epoch_{epoch + 1:02d}", module_bounds)
    finally:
        save_results(model, history, run_dir / "results.json", run_dir / "model.pt")
        if live_plot is not None:
            live_plot.save(run_dir / "curve.png")
    return accuracy


def train_hf(
    model,
    train_loader,
    test_loader,
    device,
    epochs: int,
    run_dir: Path,
    live_plot=None,
    module_bounds: list[int] | None = None,
) -> float:
    optimizer = HFOptimizer(model, curvature="categorical")
    criterion = nn.CrossEntropyLoss()
    accuracy = 0.0
    history = []
    activity_dir = run_dir / "activity"
    sample_images, _ = next(iter(test_loader))
    activity_sample = to_sequence(sample_images[:1]).to(device)
    try:
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for images, labels in train_loader:
                images = to_sequence(images).to(device)
                labels = labels.to(device)

                def objective_fn(m, images=images, labels=labels):
                    z = m(images)
                    return criterion(z, labels), z

                diagnostics = optimizer.step(objective_fn)
                total_loss += diagnostics["loss_after"]
            avg_loss = total_loss / len(train_loader)
            accuracy = evaluate(model, test_loader, device)
            logger.info(
                f"epoch {epoch + 1}/{epochs} loss {avg_loss:.4f} accuracy {accuracy:.4f} damping {optimizer.damping:.4g}"
            )
            history.append({"epoch": epoch + 1, "loss": avg_loss, "accuracy": accuracy})
            if live_plot is not None:
                live_plot.update(epoch + 1, avg_loss, accuracy)

            model.eval()
            with torch.no_grad():
                _, hidden = model(activity_sample, return_hidden=True)
            save_activity_snapshot(hidden[0], activity_dir, f"epoch_{epoch + 1:02d}", module_bounds)
    finally:
        save_results(model, history, run_dir / "results.json", run_dir / "model.pt")
        if live_plot is not None:
            live_plot.save(run_dir / "curve.png")
    return accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Train an RNN variant on MNIST (row-by-row sequence).")
    parser.add_argument("--model", choices=sorted(MODEL_DEFAULTS), default="simple_rnn")
    parser.add_argument("--optimizer", choices=("adam", "hf"), default="adam")
    parser.add_argument("--hidden-size", type=int, default=None, help="defaults to the model's usual size")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3, help="ignored when --optimizer=hf")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--min-accuracy", type=float, default=0.90)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    hidden_size = args.hidden_size or MODEL_DEFAULTS[args.model]["hidden_size"]
    if args.model == "modular_rnn" and hidden_size % 3 != 0:
        raise ValueError(f"hidden_size must be divisible by 3 for modular_rnn, got {hidden_size}")

    run_label = args.model if args.optimizer == "adam" else f"{args.model}_hf"
    run_dir = make_run_dir("mnist", root=str(Path(args.results_root) / run_label))
    setup_logger(__name__, run_dir / "train.log")
    logger.info(f"using device: {device}")
    logger.info(
        f"model: {MODEL_DEFAULTS[args.model]['cls'].__name__}"
        f"(input_size=28, hidden_size={hidden_size}, output_size=10, output_mode='last'), optimizer={args.optimizer}"
    )

    train_loader, test_loader = load_data(args.data_root, batch_size=args.batch_size)
    model = build_model(args.model, hidden_size).to(device)
    module_bounds = module_bounds_for(args.model, hidden_size)

    live_plot = LiveTrainingPlot(title=f"scripts/train_mnist.py --model {args.model} --optimizer {args.optimizer}")
    train_fn = train_adam if args.optimizer == "adam" else train_hf
    train_kwargs = {"lr": args.lr} if args.optimizer == "adam" else {}
    accuracy = train_fn(
        model,
        train_loader,
        test_loader,
        device,
        epochs=args.epochs,
        run_dir=run_dir,
        live_plot=live_plot,
        module_bounds=module_bounds,
        **train_kwargs,
    )
    logger.info(f"test accuracy: {accuracy:.4f}")
    assert accuracy > args.min_accuracy, f"expected >{args.min_accuracy:.0%} accuracy, got {accuracy:.4f}"


if __name__ == "__main__":
    main()
