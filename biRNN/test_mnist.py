import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from live_plot import LiveTrainingPlot
from model import BidirectionalRNN, get_device
from run_artifacts import make_run_dir, save_activity_snapshot, setup_logger

logger = logging.getLogger(__name__)


def load_data(batch_size: int = 128):
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def to_sequence(images: torch.Tensor) -> torch.Tensor:
    return images.squeeze(1)


def save_results(model, history, results_path: Path, model_path: Path) -> None:
    with open(results_path, "w") as f:
        json.dump(history, f, indent=2)
    torch.save(model.state_dict(), model_path)
    logger.info(f"saved {len(history)} epoch(s) of history to {results_path}, model weights to {model_path}")


def train(
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


def main():
    device = get_device()

    run_dir = make_run_dir("mnist")
    setup_logger(__name__, run_dir / "train.log")
    logger.info(f"using device: {device}")

    hidden_size = 64
    logger.info(
        f"model: BidirectionalRNN(input_size=28, hidden_size={hidden_size}, output_size=10, output_mode='last')"
    )

    train_loader, test_loader = load_data()
    model = BidirectionalRNN(input_size=28, hidden_size=hidden_size, output_size=10, output_mode="last").to(device)

    live_plot = LiveTrainingPlot(title="biRNN/test_mnist.py")
    accuracy = train(
        model,
        train_loader,
        test_loader,
        device,
        epochs=5,
        run_dir=run_dir,
        live_plot=live_plot,
        module_bounds=[hidden_size],
    )
    logger.info(f"test accuracy: {accuracy:.4f}")
    assert accuracy > 0.90, f"expected >90% accuracy, got {accuracy:.4f}"


if __name__ == "__main__":
    main()
