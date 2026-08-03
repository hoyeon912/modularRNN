import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from hf_optimizer import HFOptimizer
from live_plot import LiveTrainingPlot
from model import ModularRNN, get_device
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
    live_plot=None,
    module_bounds: list[int] | None = None,
) -> float:
    import sys
    optimizer = HFOptimizer(model, curvature="categorical")
    criterion = nn.CrossEntropyLoss()
    accuracy = 0.0
    history = []
    activity_dir = run_dir / "activity"
    print("DEBUG: Getting sample images from test_loader", file=sys.stderr, flush=True)
    sample_images, _ = next(iter(test_loader))
    activity_sample = to_sequence(sample_images[:1]).to(device)
    print("DEBUG: Sample images ready", file=sys.stderr, flush=True)
    try:
        for epoch in range(epochs):
            print(f"DEBUG: Starting epoch {epoch + 1}", file=sys.stderr, flush=True)
            model.train()
            total_loss = 0.0
            batch_count = 0
            for images, labels in train_loader:
                batch_count += 1
                if batch_count % 100 == 0:
                    print(f"DEBUG: Epoch {epoch + 1}, batch {batch_count}/{len(train_loader)}", file=sys.stderr, flush=True)
                images = to_sequence(images).to(device)
                labels = labels.to(device)

                def objective_fn(m, images=images, labels=labels):
                    z = m(images)
                    return criterion(z, labels), z

                diagnostics = optimizer.step(objective_fn)
                total_loss += diagnostics["loss_after"]
            print(f"DEBUG: Finished training batches for epoch {epoch + 1}", file=sys.stderr, flush=True)
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
            print(f"DEBUG: Completed epoch {epoch + 1}", file=sys.stderr, flush=True)
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
    import sys
    device = get_device()

    run_dir = make_run_dir("mnist")
    setup_logger(__name__, run_dir / "train.log")
    logger.info(f"using device: {device}")

    hidden_size = 63
    logger.info(f"model: ModularRNN(input_size=28, hidden_size={hidden_size}, output_size=10, output_mode='last')")

    print("DEBUG: About to load data", file=sys.stderr, flush=True)
    train_loader, test_loader = load_data()
    print("DEBUG: Data loaded", file=sys.stderr, flush=True)

    model = ModularRNN(input_size=28, hidden_size=hidden_size, output_size=10, output_mode="last").to(device)
    print("DEBUG: Model created", file=sys.stderr, flush=True)

    live_plot = LiveTrainingPlot(title="hfRNN/test_mnist.py")
    print("DEBUG: Live plot created", file=sys.stderr, flush=True)

    third = hidden_size // 3
    print("DEBUG: About to train", file=sys.stderr, flush=True)
    accuracy = train(
        model,
        train_loader,
        test_loader,
        device,
        epochs=5,
        run_dir=run_dir,
        live_plot=live_plot,
        module_bounds=[third, 2 * third],
    )
    print("DEBUG: Training complete", file=sys.stderr, flush=True)
    logger.info(f"test accuracy: {accuracy:.4f}")
    assert accuracy > 0.90, f"expected >90% accuracy, got {accuracy:.4f}"


if __name__ == "__main__":
    main()
