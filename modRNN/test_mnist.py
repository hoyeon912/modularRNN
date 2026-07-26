import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from live_plot import LiveTrainingPlot
from model import ModularBidirectionalRNN, get_device


def load_data(batch_size: int = 128):
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def to_sequence(images: torch.Tensor) -> torch.Tensor:
    return images.squeeze(1)


def train(model, train_loader, test_loader, device, epochs: int, lr: float = 1e-3, live_plot=None) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    accuracy = 0.0
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
        print(f"epoch {epoch + 1}/{epochs} loss {avg_loss:.4f} accuracy {accuracy:.4f}")
        if live_plot is not None:
            live_plot.update(epoch + 1, avg_loss, accuracy)
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
    print(f"using device: {device}")

    train_loader, test_loader = load_data()
    model = ModularBidirectionalRNN(input_size=28, hidden_size=63, output_size=10, output_mode="last").to(device)

    live_plot = LiveTrainingPlot(title="modRNN/test_mnist.py")
    accuracy = train(model, train_loader, test_loader, device, epochs=5, live_plot=live_plot)
    print(f"test accuracy: {accuracy:.4f}")
    assert accuracy > 0.90, f"expected >90% accuracy, got {accuracy:.4f}"


if __name__ == "__main__":
    main()
