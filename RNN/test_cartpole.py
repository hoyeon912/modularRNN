import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from heuristic_policy import heuristic_action
from live_plot import LiveTrainingPlot
from model import SimpleRNN, get_device


def collect_episodes(num_episodes: int, seed: int):
    env = gym.make("CartPole-v1")
    episodes = []
    for i in range(num_episodes):
        state, _ = env.reset(seed=seed + i)
        states, actions = [], []
        done = False
        while not done:
            action = heuristic_action(state)
            states.append(state)
            actions.append(action)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        episodes.append((np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64)))
    env.close()
    return episodes


class CartPoleSequenceDataset(Dataset):
    def __init__(self, episodes):
        self.episodes = episodes

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        states, actions = self.episodes[idx]
        return torch.from_numpy(states), torch.from_numpy(actions)


def collate_pad(batch):
    lengths = [states.shape[0] for states, _ in batch]
    max_len = max(lengths)
    padded_states = torch.zeros(len(batch), max_len, 4)
    padded_actions = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (states, actions) in enumerate(batch):
        length = states.shape[0]
        padded_states[i, :length] = states
        padded_actions[i, :length] = actions
        mask[i, :length] = True
    return padded_states, padded_actions, mask


def train(model, train_loader, test_loader, device, epochs: int, lr: float = 1e-3, live_plot=None) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction="none")
    accuracy = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for states, actions, mask in train_loader:
            states, actions, mask = states.to(device), actions.to(device), mask.to(device)
            optimizer.zero_grad()
            logits = model(states)  # (batch, seq_len, 2)
            loss_per_step = criterion(logits.transpose(1, 2), actions)  # (batch, seq_len)
            loss = (loss_per_step * mask).sum() / mask.sum()
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
    for states, actions, mask in loader:
        states, actions, mask = states.to(device), actions.to(device), mask.to(device)
        logits = model(states)
        preds = logits.argmax(dim=-1)
        correct += ((preds == actions) & mask).sum().item()
        total += mask.sum().item()
    return correct / total


def main():
    device = get_device()
    print(f"using device: {device}")

    train_episodes = collect_episodes(num_episodes=200, seed=0)
    test_episodes = collect_episodes(num_episodes=50, seed=1000)

    train_loader = DataLoader(
        CartPoleSequenceDataset(train_episodes), batch_size=16, shuffle=True, collate_fn=collate_pad
    )
    test_loader = DataLoader(
        CartPoleSequenceDataset(test_episodes), batch_size=16, shuffle=False, collate_fn=collate_pad
    )

    model = SimpleRNN(input_size=4, hidden_size=32, output_size=2, output_mode="all").to(device)

    live_plot = LiveTrainingPlot(title="RNN/test_cartpole.py")
    accuracy = train(model, train_loader, test_loader, device, epochs=10, live_plot=live_plot)
    print(f"per-timestep action accuracy: {accuracy:.4f}")
    assert accuracy > 0.90, f"expected >90% action accuracy, got {accuracy:.4f}"


if __name__ == "__main__":
    main()
