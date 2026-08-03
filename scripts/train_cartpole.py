import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import torch

from models.bidirectional_rnn import BidirectionalRNN
from models.common import get_device
from models.modular_rnn import ModularRNN
from models.simple_rnn import SimpleRNN
from scripts.hf_optimizer import HFOptimizer
from scripts.live_plot import LiveTrainingPlot

MODEL_DEFAULTS = {
    "simple_rnn": {"cls": SimpleRNN, "hidden_size": 32, "episodes_per_update": 8},
    "bidirectional_rnn": {"cls": BidirectionalRNN, "hidden_size": 32, "episodes_per_update": 4},
    "modular_rnn": {"cls": ModularRNN, "hidden_size": 300, "episodes_per_update": 4},
}


def build_model(model_name: str, hidden_size: int):
    cls = MODEL_DEFAULTS[model_name]["cls"]
    return cls(input_size=4, hidden_size=hidden_size, output_size=2, output_mode="all")


def rollout_episode(model, env, device, max_steps: int = 500) -> float:
    model.eval()
    state, _ = env.reset()
    states = [state]
    total_reward = 0.0
    with torch.no_grad():
        for _ in range(max_steps):
            x = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
            logits = model(x)
            action = logits[0, -1].argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            states.append(state)
            if terminated or truncated:
                break
    return total_reward


def evaluate_reward(model, device, num_episodes: int = 3, max_steps: int = 500) -> float:
    env = gym.make("CartPole-v1")
    total = 0.0
    for _ in range(num_episodes):
        total += rollout_episode(model, env, device, max_steps=max_steps)
    env.close()
    return total / num_episodes


def compute_returns(rewards, gamma: float = 0.99):
    returns = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        returns.insert(0, running)
    return returns


def save_results(model, history, results_path: str, model_path: str) -> None:
    with open(results_path, "w") as f:
        json.dump(history, f, indent=2)
    torch.save(model.state_dict(), model_path)
    print(f"saved {len(history)} update(s) of history to {results_path}, model weights to {model_path}")


# --- adam / REINFORCE-with-autograd path ---


def collect_episode_stochastic_adam(model, env, device, max_steps: int = 500):
    model.train()
    state, _ = env.reset()
    states = [state]
    log_probs = []
    entropies = []
    rewards = []
    for _ in range(max_steps):
        x = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
        logits = model(x)
        dist = torch.distributions.Categorical(logits=logits[0, -1])
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())
        state, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        states.append(state)
        if terminated or truncated:
            break
    return log_probs, entropies, rewards


def reinforce_update(model, optimizer, episode_batch, gamma: float = 0.99, entropy_coef: float = 0.01) -> float:
    all_log_probs = []
    all_entropies = []
    all_returns = []
    for log_probs, entropies, rewards in episode_batch:
        all_log_probs.extend(log_probs)
        all_entropies.extend(entropies)
        all_returns.extend(compute_returns(rewards, gamma))

    returns_tensor = torch.tensor(all_returns, dtype=torch.float32, device=all_log_probs[0].device)
    baseline = returns_tensor.mean()
    advantages = returns_tensor - baseline

    log_probs_tensor = torch.stack(all_log_probs)
    entropy_tensor = torch.stack(all_entropies)
    policy_loss = -(log_probs_tensor * advantages).mean() - entropy_coef * entropy_tensor.mean()

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    return policy_loss.item()


def train_adam(
    model,
    device,
    num_updates: int,
    episodes_per_update: int,
    live_plot=None,
    results_path: str = "cartpole_results.json",
    model_path: str = "cartpole_model.pt",
):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    env = gym.make("CartPole-v1")
    avg_reward = 0.0
    history = []
    try:
        for update in range(num_updates):
            batch = [collect_episode_stochastic_adam(model, env, device) for _ in range(episodes_per_update)]
            loss = reinforce_update(model, optimizer, batch)
            avg_reward = evaluate_reward(model, device)
            print(f"update {update + 1}/{num_updates} loss {loss:.4f} reward {avg_reward:.1f}")
            history.append({"update": update + 1, "loss": loss, "reward": avg_reward})
            if live_plot is not None:
                live_plot.update(update + 1, loss, avg_reward)
            if device.type == "mps" and (update + 1) % 10 == 0:
                torch.mps.empty_cache()
            if avg_reward >= 500:
                print(f"reached max reward (500) at update {update + 1}, stopping early")
                break
    finally:
        env.close()
        save_results(model, history, results_path, model_path)
    return avg_reward, history


# --- hf path ---


def collect_episode_stochastic_hf(model, env, device, max_steps: int = 500):
    model.eval()
    state, _ = env.reset()
    states = [state]
    actions = []
    rewards = []
    with torch.no_grad():
        for _ in range(max_steps):
            x = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
            logits = model(x)
            dist = torch.distributions.Categorical(logits=logits[0, -1])
            action = dist.sample()
            state, reward, terminated, truncated, _ = env.step(action.item())
            actions.append(action.item())
            rewards.append(reward)
            states.append(state)
            if terminated or truncated:
                break
    return states[:-1], actions, rewards


def build_reinforce_objective(states, actions, returns, device, entropy_coef: float = 0.01):
    """`states`/`actions`/`returns` are flat lists pooled across an episode batch (one
    entry per visited timestep). Returns an objective_fn(model) -> (loss, z) closure for
    HFOptimizer.step() — recomputes the forward pass under the model's *current* params
    each call, so it's valid across every CG iteration within one HF step."""
    states_tensor = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(1)  # (N, 1, obs_dim)
    actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
    returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
    baseline = returns_tensor.mean()
    advantages = returns_tensor - baseline

    def objective_fn(model):
        z = model(states_tensor)[:, -1, :]  # (N, num_actions), output_mode="all" sliced to last step of each 1-step sequence
        dist = torch.distributions.Categorical(logits=z)
        log_probs = dist.log_prob(actions_tensor)
        loss = -(log_probs * advantages).mean() - entropy_coef * dist.entropy().mean()
        return loss, z

    return objective_fn


def train_hf(
    model,
    device,
    num_updates: int,
    episodes_per_update: int,
    live_plot=None,
    results_path: str = "cartpole_results.json",
    model_path: str = "cartpole_model.pt",
):
    optimizer = HFOptimizer(model, curvature="categorical")
    env = gym.make("CartPole-v1")
    avg_reward = 0.0
    history = []
    try:
        for update in range(num_updates):
            all_states, all_actions, all_returns = [], [], []
            for _ in range(episodes_per_update):
                states, actions, rewards = collect_episode_stochastic_hf(model, env, device)
                all_states.extend(states)
                all_actions.extend(actions)
                all_returns.extend(compute_returns(rewards))

            objective_fn = build_reinforce_objective(all_states, all_actions, all_returns, device)
            diagnostics = optimizer.step(objective_fn)
            avg_reward = evaluate_reward(model, device)
            print(
                f"update {update + 1}/{num_updates} loss {diagnostics['loss_after']:.4f} "
                f"reward {avg_reward:.1f} damping {optimizer.damping:.4g}"
            )
            history.append({"update": update + 1, "loss": diagnostics["loss_after"], "reward": avg_reward})
            if live_plot is not None:
                live_plot.update(update + 1, diagnostics["loss_after"], avg_reward)
            if device.type == "mps" and (update + 1) % 10 == 0:
                torch.mps.empty_cache()
            if avg_reward >= 500:
                print(f"reached max reward (500) at update {update + 1}, stopping early")
                break
    finally:
        env.close()
        save_results(model, history, results_path, model_path)
    return avg_reward, history


def parse_args():
    parser = argparse.ArgumentParser(description="Train an RNN variant on CartPole-v1 with REINFORCE.")
    parser.add_argument("--model", choices=sorted(MODEL_DEFAULTS), default="simple_rnn")
    parser.add_argument("--optimizer", choices=("adam", "hf"), default="adam")
    parser.add_argument("--hidden-size", type=int, default=None, help="defaults to the model's usual size")
    parser.add_argument("--num-updates", type=int, default=5)
    parser.add_argument("--episodes-per-update", type=int, default=None, help="defaults to the model's usual count")
    parser.add_argument("--min-reward", type=float, default=150.0)
    parser.add_argument("--results-path", default="cartpole_results.json")
    parser.add_argument("--model-path", default="cartpole_model.pt")
    parser.add_argument("--render", action="store_true", help="render one episode after training")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"using device: {device}")

    defaults = MODEL_DEFAULTS[args.model]
    hidden_size = args.hidden_size or defaults["hidden_size"]
    episodes_per_update = args.episodes_per_update or defaults["episodes_per_update"]

    model = build_model(args.model, hidden_size).to(device)

    live_plot = LiveTrainingPlot(
        title=f"scripts/train_cartpole.py --model {args.model} --optimizer {args.optimizer}",
        metrics=("loss", "reward"),
    )
    train_fn = train_adam if args.optimizer == "adam" else train_hf
    avg_reward, _ = train_fn(
        model,
        device,
        num_updates=args.num_updates,
        episodes_per_update=episodes_per_update,
        live_plot=live_plot,
        results_path=args.results_path,
        model_path=args.model_path,
    )
    print(f"average reward: {avg_reward:.1f}")
    assert avg_reward > args.min_reward, f"expected average reward > {args.min_reward:.0f}, got {avg_reward:.1f}"

    if args.render:
        try:
            render_env = gym.make("CartPole-v1", render_mode="human")
            reward = rollout_episode(model, render_env, device)
            render_env.close()
            print(f"rendered episode reward: {reward:.0f}")
        except Exception as e:
            print(f"render skipped (no display available): {e}")


if __name__ == "__main__":
    main()
