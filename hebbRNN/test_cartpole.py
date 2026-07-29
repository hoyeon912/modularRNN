# hebbRNN/test_cartpole.py
import json

import gymnasium as gym
import torch

from hf_optimizer import HFOptimizer
from live_plot import LiveTrainingPlot
from model import ModularBidirectionalRNN, get_device


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


def collect_episode_stochastic(model, env, device, max_steps: int = 500):
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


def save_results(model, history, results_path: str, model_path: str) -> None:
    with open(results_path, "w") as f:
        json.dump(history, f, indent=2)
    torch.save(model.state_dict(), model_path)
    print(f"saved {len(history)} update(s) of history to {results_path}, model weights to {model_path}")


def train(
    model,
    device,
    num_updates: int,
    episodes_per_update: int = 8,
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
                states, actions, rewards = collect_episode_stochastic(model, env, device)
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


def main():
    device = get_device()
    print(f"using device: {device}")

    model = ModularBidirectionalRNN(input_size=4, hidden_size=300, output_size=2, output_mode="all").to(device)

    live_plot = LiveTrainingPlot(title="hebbRNN/test_cartpole.py", metrics=("loss", "reward"))
    avg_reward, _ = train(model, device, num_updates=5, episodes_per_update=4, live_plot=live_plot)
    print(f"average reward: {avg_reward:.1f}")
    assert avg_reward > 150, f"expected average reward > 150, got {avg_reward:.1f}"

    try:
        render_env = gym.make("CartPole-v1", render_mode="human")
        reward = rollout_episode(model, render_env, device)
        render_env.close()
        print(f"rendered episode reward: {reward:.0f}")
    except Exception as e:
        print(f"render skipped (no display available): {e}")


if __name__ == "__main__":
    main()
