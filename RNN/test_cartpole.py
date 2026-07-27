import gymnasium as gym
import torch

from live_plot import LiveTrainingPlot
from model import SimpleRNN, get_device


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


def collect_episode_stochastic(model, env, device, max_steps: int = 500):
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


def compute_returns(rewards, gamma: float = 0.99):
    returns = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        returns.insert(0, running)
    return returns


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


def train(model, device, num_updates: int, episodes_per_update: int = 8, live_plot=None) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    env = gym.make("CartPole-v1")
    avg_reward = 0.0
    for update in range(num_updates):
        batch = [collect_episode_stochastic(model, env, device) for _ in range(episodes_per_update)]
        loss = reinforce_update(model, optimizer, batch)
        avg_reward = evaluate_reward(model, device)
        print(f"update {update + 1}/{num_updates} loss {loss:.4f} reward {avg_reward:.1f}")
        if live_plot is not None:
            live_plot.update(update + 1, loss, avg_reward)
        if device.type == "mps" and (update + 1) % 10 == 0:
            torch.mps.empty_cache()
        if avg_reward >= 500:
            print(f"reached max reward (500) at update {update + 1}, stopping early")
            break
    env.close()
    return avg_reward


def main():
    device = get_device()
    print(f"using device: {device}")

    model = SimpleRNN(input_size=4, hidden_size=32, output_size=2, output_mode="all").to(device)

    live_plot = LiveTrainingPlot(title="RNN/test_cartpole.py", metrics=("loss", "reward"))
    avg_reward = train(model, device, num_updates=100, live_plot=live_plot)
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
