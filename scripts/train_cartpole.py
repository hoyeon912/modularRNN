import argparse
import copy
import json
import sys
from datetime import datetime
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


def rollout_episode(model, env, device, max_steps: int = 500, render_fn=None) -> float:
    # Carries hidden state incrementally via model.step() when the model supports it
    # (ModularRNN/CNNModularRNN) instead of replaying the whole growing state history
    # through model() on every env step -- verified mathematically identical to the
    # batched forward pass, but O(T) instead of O(T^2), which matters once a per-frame
    # CNN encoder makes each individual step expensive.
    #
    # SimpleRNN/BidirectionalRNN have no step()/init_hidden(): they're genuinely
    # bidirectional (nn.RNN(..., bidirectional=True) / an explicit backward cell pass),
    # so a timestep's output depends on reprocessing the sequence backward from wherever
    # it currently ends -- there's no way to carry that forward incrementally, it must
    # be recomputed from the full history every step. Fall back to that for such models.
    incremental = hasattr(model, "step") and hasattr(model, "init_hidden")
    model.eval()
    state, _ = env.reset()
    states = [render_fn(state) if render_fn else state]
    h = model.init_hidden(1, device) if incremental else None

    def q_for_latest_obs():
        nonlocal h
        if incremental:
            x = torch.tensor(states[-1], dtype=torch.float32, device=device).unsqueeze(0)
            q, h = model.step(x, h)
            return q[0]
        x = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
        return model(x)[0, -1, :]

    total_reward = 0.0
    with torch.no_grad():
        q = q_for_latest_obs()
        for _ in range(max_steps):
            action = q.argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
            states.append(render_fn(state) if render_fn else state)
            q = q_for_latest_obs()
    return total_reward


def evaluate_reward(model, device, num_episodes: int = 3, max_steps: int = 500, render_fn=None) -> float:
    env = gym.make("CartPole-v1")
    total = 0.0
    for _ in range(num_episodes):
        total += rollout_episode(model, env, device, max_steps=max_steps, render_fn=render_fn)
    env.close()
    return total / num_episodes


def compute_returns(rewards, gamma: float = 0.99):
    returns = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        returns.insert(0, running)
    return returns


def _archive_if_exists(path: str) -> None:
    """Renames an existing file aside with a timestamp suffix instead of letting a
    re-run silently overwrite the previous training's log/model."""
    p = Path(path)
    if p.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        p.rename(p.with_name(f"{p.stem}.{timestamp}{p.suffix}"))


def save_results(model, history, results_path: str, model_path: str) -> None:
    _archive_if_exists(results_path)
    _archive_if_exists(model_path)
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
    lr: float = 1e-3,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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


# --- hf path (TD learning, per CLAUDE.md: delta = r + gamma*V(t+1) - V(t), output = expected
# future reward per action) ---


def collect_episode_epsilon_greedy_hf(model, env, device, epsilon: float, max_steps: int = 500, render_fn=None):
    """Rolls out one episode with epsilon-greedy action selection over the model's Q-value
    output, growing the state sequence each step so the model sees true recurrent context
    (matching how evaluate_reward/rollout_episode behave at eval time). Returns the full
    `states` list (one more entry than actions/rewards: the trailing next-state needed for
    the TD bootstrap target) plus `terminated` (True only on genuine termination, not a
    max-steps truncation, since only genuine termination has zero bootstrap value).

    `render_fn`, if given, converts each raw env state into the array actually stored/fed
    to the model (e.g. a rendered image for a CNN front-end) -- env.step()/reset() still see
    the raw physics state, only the model's input representation changes.

    Uses model.step() to carry hidden state incrementally when the model supports it,
    rather than replaying the whole growing history every env step -- see rollout_episode
    for why (same O(T^2) -> O(T) fix, verified numerically identical to the batched
    forward; falls back to the full-history recompute for bidirectional models, which
    can't be stepped incrementally at all)."""
    incremental = hasattr(model, "step") and hasattr(model, "init_hidden")
    model.eval()
    state, _ = env.reset()
    states = [render_fn(state) if render_fn else state]
    actions = []
    rewards = []
    terminated = False
    h = model.init_hidden(1, device) if incremental else None

    def q_for_latest_obs():
        nonlocal h
        if incremental:
            x = torch.tensor(states[-1], dtype=torch.float32, device=device).unsqueeze(0)
            q, h = model.step(x, h)
            return q[0]
        x = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
        return model(x)[0, -1, :]

    with torch.no_grad():
        q = q_for_latest_obs()
        for _ in range(max_steps):
            if torch.rand(()).item() < epsilon:
                action = int(torch.randint(0, q.shape[0], (1,)).item())
            else:
                action = int(q.argmax().item())
            state, reward, term, trunc, _ = env.step(action)
            actions.append(action)
            rewards.append(reward)
            states.append(render_fn(state) if render_fn else state)
            if term or trunc:
                terminated = term
                break
            q = q_for_latest_obs()
    return states, actions, rewards, terminated


def build_td_objective(episodes, device, online_model, target_model, gamma: float = 0.99):
    """`episodes` is a list of (states, actions, rewards, terminated) tuples, one per
    collected episode. Returns objective_fn(model) -> (loss, z) closure for HFOptimizer.step().
    `z` is the concatenated taken-Q-values -- the only tensor entering the loss nonlinearly
    through `model`'s parameters (semi-gradient TD, matching CLAUDE.md's
    delta = r + gamma*V(t+1) - V(t)).

    The bootstrap target V(t+1) is read from a separate, periodically-synced `target_model`
    rather than from `model` itself: regressing a live network toward a target computed from
    its own most recent update is a moving target that a second-order optimizer's confident
    Newton steps chase and overshoot every update (observed empirically as *rising* loss and
    reward stuck at the random baseline) -- the standard DQN fix is a frozen target network.

    Double-Q target: the next-state action is chosen by `model` (argmax), but its value is
    read from `target_model`. Plain `target_model.max(-1)` combines selection and evaluation
    in the same (lagged, still-noisy) network, which systematically overestimates Q-values --
    observed empirically as TD loss climbing ~20x over a run even as reward improved, i.e. the
    targets were drifting upward faster than the model could track them. Decoupling selection
    (online) from evaluation (target) removes that positive bias (van Hasselt et al., 2016)."""
    episode_states = [torch.tensor(s, dtype=torch.float32, device=device) for s, _, _, _ in episodes]
    episode_actions = [torch.tensor(a, dtype=torch.long, device=device) for _, a, _, _ in episodes]
    episode_rewards = [torch.tensor(r, dtype=torch.float32, device=device) for _, _, r, _ in episodes]
    episode_terminated = [term for _, _, _, term in episodes]

    with torch.no_grad():
        target_next_max = []
        for states_t in episode_states:
            online_next = online_model(states_t.unsqueeze(0))[0, 1:]
            best_actions = online_next.argmax(dim=-1)
            target_next = target_model(states_t.unsqueeze(0))[0, 1:]
            target_next_max.append(target_next.gather(1, best_actions.unsqueeze(1)).squeeze(1))

    def objective_fn(model):
        q_taken_parts = []
        target_parts = []
        for states_t, actions_t, rewards_t, terminated, q_next_max in zip(
            episode_states, episode_actions, episode_rewards, episode_terminated, target_next_max
        ):
            q_all = model(states_t.unsqueeze(0))[0]  # (T+1, num_actions), full recurrent replay
            steps = actions_t.shape[0]
            q_taken = q_all[:steps].gather(1, actions_t.unsqueeze(1)).squeeze(1)  # (T,)
            mask = torch.ones(steps, device=device)
            if terminated:
                mask[-1] = 0.0  # terminal state has zero future value
            target = rewards_t + gamma * q_next_max * mask
            q_taken_parts.append(q_taken)
            target_parts.append(target)

        z = torch.cat(q_taken_parts)
        target = torch.cat(target_parts).detach()
        loss = 0.5 * ((z - target) ** 2).mean()
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
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.02,
    epsilon_decay_updates: int = 150,
    initial_damping: float = 1.0,
    damping_min: float = 1e-8,
    required_streak: int = 3,
    target_sync_interval: int = 10,
    replay_capacity: int = 300,
    replay_batch_size: int = 16,
    render_fn=None,
):
    # cg_min_iter=1, checkpoint_every=1: this model's loss surface (a tanh RNN unrolled over
    # long CartPole episodes) is only locally quadratic within a very small trust region, so
    # CG must backtrack against the *true* loss starting from its very first iterate -- with
    # the old defaults (min_iter=10, checkpoint_every=5) CG always wandered well outside that
    # region before ever being checked, so backtracking picked the zero step every time and
    # training never moved at all.
    optimizer = HFOptimizer(
        model,
        curvature="mse",
        initial_damping=initial_damping,
        damping_min=damping_min,
        cg_max_iter=20,
        cg_min_iter=1,
        checkpoint_every=1,
    )
    target_model = copy.deepcopy(model)
    target_model.eval()
    env = gym.make("CartPole-v1")
    avg_reward = 0.0
    history = []
    streak = 0
    # Episode-level replay buffer (Q-learning's max_a' bootstrap is off-policy, so replaying
    # episodes collected under an older, worse policy remains statistically valid, same as
    # DQN's replay memory) -- each fresh HF step otherwise trained on ~episodes_per_update
    # episodes and then discarded them, which is extremely sample-inefficient: a 5000-update
    # run with fresh-only data plateaued at reward ~10 (random-baseline level) with no upward
    # trend at all. Whole episodes (not single transitions) are stored since the recurrent
    # Q-value at any timestep depends on the full trajectory replayed from t=0.
    replay_buffer = []
    try:
        for update in range(num_updates):
            frac = min(1.0, update / max(1, epsilon_decay_updates))
            epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)
            for _ in range(episodes_per_update):
                replay_buffer.append(
                    collect_episode_epsilon_greedy_hf(model, env, device, epsilon, render_fn=render_fn)
                )
                if len(replay_buffer) > replay_capacity:
                    replay_buffer.pop(0)

            batch_size = min(replay_batch_size, len(replay_buffer))
            idx = torch.randperm(len(replay_buffer))[:batch_size]
            episodes = [replay_buffer[i] for i in idx]

            objective_fn = build_td_objective(episodes, device, model, target_model, gamma=gamma)
            # collect/evaluate leave the model in eval() mode; cuDNN-backed RNNs (SimpleRNN)
            # refuse to run backward in eval mode, so switch back before the optimizer step.
            model.train()
            diagnostics = optimizer.step(objective_fn)
            if (update + 1) % target_sync_interval == 0:
                target_model.load_state_dict(model.state_dict())
            avg_reward = evaluate_reward(model, device, render_fn=render_fn)
            print(
                f"update {update + 1}/{num_updates} loss {diagnostics['loss_after']:.4f} "
                f"reward {avg_reward:.1f} damping {optimizer.damping:.4g} eps {epsilon:.3f}"
            )
            history.append({"update": update + 1, "loss": diagnostics["loss_after"], "reward": avg_reward})
            if live_plot is not None:
                live_plot.update(update + 1, diagnostics["loss_after"], avg_reward)
            if device.type == "mps" and (update + 1) % 10 == 0:
                torch.mps.empty_cache()

            streak = streak + 1 if avg_reward >= 500 else 0
            if streak >= required_streak:
                print(f"reached max reward (500) for {streak} continuous checks at update {update + 1}, stopping")
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
    parser.add_argument("--lr", type=float, default=1e-3, help="ignored when --optimizer=hf")
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
    train_kwargs = {"lr": args.lr} if args.optimizer == "adam" else {}
    avg_reward, _ = train_fn(
        model,
        device,
        num_updates=args.num_updates,
        episodes_per_update=episodes_per_update,
        live_plot=live_plot,
        results_path=args.results_path,
        model_path=args.model_path,
        **train_kwargs,
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
