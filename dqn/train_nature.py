"""Run with python -m dqn.train_nature --help from the repository root."""

import argparse
from collections import deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .nature import DQNAgent, DQNConfig, ReplayMemory


def make_env(env_id, render_mode=None):
    """Return single uint8 frames; ReplayMemory constructs the 4-frame stack."""
    if env_id.startswith("ALE/"):
        return make_atari(env_id, render_mode)
    if env_id != "CartPole-v1":
        raise ValueError("supported environments: CartPole-v1 or ALE/*")

    import gymnasium as gym

    # Always render to pixels for the CNN, even when no window is requested.
    # CartPole takes one physics step per action, with no Atari action repeat.
    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.AddRenderObservation(env, render_only=True)
    # Use dqn/cart_pole.py's wrappers, keeping Nature's luminance-first order.
    # Replay zero-pads uint8 stacks; NatureDQN normalizes once on input.
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    if render_mode == "human":
        env = gym.wrappers.HumanRendering(env)
    return env


def make_history(observation):
    """Evaluation uses the same initial stack as training replay."""
    padding = np.zeros_like(observation)
    return deque([padding] * 3 + [observation], maxlen=4)


def make_atari(env_id, render_mode=None):
    import gymnasium as gym

    try:
        import ale_py
    except ImportError as error:
        raise RuntimeError(
            "Atari requires: python -m pip install -r dqn/requirements.txt"
        ) from error

    gym.register_envs(ale_py)
    env = gym.make(
        env_id,
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=False,
        render_mode=render_mode,
        max_episode_steps=108_000,
    )
    # Only this wrapper repeats actions. Training handles life-loss targets
    # separately so losing a life does not reset the entire game.
    return gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False,
    )


def save_checkpoint(path, agent, step, env_id):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "online": agent.online.state_dict(),
            "target": agent.target.state_dict(),
            "optimizer": agent.optimizer.state_dict(),
            "config": asdict(agent.config),
            "n_actions": agent.n_actions,
            "step": step,
            "env_id": env_id,
        },
        path,
    )


def train(
    env,
    config,
    total_steps,
    device="cpu",
    seed=0,
    checkpoint="dqn/checkpoints/nature.pt",
    log_interval=10_000,
    save_interval=125_000,
    env_id=None,
    writer=None,
):
    """Train on uint8 84x84 frames; env.step returns Gymnasium's 5-tuple.

    Caller owns and closes env. Info['lives'] enables Atari life-loss TD
    terminals. Truncations reset the history but still bootstrap from the
    final observation. Reported rewards are original, unclipped game scores.
    Optional TensorBoard writer is owned and closed by the caller.
    """
    if total_steps <= 0 or log_interval <= 0 or save_interval <= 0:
        raise ValueError("step counts and logging intervals must be positive")
    torch.manual_seed(seed)
    agent = DQNAgent(env.action_space.n, config, device, seed)
    memory = ReplayMemory(config.replay_capacity, seed)
    observation, info = env.reset(seed=seed)
    memory.start_episode(observation)
    lives = info.get("lives", 0)
    episode_return, episodes, updates, last_loss = 0.0, 0, 0, None
    for step in range(1, total_steps + 1):
        epsilon = config.epsilon(step - 1)
        action = agent.act(memory.state, epsilon)
        next_frame, reward, terminated, truncated, info = env.step(action)
        next_lives = info.get("lives", lives)
        life_lost = next_lives < lives
        memory.append(action, reward, next_frame, terminated or life_lost)
        episode_return += float(reward)

        if (
            step > config.learning_starts
            and step % config.train_frequency == 0
            and len(memory) >= config.batch_size
        ):
            last_loss = agent.update(memory.sample(config.batch_size))
            updates += 1
            if writer is not None:
                writer.add_scalar("loss/td_loss", last_loss, step)
        if step % config.target_update_interval == 0:
            agent.sync_target()

        if terminated or truncated:
            episodes += 1
            if writer is not None:
                writer.add_scalar(
                    "reward/episode_return", episode_return, episodes
                )
            print(
                f"step={step} episode={episodes} return={episode_return:g}",
                flush=True,
            )
            observation, info = env.reset()
            memory.start_episode(observation)
            lives = info.get("lives", 0)
            episode_return = 0.0
        else:
            lives = next_lives
            if life_lost:
                memory.start_episode(next_frame)

        if step % log_interval == 0:
            print(
                f"step={step} epsilon={epsilon:.4f} replay={len(memory)} "
                f"updates={updates} loss={last_loss}",
                flush=True,
            )
        if checkpoint is not None and step % save_interval == 0:
            save_checkpoint(checkpoint, agent, step, env_id)

    if checkpoint is not None:
        save_checkpoint(checkpoint, agent, total_steps, env_id)
    return {
        "steps": total_steps,
        "episodes": episodes,
        "updates": updates,
        "loss": last_loss,
    }


def evaluate(env, agent, episodes=10, seed=0):
    """Full-game evaluation with epsilon=.05 and no reward clipping."""
    returns = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        history = make_history(observation)
        score = 0.0
        while True:
            action = agent.act(np.stack(history), epsilon=0.05)
            observation, reward, terminated, truncated, _ = env.step(action)
            history.append(observation)
            score += float(reward)
            if terminated or truncated:
                break
        returns.append(score)
        print(f"episode={episode + 1} return={score:g}", flush=True)
    print(f"mean_return={np.mean(returns):g} std_return={np.std(returns):g}")
    return returns


def main():
    parser = argparse.ArgumentParser(description="Nature 2015 DQN (PyTorch)")
    parser.add_argument("--env", default="ALE/Breakout-v5")
    parser.add_argument(
        "--total-steps",
        type=int,
        default=50_000_000,
        help="agent decisions; 4 ALE frames or 1 CartPole physics step each",
    )
    parser.add_argument("--replay-capacity", type=int, default=1_000_000)
    parser.add_argument("--learning-starts", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--exploration-steps", type=int, default=1_000_000)
    parser.add_argument("--target-update-interval", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", default="dqn/checkpoints/nature.pt")
    parser.add_argument("--log-interval", type=int, default=10_000)
    parser.add_argument(
        "--log-dir",
        default="runs/nature_dqn",
        help="TensorBoard root; a separate environment/timestamp run is created",
    )
    parser.add_argument("--save-interval", type=int, default=125_000)
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="evaluate the checkpoint instead of training",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    device = args.device
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    config = DQNConfig(
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        learning_starts=args.learning_starts,
        exploration_steps=args.exploration_steps,
        target_update_interval=args.target_update_interval,
    )
    saved = None
    if args.evaluate:
        saved = torch.load(
            args.checkpoint, map_location=device, weights_only=True
        )
        if saved["env_id"] is not None and saved["env_id"] != args.env:
            parser.error("--env must match the checkpoint environment")
    env = make_env(args.env, "human" if args.render else None)
    try:
        if saved is not None:
            agent = DQNAgent(
                saved["n_actions"],
                DQNConfig(**saved["config"]),
                device,
                args.seed,
            )
            if env.action_space.n != agent.n_actions:
                raise ValueError("checkpoint action count differs from env")
            agent.online.load_state_dict(saved["online"])
            evaluate(env, agent, args.episodes, args.seed)
        else:
            from torch.utils.tensorboard import SummaryWriter

            run_name = (
                f"{args.env.replace('/', '_')}_"
                f"{datetime.now().astimezone():%Y%m%d-%H%M%S-%f}_seed{args.seed}"
            )
            run_dir = Path(args.log_dir) / run_name
            print(f"env={args.env} device={device} config={config}", flush=True)
            print(f"TensorBoard log: {run_dir}", flush=True)
            # Context manager flushes and closes even on interruption/errors.
            with SummaryWriter(log_dir=str(run_dir), flush_secs=10) as writer:
                train(
                    env,
                    config,
                    args.total_steps,
                    device,
                    args.seed,
                    args.checkpoint,
                    args.log_interval,
                    args.save_interval,
                    args.env,
                    writer=writer,
                )
    finally:
        env.close()


if __name__ == "__main__":
    main()
