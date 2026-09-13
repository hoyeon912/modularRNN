"""Fast Nature DQN regression tests; no Atari ROMs required."""

import numpy as np
import pytest
import torch

from dqn.nature import (
    DQNAgent,
    DQNConfig,
    NatureDQN,
    NatureRMSprop,
    ReplayMemory,
    td_targets,
)


def frame(value):
    return np.full((84, 84), value, dtype=np.uint8)


def test_network_shape_and_uint8_normalization():
    net = NatureDQN(6)
    pixels = torch.randint(256, (2, 4, 84, 84), dtype=torch.uint8)
    assert net(pixels).shape == (2, 6)
    torch.testing.assert_close(net(pixels), net(pixels.float() / 255))
    assert sum(p.numel() for p in net.parameters()) == 1_687_206


def test_targets_use_target_max_and_mask_only_terminal():
    result = td_targets(
        torch.tensor([1.0, -1.0, 0.0]),
        torch.tensor([True, False, False]),
        torch.tensor([[100.0, 200.0], [2.0, 4.0], [3.0, 1.0]]),
        0.99,
    )
    torch.testing.assert_close(result, torch.tensor([1.0, 2.96, 2.97]))


def test_rmsprop_matches_two_hand_calculated_updates():
    weight = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = NatureRMSprop([weight], lr=0.00025)
    expected, mean, square = 1.0, 0.0, 0.0
    for gradient in (2.0, -3.0):
        weight.grad = torch.tensor([gradient])
        optimizer.step()
        mean = 0.95 * mean + 0.05 * gradient
        square = 0.95 * square + 0.05 * gradient**2
        expected -= 0.00025 * gradient / (square - mean**2 + 0.01) ** 0.5
        assert weight.item() == pytest.approx(expected, abs=1e-7)


def test_replay_preserves_history_after_overwrite_and_reset():
    memory = ReplayMemory(2, seed=1)
    memory.start_episode(frame(1))
    for value in (2, 3, 4):
        next_frame = frame(value)
        memory.append(value, 5.0, next_frame, terminal=False)
        next_frame.fill(99)  # Stored pixels must own their memory.
    assert len(memory) == 2
    states, actions, rewards, next_states, terminals = memory.sample(2)
    for state, action, next_state in zip(states, actions, next_states):
        assert next_state[-1, 0, 0] == action
        torch.testing.assert_close(state[1:], next_state[:-1])
    assert rewards.tolist() == [1.0, 1.0]
    assert not terminals.any()
    memory.start_episode(frame(20))
    memory.append(0, -7, frame(21), terminal=True)
    states, actions, rewards, next_states, terminals = memory.sample(2)
    row = (actions == 0).nonzero().item()
    assert states[row, :, 0, 0].tolist() == [0, 0, 0, 20]
    assert next_states[row, :, 0, 0].tolist() == [0, 0, 20, 21]
    assert rewards[row].item() == -1
    assert terminals[row].item()


def test_epsilon_warmup_and_linear_decay():
    config = DQNConfig()
    assert config.epsilon(0) == 1.0
    assert config.epsilon(50_000) == 1.0
    assert config.epsilon(550_000) == pytest.approx(0.55)
    assert config.epsilon(1_050_000) == pytest.approx(0.1)
    assert config.epsilon(2_000_000) == pytest.approx(0.1)


def test_update_changes_online_only_until_hard_sync():
    torch.manual_seed(0)
    agent = DQNAgent(3, device="cpu")
    memory = ReplayMemory(4)
    memory.start_episode(frame(1))
    for value in range(2, 6):
        memory.append(1, 1, frame(value), terminal=True)
        memory.start_episode(frame(value))
    before = {
        key: value.clone() for key, value in agent.online.state_dict().items()
    }
    loss = agent.update(memory.sample(4))
    assert np.isfinite(loss)
    assert any(
        not torch.equal(value, before[key])
        for key, value in agent.online.state_dict().items()
    )
    for key, value in agent.target.state_dict().items():
        torch.testing.assert_close(value, before[key])
    assert all(p.grad is None for p in agent.target.parameters())
    agent.sync_target()
    for key, value in agent.target.state_dict().items():
        torch.testing.assert_close(value, agent.online.state_dict()[key])


def test_invalid_replay_input_is_rejected():
    memory = ReplayMemory(2)
    with pytest.raises(ValueError):
        memory.start_episode(np.zeros((84, 84), dtype=np.float32))
    with pytest.raises(ValueError):
        memory.sample(1)


def test_training_loop_and_checkpoint(tmp_path, monkeypatch):
    from gymnasium.spaces import Discrete
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )
    from torch.utils.tensorboard import SummaryWriter

    from dqn.train_nature import train

    class PixelEnv:
        action_space = Discrete(3)

        def __init__(self):
            self.steps = 0
            self.resets = 0

        def reset(self, seed=None):
            self.steps = 0
            self.resets += 1
            return frame(1), {"lives": 3}

        def step(self, action):
            assert self.action_space.contains(action)
            self.steps += 1
            # Lose a life at step 2; time limit at step 3.
            return (
                frame(self.steps + 1),
                2.0,
                False,
                self.steps == 3,
                {"lives": 3 if self.steps < 2 else 2},
            )

    config = DQNConfig(
        batch_size=2,
        replay_capacity=8,
        learning_starts=2,
        train_frequency=1,
        target_update_interval=2,
    )
    memory = ReplayMemory(8)
    monkeypatch.setattr("dqn.train_nature.ReplayMemory", lambda *args: memory)
    env = PixelEnv()
    checkpoint = tmp_path / "dqn.pt"
    log_dir = tmp_path / "tensorboard"
    with SummaryWriter(log_dir=str(log_dir)) as writer:
        result = train(
            env,
            config,
            total_steps=6,
            device="cpu",
            checkpoint=checkpoint,
            log_interval=100,
            writer=writer,
        )
    events = EventAccumulator(str(log_dir)).Reload()
    losses = events.Scalars("loss/td_loss")
    rewards = events.Scalars("reward/episode_return")
    assert [event.step for event in losses] == [3, 4, 5, 6]
    assert losses[-1].value == pytest.approx(result["loss"])
    assert all(np.isfinite(event.value) for event in losses)
    assert [event.step for event in rewards] == [1, 2]
    assert [event.value for event in rewards] == [6.0, 6.0]
    assert result["updates"] == 4
    assert result["episodes"] == 2
    assert env.resets == 3  # A lost life must not reset the actual game.
    # Life loss is a Bellman terminal; time limits retain bootstrapping.
    assert [entry[4] for entry in memory.memory] == [False, True, False] * 2
    assert [f[0, 0] for f in memory.memory[2][0]] == [0, 0, 0, 3]
    assert memory.memory[2][3][-1][0, 0] == 4
    saved = torch.load(checkpoint, weights_only=True)
    net = NatureDQN(3)
    net.load_state_dict(saved["online"])
    assert saved["step"] == 6
    assert saved["config"]["batch_size"] == 2
    assert saved["optimizer"]["state"]
    for key, value in saved["target"].items():
        torch.testing.assert_close(value, saved["online"][key])


def test_td_derivative_is_clipped_and_summed_across_batch():
    agent = DQNAgent(3)
    with torch.no_grad():
        for parameter in agent.online.parameters():
            parameter.zero_()
    states = torch.zeros((2, 4, 84, 84), dtype=torch.uint8)
    batch = (
        states,
        torch.tensor([0, 0]),
        torch.tensor([10.0, 10.0]),
        states,
        torch.tensor([True, True]),
    )
    assert agent.update(batch) == pytest.approx(9.5)
    torch.testing.assert_close(
        agent.online.q_head[-1].bias.grad, torch.tensor([-2.0, 0.0, 0.0])
    )


def test_cartpole_pixels_preserve_one_step_dynamics(monkeypatch):
    import gymnasium as gym

    from dqn.train_nature import make_env

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    env = make_env("CartPole-v1")
    reference = gym.make("CartPole-v1")
    try:
        pixels, _ = env.reset(seed=12)
        reference.reset(seed=12)
        assert env.action_space.n == 2
        assert env.observation_space.contains(pixels)
        assert pixels.shape == (84, 84) and pixels.dtype == np.uint8
        assert pixels.min() < pixels.max()  # Actual scene, not a blank frame.
        for action in (0, 1, 0, 1):
            pixels, reward, terminated, truncated, _ = env.step(action)
            _, expected_reward, expected_term, expected_trunc, _ = (
                reference.step(action)
            )
            assert env.observation_space.contains(pixels)
            assert (reward, terminated, truncated) == (
                expected_reward,
                expected_term,
                expected_trunc,
            )
            np.testing.assert_allclose(
                env.unwrapped.state, reference.unwrapped.state
            )
    finally:
        env.close()
        reference.close()


def test_cartpole_render_keeps_image_observations(monkeypatch):
    from dqn.train_nature import make_env

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    env = make_env("CartPole-v1", render_mode="human")
    try:
        pixels, _ = env.reset(seed=0)
        assert env.render_mode == "human"
        assert env.observation_space.contains(pixels)
        pixels, *_ = env.step(0)
        assert pixels.shape == (84, 84)
    finally:
        env.close()


def test_cartpole_model_input_matches_reference_pipeline(monkeypatch):
    import gymnasium as gym
    from gymnasium.spaces import Box
    from gymnasium.wrappers import (
        AddRenderObservation,
        FrameStackObservation,
        GrayscaleObservation,
        ResizeObservation,
        TransformObservation,
    )

    from dqn.train_nature import make_env, make_history

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    # Use dqn/cart_pole.py's wrappers with Nature's grayscale-first order
    # and zero padding, rather than FrameStackObservation's reset default.
    reference = gym.make("CartPole-v1", render_mode="rgb_array")
    reference = AddRenderObservation(reference, render_only=True)
    reference = GrayscaleObservation(reference, keep_dim=False)
    reference = ResizeObservation(reference, (84, 84))
    reference = FrameStackObservation(
        reference, stack_size=4, padding_type="zero"
    )
    reference = TransformObservation(
        reference,
        func=lambda obs: obs.astype(np.float32) / 255.0,
        observation_space=Box(0.0, 1.0, (4, 84, 84), dtype=np.float32),
    )
    env = make_env("CartPole-v1")
    replay = ReplayMemory(4)
    try:
        for seed in (0, 12):
            expected, _ = reference.reset(seed=seed)
            observation, _ = env.reset(seed=seed)
            replay.start_episode(observation)
            history = make_history(observation)
            np.testing.assert_array_equal(
                replay.state.astype(np.float32) / 255, expected
            )
            np.testing.assert_array_equal(np.stack(history), replay.state)
            for action in (0, 1, 0, 1):
                expected, *_ = reference.step(action)
                observation, reward, terminal, _, _ = env.step(action)
                replay.append(action, reward, observation, terminal)
                history.append(observation)
                np.testing.assert_array_equal(
                    replay.state.astype(np.float32) / 255, expected
                )
                np.testing.assert_array_equal(np.stack(history), replay.state)
    finally:
        reference.close()
        env.close()
