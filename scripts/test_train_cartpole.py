import numpy as np
import torch

from models.cnn_modular_rnn import CNNModularRNN
from models.modular_rnn import ModularRNN
from scripts.train_cartpole import build_td_objective


def _toy_episode(obs, actions, rewards, terminated):
    return (obs, actions, rewards, terminated)


def test_td_objective_gives_nonzero_gradient_to_recurrent_weights():
    # Regression test for a bug where feeding each timestep as an independent length-1
    # sequence (zero initial hidden state) made the input/intermediate modules
    # unreachable from the output module within one step, so weight_ih/weight_hh got
    # exactly zero gradient no matter what the objective was. build_td_objective must
    # replay each episode's full state sequence in one recurrent forward pass instead.
    #
    # Episodes need >=3 timesteps: the module topology (input<->intermediate<->output,
    # no input<->output link) means the output module only receives input-derived
    # information after two recurrent hops, so a taken action's Q-value only carries
    # gradient into weight_ih/weight_hh once its timestep index is >= 2.
    # near_module_sparsity is explicit (not the 0.1 default) because round(0.1 * 4) == 0
    # for a 4-units-per-module test model -- the default would leave modules with zero
    # cross-module connections at this tiny size and make the test meaningless.
    torch.manual_seed(0)
    model = ModularRNN(input_size=4, hidden_size=12, output_size=2, output_mode="all", near_module_sparsity=0.5)

    def make_obs(base):
        return [[base + 0.01 * t, base * 0.5, -base + 0.02 * t, base * 0.2] for t in range(6)]

    episodes = [
        _toy_episode(
            obs=make_obs(0.1),
            actions=[0, 1, 0, 1, 0],
            rewards=[1.0, 1.0, 1.0, 1.0, 1.0],
            terminated=False,
        ),
        _toy_episode(
            obs=make_obs(-0.2),
            actions=[1, 0, 1, 0, 1],
            rewards=[1.0, 1.0, 1.0, 1.0, 1.0],
            terminated=True,
        ),
    ]

    objective_fn = build_td_objective(
        episodes, device=torch.device("cpu"), online_model=model, target_model=model, gamma=0.99
    )
    loss, z = objective_fn(model)
    assert z.requires_grad

    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, allow_unused=True)

    grad_by_name = dict(zip((n for n, _ in model.named_parameters() if _.requires_grad), grads))
    ih_grad = grad_by_name["cell.weight_ih"]
    hh_grad = grad_by_name["cell.weight_hh"]
    assert ih_grad is not None and ih_grad.norm().item() > 0.0
    assert hh_grad is not None and hh_grad.norm().item() > 0.0


def test_td_objective_output_depends_on_trajectory():
    # Same >=3-timestep reasoning as above: only a Q-value taken at t>=2 can possibly
    # depend on the input, so the episode here has 4 steps and takes the action at t=3.
    torch.manual_seed(0)
    model = ModularRNN(input_size=4, hidden_size=12, output_size=2, output_mode="all", near_module_sparsity=0.5)

    def make_episode(scale):
        obs = [[scale + 0.01 * t, scale * 0.5, -scale + 0.02 * t, scale * 0.2] for t in range(4)]
        return _toy_episode(obs=obs, actions=[0, 1, 0], rewards=[1.0, 1.0, 1.0], terminated=False)

    z_values = []
    for scale in (0.1, 0.5, -0.3, 1.0, -1.0):
        objective_fn = build_td_objective(
            [make_episode(scale)], device=torch.device("cpu"), online_model=model, target_model=model
        )
        _, z = objective_fn(model)
        z_values.append(z.detach().clone())

    for i in range(1, len(z_values)):
        assert not torch.allclose(z_values[0], z_values[i]), "output must vary with input trajectory"


def test_td_objective_gives_nonzero_gradient_to_cnn_and_recurrent_weights():
    # Same recurrent-context reasoning as the feature-input test above, but for the
    # CNN front-end: gradient must flow through the per-frame conv encoder as well as
    # into the modular RNN's recurrent weights.
    torch.manual_seed(0)
    model = CNNModularRNN(
        in_channels=1, hidden_size=12, output_size=2, cnn_feature_dim=8, near_module_sparsity=0.5, image_size=16
    )

    def make_frames(base, steps=6):
        return [np.full((1, 16, 16), base + 0.01 * t, dtype=np.float32) for t in range(steps)]

    episodes = [
        _toy_episode(
            obs=make_frames(0.1),
            actions=[0, 1, 0, 1, 0],
            rewards=[1.0, 1.0, 1.0, 1.0, 1.0],
            terminated=False,
        ),
        _toy_episode(
            obs=make_frames(-0.2),
            actions=[1, 0, 1, 0, 1],
            rewards=[1.0, 1.0, 1.0, 1.0, 1.0],
            terminated=True,
        ),
    ]

    objective_fn = build_td_objective(
        episodes, device=torch.device("cpu"), online_model=model, target_model=model, gamma=0.99
    )
    loss, z = objective_fn(model)
    assert z.requires_grad

    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    grad_by_name = dict(zip((n for n, _ in model.named_parameters() if _.requires_grad), grads))

    conv_grad = grad_by_name["encoder.conv.0.weight"]
    ih_grad = grad_by_name["rnn.cell.weight_ih"]
    hh_grad = grad_by_name["rnn.cell.weight_hh"]
    assert conv_grad is not None and conv_grad.norm().item() > 0.0
    assert ih_grad is not None and ih_grad.norm().item() > 0.0
    assert hh_grad is not None and hh_grad.norm().item() > 0.0
