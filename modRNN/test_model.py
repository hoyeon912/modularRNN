import torch

from model import ModularBidirectionalRNN, ModularRNNCell, get_device


def _module_bounds(hidden_size: int) -> tuple[int, int]:
    third = hidden_size // 3
    return third, 2 * third


def test_output_shape_last_mode():
    model = ModularBidirectionalRNN(input_size=4, hidden_size=9, output_size=2, output_mode="last")
    x = torch.randn(3, 10, 4)
    out = model(x)
    assert out.shape == (3, 2)


def test_output_shape_all_mode():
    model = ModularBidirectionalRNN(input_size=4, hidden_size=9, output_size=2, output_mode="all")
    x = torch.randn(3, 10, 4)
    out = model(x)
    assert out.shape == (3, 10, 2)


def test_default_output_mode_is_last():
    model = ModularBidirectionalRNN(input_size=4, hidden_size=9, output_size=2)
    x = torch.randn(2, 5, 4)
    out = model(x)
    assert out.shape == (2, 2)


def test_invalid_output_mode_raises():
    try:
        ModularBidirectionalRNN(input_size=4, hidden_size=9, output_size=2, output_mode="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_hidden_size_not_divisible_by_3_raises():
    try:
        ModularBidirectionalRNN(input_size=4, hidden_size=10, output_size=2)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_get_device_returns_torch_device():
    device = get_device()
    assert isinstance(device, torch.device)


def test_ih_mask_restricts_external_input_to_input_module():
    cell = ModularRNNCell(hidden_size=9, near_module_sparsity=0.1)
    lo, hi = _module_bounds(9)
    assert torch.all(cell.ih_mask[:lo, :] == 1.0)
    assert torch.all(cell.ih_mask[lo:, :] == 0.0)


def test_hh_mask_same_module_blocks_are_dense():
    cell = ModularRNNCell(hidden_size=9, near_module_sparsity=0.1)
    lo, hi = _module_bounds(9)
    assert torch.all(cell.hh_mask[:lo, :lo] == 1.0)
    assert torch.all(cell.hh_mask[lo:hi, lo:hi] == 1.0)
    assert torch.all(cell.hh_mask[hi:, hi:] == 1.0)


def test_hh_mask_input_output_blocks_are_zero():
    cell = ModularRNNCell(hidden_size=9, near_module_sparsity=0.1)
    lo, hi = _module_bounds(9)
    assert torch.all(cell.hh_mask[:lo, hi:] == 0.0)
    assert torch.all(cell.hh_mask[hi:, :lo] == 0.0)


def test_hh_mask_near_module_density_matches_sparsity():
    torch.manual_seed(0)
    cell = ModularRNNCell(hidden_size=300, near_module_sparsity=0.1)
    lo, hi = _module_bounds(300)
    density = cell.hh_mask[:lo, lo:hi].mean().item()
    assert 0.05 < density < 0.15


def test_forbidden_hh_blocks_never_contribute_regardless_of_weight():
    cell = ModularRNNCell(hidden_size=9, near_module_sparsity=0.1)
    lo, hi = _module_bounds(9)
    masked = cell.weight_hh * cell.hh_mask
    assert torch.all(masked[:lo, hi:] == 0.0)
    assert torch.all(masked[hi:, :lo] == 0.0)


def test_output_mask_restricts_to_output_module_both_directions():
    model = ModularBidirectionalRNN(input_size=4, hidden_size=9, output_size=2)
    lo, hi = _module_bounds(9)
    mask = model.output_mask
    assert torch.all(mask[hi:9] == 1.0)
    assert torch.all(mask[9 + hi : 18] == 1.0)
    assert torch.all(mask[:hi] == 0.0)
    assert torch.all(mask[9 : 9 + hi] == 0.0)
