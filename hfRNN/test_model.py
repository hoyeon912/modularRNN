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
    cell = ModularRNNCell(input_size=4, hidden_size=9, near_module_sparsity=0.1)
    lo, hi = _module_bounds(9)
    assert cell.ih_mask.shape == (9, 4)
    assert torch.all(cell.ih_mask[:lo, :] == 1.0)
    assert torch.all(cell.ih_mask[lo:, :] == 0.0)


def test_hh_mask_same_module_blocks_are_dense():
    cell = ModularRNNCell(input_size=4, hidden_size=9, near_module_sparsity=0.1)
    lo, hi = _module_bounds(9)
    assert torch.all(cell.hh_mask[:lo, :lo] == 1.0)
    assert torch.all(cell.hh_mask[lo:hi, lo:hi] == 1.0)
    assert torch.all(cell.hh_mask[hi:, hi:] == 1.0)


def test_hh_mask_input_output_blocks_are_zero():
    cell = ModularRNNCell(input_size=4, hidden_size=9, near_module_sparsity=0.1)
    lo, hi = _module_bounds(9)
    assert torch.all(cell.hh_mask[:lo, hi:] == 0.0)
    assert torch.all(cell.hh_mask[hi:, :lo] == 0.0)


def test_hh_mask_near_module_indegree_is_exact():
    torch.manual_seed(0)
    cell = ModularRNNCell(input_size=4, hidden_size=300, near_module_sparsity=0.1)
    lo, hi = _module_bounds(300)
    expected = round(0.1 * lo)
    row_counts = cell.hh_mask[:lo, lo:hi].sum(dim=1)
    assert torch.all(row_counts == expected)


def test_forbidden_hh_blocks_never_contribute_regardless_of_weight():
    cell = ModularRNNCell(input_size=4, hidden_size=9, near_module_sparsity=0.1)
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


def test_raw_weight_data_is_zero_outside_mask_after_init():
    cell = ModularRNNCell(input_size=4, hidden_size=9, near_module_sparsity=0.1)
    assert torch.all(cell.weight_ih.data[cell.ih_mask == 0.0] == 0.0)
    assert torch.all(cell.weight_hh.data[cell.hh_mask == 0.0] == 0.0)

    model = ModularBidirectionalRNN(input_size=4, hidden_size=9, output_size=2)
    assert torch.all(model.output_proj.weight.data[:, model.output_mask == 0.0] == 0.0)


def test_masked_entries_stay_zero_after_optimizer_step():
    model = ModularBidirectionalRNN(input_size=4, hidden_size=9, output_size=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    x = torch.randn(3, 5, 4)
    out = model(x)
    out.sum().backward()
    optimizer.step()

    fwd = model.fwd_cell
    assert torch.all(fwd.weight_ih.data[fwd.ih_mask == 0.0] == 0.0)
    assert torch.all(fwd.weight_hh.data[fwd.hh_mask == 0.0] == 0.0)
    assert torch.all(model.output_proj.weight.data[:, model.output_mask == 0.0] == 0.0)


def test_hh_weight_scale_matches_gain_over_sqrt_c():
    torch.manual_seed(0)
    hidden_size = 300
    gain = 1.4
    cell = ModularRNNCell(input_size=4, hidden_size=hidden_size, near_module_sparsity=0.1, recurrent_gain=gain)
    lo, hi = _module_bounds(hidden_size)

    diag_block = cell.weight_hh.data[:lo, :lo]
    expected_std = gain / (lo**0.5)
    assert abs(diag_block.std().item() - expected_std) < 0.1 * expected_std


def test_ih_weight_has_no_sqrt_input_size_scaling():
    torch.manual_seed(0)
    hidden_size, input_size, gain = 300, 50, 1.0
    cell = ModularRNNCell(input_size=input_size, hidden_size=hidden_size, near_module_sparsity=0.1, input_gain=gain)
    lo, _ = _module_bounds(hidden_size)

    block = cell.weight_ih.data[:lo, :]
    assert abs(block.std().item() - gain) < 0.1 * gain


def test_no_input_proj_attribute():
    model = ModularBidirectionalRNN(input_size=4, hidden_size=9, output_size=2)
    assert not hasattr(model, "input_proj")
