import importlib.util
import os

import torch

from model import BidirectionalRNN, get_device

_rnn_model_path = os.path.join(os.path.dirname(__file__), "..", "RNN", "model.py")
_spec = importlib.util.spec_from_file_location("rnn_model_for_parity_test", _rnn_model_path)
_rnn_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rnn_model)
SimpleRNN = _rnn_model.SimpleRNN


def _copy_weights(simple: SimpleRNN, manual: BidirectionalRNN) -> None:
    manual.input_proj.weight.data.copy_(simple.input_proj.weight.data)
    manual.input_proj.bias.data.copy_(simple.input_proj.bias.data)
    manual.output_proj.weight.data.copy_(simple.output_proj.weight.data)
    manual.output_proj.bias.data.copy_(simple.output_proj.bias.data)

    manual.fwd_cell.weight_ih.data.copy_(simple.rnn.weight_ih_l0.data)
    manual.fwd_cell.weight_hh.data.copy_(simple.rnn.weight_hh_l0.data)
    manual.fwd_cell.bias_ih.data.copy_(simple.rnn.bias_ih_l0.data)
    manual.fwd_cell.bias_hh.data.copy_(simple.rnn.bias_hh_l0.data)

    manual.bwd_cell.weight_ih.data.copy_(simple.rnn.weight_ih_l0_reverse.data)
    manual.bwd_cell.weight_hh.data.copy_(simple.rnn.weight_hh_l0_reverse.data)
    manual.bwd_cell.bias_ih.data.copy_(simple.rnn.bias_ih_l0_reverse.data)
    manual.bwd_cell.bias_hh.data.copy_(simple.rnn.bias_hh_l0_reverse.data)


def test_output_shape_last_mode():
    model = BidirectionalRNN(input_size=4, hidden_size=8, output_size=2, output_mode="last")
    x = torch.randn(3, 10, 4)
    out = model(x)
    assert out.shape == (3, 2)


def test_output_shape_all_mode():
    model = BidirectionalRNN(input_size=4, hidden_size=8, output_size=2, output_mode="all")
    x = torch.randn(3, 10, 4)
    out = model(x)
    assert out.shape == (3, 10, 2)


def test_default_output_mode_is_last():
    model = BidirectionalRNN(input_size=4, hidden_size=8, output_size=2)
    x = torch.randn(2, 5, 4)
    out = model(x)
    assert out.shape == (2, 2)


def test_invalid_output_mode_raises():
    try:
        BidirectionalRNN(input_size=4, hidden_size=8, output_size=2, output_mode="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_get_device_returns_torch_device():
    device = get_device()
    assert isinstance(device, torch.device)


def test_parity_with_nn_rnn_last_mode():
    torch.manual_seed(0)
    simple = SimpleRNN(input_size=4, hidden_size=8, output_size=3, output_mode="last")
    manual = BidirectionalRNN(input_size=4, hidden_size=8, output_size=3, output_mode="last")
    _copy_weights(simple, manual)

    x = torch.randn(5, 6, 4)
    simple.eval()
    manual.eval()
    with torch.no_grad():
        out_simple = simple(x)
        out_manual = manual(x)

    assert torch.allclose(out_simple, out_manual, atol=1e-5)


def test_parity_with_nn_rnn_all_mode():
    torch.manual_seed(1)
    simple = SimpleRNN(input_size=4, hidden_size=8, output_size=3, output_mode="all")
    manual = BidirectionalRNN(input_size=4, hidden_size=8, output_size=3, output_mode="all")
    _copy_weights(simple, manual)

    x = torch.randn(5, 6, 4)
    simple.eval()
    manual.eval()
    with torch.no_grad():
        out_simple = simple(x)
        out_manual = manual(x)

    assert torch.allclose(out_simple, out_manual, atol=1e-5)
