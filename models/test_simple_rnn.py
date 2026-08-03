import torch

from models.simple_rnn import SimpleRNN, get_device


def test_output_shape_last_mode():
    model = SimpleRNN(input_size=4, hidden_size=8, output_size=2, output_mode="last")
    x = torch.randn(3, 10, 4)  # batch=3, seq_len=10, input_size=4
    out = model(x)
    assert out.shape == (3, 2)


def test_output_shape_all_mode():
    model = SimpleRNN(input_size=4, hidden_size=8, output_size=2, output_mode="all")
    x = torch.randn(3, 10, 4)
    out = model(x)
    assert out.shape == (3, 10, 2)


def test_default_output_mode_is_last():
    model = SimpleRNN(input_size=4, hidden_size=8, output_size=2)
    x = torch.randn(2, 5, 4)
    out = model(x)
    assert out.shape == (2, 2)


def test_invalid_output_mode_raises():
    try:
        SimpleRNN(input_size=4, hidden_size=8, output_size=2, output_mode="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_get_device_returns_torch_device():
    device = get_device()
    assert isinstance(device, torch.device)


def test_forward_return_hidden_shape():
    model = SimpleRNN(input_size=4, hidden_size=8, output_size=2, output_mode="last")
    x = torch.randn(3, 5, 4)
    out, hidden = model(x, return_hidden=True)
    assert out.shape == (3, 2)
    assert hidden.shape == (3, 5, 16)


def test_forward_return_hidden_shape_all_mode():
    model = SimpleRNN(input_size=4, hidden_size=8, output_size=2, output_mode="all")
    x = torch.randn(3, 5, 4)
    out, hidden = model(x, return_hidden=True)
    assert out.shape == (3, 5, 2)
    assert hidden.shape == (3, 5, 16)


def test_forward_without_return_hidden_returns_tensor_only():
    model = SimpleRNN(input_size=4, hidden_size=8, output_size=2, output_mode="last")
    x = torch.randn(3, 5, 4)
    out = model(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 2)
