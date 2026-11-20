"""Check the compressed representation and the gradient-generation boundary."""

import torch

from research.gradient_field.experiment import decode_tensor, encode_tensor, potential


def test_quantization_round_trip_and_packing():
    torch.manual_seed(7)
    value = torch.randn(33, 17) * 0.1
    for bits in [4, 8]:
        code = encode_tensor(value, bits)
        recovered = decode_tensor(code)
        assert recovered.shape == value.shape
        assert torch.isfinite(recovered).all()
        assert torch.max(torch.abs(recovered - value)) <= code["scale"].float().max() * 0.51
        padded = ((value.numel() + 127) // 128) * 128
        assert code["values"].numel() == padded * bits // 8


def test_serialized_code_contains_every_decoding_input(tmp_path):
    value = torch.linspace(-0.25, 0.25, 300).reshape(20, 15)
    code = encode_tensor(value, 4)
    path = tmp_path / "code.pt"
    torch.save(code, path)
    recovered = decode_tensor(torch.load(path, weights_only=True))
    assert torch.equal(recovered, decode_tensor(code))


def test_zero_blocks_have_finite_positive_scales():
    for bits in [4, 8]:
        code = encode_tensor(torch.zeros(257), bits)
        assert (code["scale"] > 0).all()
        assert torch.equal(decode_tensor(code), torch.zeros(257))


def test_potential_reacts_to_current_student():
    model = torch.nn.Linear(3, 2, bias=False)
    target = {"weight": torch.zeros_like(model.weight)}
    potential(model, target, "quadratic").backward()
    old = model.weight.grad.clone()
    with torch.no_grad():
        model.weight.add_(0.25)
    model.zero_grad(set_to_none=True)
    potential(model, target, "quadratic").backward()
    assert torch.allclose(model.weight.grad - old, torch.full_like(old, 0.25))
    assert not target["weight"].requires_grad


def test_huber_gradient_saturation_and_zero_target():
    model = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.5)
    target = {"weight": torch.zeros_like(model.weight)}
    potential(model, target, "huber").backward()
    assert torch.allclose(model.weight.grad, torch.full_like(model.weight, 0.01))
    model.zero_grad(set_to_none=True)
    target["weight"] = model.weight.detach().clone()
    potential(model, target, "quadratic").backward()
    assert torch.count_nonzero(model.weight.grad) == 0
