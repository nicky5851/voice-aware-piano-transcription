import torch

from vapt.models.grad_scale import grad_scale


def test_forward_is_identity():
    h = torch.randn(4, 5, requires_grad=True)
    for lam in (0.0, 0.3, 0.5, 1.0):
        out = grad_scale(h, lam)
        assert torch.equal(out, h)


def test_lam_zero_gives_exact_zero_gradient_no_graph():
    h = torch.randn(4, 5, requires_grad=True)
    out = grad_scale(h, 0.0)
    assert out.grad_fn is None  # detached: no autograd node at all
    assert not out.requires_grad
    w = torch.randn(4, 5, requires_grad=True)  # a second leaf, so the loss still has a graph
    loss = (out * w).sum()
    loss.backward()
    assert h.grad is None  # nothing flowed back to h
    assert w.grad is not None


def test_lam_one_gives_full_gradient():
    h = torch.randn(4, 5, requires_grad=True)
    upstream = torch.randn(4, 5)
    out = grad_scale(h, 1.0)
    (out * upstream).sum().backward()
    assert torch.allclose(h.grad, upstream)


def test_intermediate_lam_scales_gradient():
    for lam in (0.25, 0.5, 0.75):
        h = torch.randn(4, 5, requires_grad=True)
        upstream = torch.randn(4, 5)
        out = grad_scale(h, lam)
        (out * upstream).sum().backward()
        assert torch.allclose(h.grad, lam * upstream, rtol=1e-6)
