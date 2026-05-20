import torch

from elasticneuralnetwork import (
    AdaptiveConv2d,
    HiddenActivation,
    ReversibleLinear,
    ReversibleNetwork,
    ReversibleNetworkConfig,
)


def test_reversible_linear_forward_matches_torch_linear_when_all_active():
    torch.manual_seed(0)
    rl = ReversibleLinear(in_features=8, out_features=4)
    x = torch.randn(3, 8)
    out = rl.forward(x)
    expected = torch.nn.functional.linear(x, rl.weight, rl.bias)
    assert torch.allclose(out, expected)


def test_reversible_linear_prune_then_add_round_trip():
    torch.manual_seed(0)
    rl = ReversibleLinear(8, 4)
    x = torch.randn(3, 8)
    full = rl.forward(x).clone()
    rl.prune_nodes([1, 2])
    pruned = rl.forward(x)
    assert torch.allclose(pruned[:, 0], full[:, 0])
    assert torch.allclose(pruned[:, 3], full[:, 3])
    assert torch.allclose(pruned[:, 1], torch.zeros_like(pruned[:, 1]))
    reactivated = rl.add_nodes(2)
    assert reactivated == 2
    after = rl.forward(x)
    assert torch.allclose(after, full)


def test_reversible_linear_topology_version_monotonic():
    rl = ReversibleLinear(4, 4)
    v0 = rl.topology_version()
    rl.prune_nodes([0])
    v1 = rl.topology_version()
    rl.add_nodes(1)
    v2 = rl.topology_version()
    rl.compact()
    v3 = rl.topology_version()
    assert v0 < v1 < v2 < v3


def test_reversible_linear_compact_drops_inactive_rows():
    rl = ReversibleLinear(4, 6)
    rl.prune_nodes([1, 3, 5])
    dropped = rl.compact()
    assert dropped == 3
    assert rl.out_features() == 3


def test_reversible_network_forward_shape():
    cfg = ReversibleNetworkConfig()
    cfg.input_features = 8
    cfg.output_features = 3
    cfg.hidden_sizes = [6, 4]
    cfg.hidden_activation = HiddenActivation.ReLU
    net = ReversibleNetwork(cfg)
    x = torch.randn(5, 8)
    out = net.forward(x)
    assert out.shape == (5, 3)
    assert net.num_layers() == 3


def test_reversible_network_compact_reduces_topology():
    cfg = ReversibleNetworkConfig()
    cfg.input_features = 6
    cfg.output_features = 2
    cfg.hidden_sizes = [4, 4]
    net = ReversibleNetwork(cfg)
    net.layer(0).prune_nodes([0, 1])
    dropped = net.compact()
    assert dropped >= 2
    x = torch.randn(2, 6)
    out = net.forward(x)
    assert out.shape == (2, 2)


def test_adaptive_conv2d_forward_shape():
    conv = AdaptiveConv2d(in_channels=3, out_channels=8,
                            candidate_kernel_sizes=[3, 5, 7])
    x = torch.randn(2, 3, 16, 16)
    y = conv.forward(x)
    assert y.shape == (2, 8, 16, 16)


def test_adaptive_conv2d_prune_smallest_keeps_at_least_one():
    conv = AdaptiveConv2d(3, 4, [3, 5, 7])
    conv.prune_smallest_candidate()
    conv.prune_smallest_candidate()
    conv.prune_smallest_candidate()
    assert conv.candidate_mask.sum().item() >= 1


def test_adaptive_conv2d_topology_version_increments():
    conv = AdaptiveConv2d(3, 4, [3, 5])
    v0 = conv.topology_version()
    conv.prune_smallest_candidate()
    assert conv.topology_version() > v0
