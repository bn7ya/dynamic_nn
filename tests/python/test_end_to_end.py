import torch

from elasticneuralnetwork import ElasticNetwork


def test_end_to_end_fit_predict_classification():
    torch.manual_seed(0)
    X = torch.randn(128, 16)
    y = torch.eye(4)[torch.randint(0, 4, (128,))]
    model = ElasticNetwork(input_shape=(16,), output_size=4, seed=42,
                            hidden_depth=1)
    result = model.fit(X, y)
    assert result.epochs_completed > 0
    assert len(result.cost_trajectory) == result.epochs_completed
    assert len(result.utilization_trajectory) == result.epochs_completed
    assert result.parameter_count_final > 0
    preds = model.predict(X[:4])
    assert preds.shape == (4, 4)


def test_end_to_end_topology_version_monotonic():
    torch.manual_seed(0)
    X = torch.randn(64, 8)
    y = torch.eye(3)[torch.randint(0, 3, (64,))]
    model = ElasticNetwork(input_shape=(8,), output_size=3, seed=0,
                            hidden_depth=1)
    v0 = model.topology_version()
    model.fit(X, y)
    v1 = model.topology_version()
    assert v1 >= v0


def test_statistics_available_after_fit():
    torch.manual_seed(0)
    X = torch.randn(100, 6)
    y = torch.eye(2)[torch.randint(0, 2, (100,))]
    model = ElasticNetwork(input_shape=(6,), output_size=2, seed=1,
                            hidden_depth=1)
    model.fit(X, y)
    s = model.statistics()
    assert s is not None
    assert s.num_samples == 100
    assert s.num_features == 6
