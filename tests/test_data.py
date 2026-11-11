import pytest
import torch
import numpy as np

from DRL_Lei.creat_vrp import creat_instance, creat_data, reward1


class TestCreatInstance:
    def test_returns_correct_types(self):
        datas, edges, demand, capacity = creat_instance(1, n_nodes=11, random_seed=42)
        assert isinstance(datas, np.ndarray)
        assert isinstance(edges, np.ndarray)
        assert isinstance(demand, np.ndarray)
        assert isinstance(capacity, float)

    def test_shapes(self):
        n = 11
        datas, edges, demand, capacity = creat_instance(1, n_nodes=n, random_seed=42)
        assert datas.shape == (n, 2)
        assert edges.shape == (n * n, 1)
        assert demand.shape == (n,)
        assert demand[0] == 0.0  # depot has no demand

    def test_deterministic(self):
        d1, e1, dem1, c1 = creat_instance(1, n_nodes=11, random_seed=123)
        d2, e2, dem2, c2 = creat_instance(1, n_nodes=11, random_seed=123)
        np.testing.assert_array_equal(d1, d2)
        np.testing.assert_array_equal(e1, e2)

    @pytest.mark.parametrize("n_nodes", [10, 20, 50, 100])
    def test_capacity_values(self, n_nodes):
        _, _, _, cap = creat_instance(1, n_nodes=n_nodes + 1, random_seed=42)
        expected = {10: 2.0, 20: 3.0, 50: 4.0, 100: 5.0}
        assert cap == expected[n_nodes]


class TestCreatData:
    def test_returns_dataloader(self):
        dl = creat_data(11, num_samples=4, batch_size=2)
        batch = next(iter(dl))
        assert hasattr(batch, "x")
        assert hasattr(batch, "edge_index")
        assert hasattr(batch, "edge_attr")
        assert hasattr(batch, "demand")
        assert hasattr(batch, "capcity")

    def test_batch_shapes(self):
        n_nodes = 11
        dl = creat_data(n_nodes, num_samples=4, batch_size=2)
        batch = next(iter(dl))
        assert batch.x.shape[1] == 2  # (batch*nodes, 2) coordinates


class TestReward1:
    def test_output_shape(self):
        n_nodes = 6
        batch_size = 2
        static = torch.rand(batch_size * n_nodes, 2)
        tour = torch.arange(n_nodes).unsqueeze(0).expand(batch_size, -1)
        result = reward1(static, tour, n_nodes)
        assert result.shape == (batch_size,)

    def test_positive_costs(self):
        n_nodes = 6
        batch_size = 2
        static = torch.rand(batch_size * n_nodes, 2)
        tour = torch.arange(n_nodes).unsqueeze(0).expand(batch_size, -1)
        result = reward1(static, tour, n_nodes)
        assert (result > 0).all()
