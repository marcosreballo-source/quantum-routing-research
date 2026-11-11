import pytest

from DRL_Lei.training import TRAINERS, REQUIRED_PARAMS


class TestTrainerRegistry:
    def test_ppo_registered(self):
        assert 'ppo' in TRAINERS
        assert callable(TRAINERS['ppo'])

    def test_rollout_registered(self):
        assert 'rollout' in TRAINERS
        assert callable(TRAINERS['rollout'])

    def test_required_params_ppo(self):
        assert 'ppo' in REQUIRED_PARAMS
        assert 'lr' in REQUIRED_PARAMS['ppo']
        assert 'batch_size' in REQUIRED_PARAMS['ppo']
        assert 'ppo_epoch' in REQUIRED_PARAMS['ppo']

    def test_required_params_rollout(self):
        assert 'rollout' in REQUIRED_PARAMS
        assert 'lr' in REQUIRED_PARAMS['rollout']
        assert 'lr_decay' in REQUIRED_PARAMS['rollout']


class TestAdvNormalize:
    def test_ppo_adv_normalize(self):
        import torch
        from DRL_Lei.VRP_PPO_Model import Agentppo

        agent = Agentppo(
            steps=22, greedy=True, lr=1e-4,
            input_node_dim=3, hidden_node_dim=64,
            input_edge_dim=1, hidden_edge_dim=16,
            conv_laysers=1, n_nodes=11,
        )
        adv = torch.randn(16)
        result = agent.adv_normalize(adv)
        assert abs(result.mean().item()) < 1e-5
        assert abs(result.std().item() - 1.0) < 0.2

    def test_rollout_adv_normalize(self):
        import torch
        from DRL_Lei.VRP_Rollout_train import adv_normalize

        adv = torch.randn(16)
        result = adv_normalize(adv)
        assert abs(result.mean().item()) < 1e-5


class TestComputeTemperature:
    def test_constant(self):
        from DRL_Lei.VRP_PPO_Model import Agentppo

        agent = Agentppo(
            steps=22, greedy=True, lr=1e-4,
            input_node_dim=3, hidden_node_dim=64,
            input_edge_dim=1, hidden_edge_dim=16,
            conv_laysers=1, n_nodes=11,
            temperature_config={'initial': 2.5, 'final': 2.5, 'schedule': 'constant', 'decay_rate': 1.0},
        )
        assert agent.compute_temperature(0, 100) == 2.5
        assert agent.compute_temperature(50, 100) == 2.5

    def test_linear(self):
        from DRL_Lei.VRP_PPO_Model import Agentppo

        agent = Agentppo(
            steps=22, greedy=True, lr=1e-4,
            input_node_dim=3, hidden_node_dim=64,
            input_edge_dim=1, hidden_edge_dim=16,
            conv_laysers=1, n_nodes=11,
            temperature_config={'initial': 2.5, 'final': 1.0, 'schedule': 'linear', 'decay_rate': 1.0},
        )
        t_start = agent.compute_temperature(0, 100)
        t_end = agent.compute_temperature(99, 100)
        assert abs(t_start - 2.5) < 1e-6
        assert abs(t_end - 1.0) < 0.1

    def test_exponential(self):
        from DRL_Lei.VRP_PPO_Model import Agentppo

        agent = Agentppo(
            steps=22, greedy=True, lr=1e-4,
            input_node_dim=3, hidden_node_dim=64,
            input_edge_dim=1, hidden_edge_dim=16,
            conv_laysers=1, n_nodes=11,
            temperature_config={'initial': 2.5, 'final': 1.0, 'schedule': 'exponential', 'decay_rate': 0.99},
        )
        t0 = agent.compute_temperature(0, 100)
        t10 = agent.compute_temperature(10, 100)
        assert t0 == 2.5
        assert t10 < t0
