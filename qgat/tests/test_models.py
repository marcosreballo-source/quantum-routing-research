import pytest
import torch
from pathlib import Path

from DRL_Lei.VRP_PPO_Model import (
    GatConv, Encoder, Attention1, ProbAttention,
    Decoder1, Model, Critic, Actor_critic, Agentppo, Memory,
)
from DRL_Lei.PPO_train import BlockTimer, _NullTimer
from DRL_Lei.utils.config import load_config
from DRL_Lei.creat_vrp import creat_data, reward1


# ---------------------------------------------------------------------------
# Minimal training helper (1 batch, 1 epoch, seed 42, light_test config)
# ---------------------------------------------------------------------------
def train_minimal_model():
    """Train a Model+Critic for 1 PPO update so forward passes are stable."""
    cfg = load_config(
        str(Path(__file__).resolve().parent.parent / "configs" / "base.yaml"),
        str(Path(__file__).resolve().parent.parent / "configs" / "light_test.yaml"),
    )

    n_nodes = cfg.problem.n_nodes + 1  # +1 for depot, same as PPO_train
    device = torch.device("cpu")
    batch_size = 2

    agent = Agentppo(
        steps=cfg.training.timestep,
        greedy=False,
        lr=cfg.training.lr,
        input_node_dim=3,
        hidden_node_dim=cfg.model.hidden_node_dim,
        input_edge_dim=1,
        hidden_edge_dim=cfg.model.hidden_edge_dim,
        epoch=1,
        batch_size=batch_size,
        conv_laysers=cfg.model.conv_layers,
        n_nodes=n_nodes,
        seeds_cfg={"data_seed": 42, "model_seed": 42, "random_seed": 42},
        entropy_value=cfg.training.entropy_value,
        eps_clip=cfg.training.eps_clip,
        device=device,
    )

    dl = creat_data(n_nodes, num_samples=batch_size, batch_size=batch_size)
    batch = next(iter(dl))
    batch = batch.to(device)

    # Reshape exactly like PPO_train.run_train
    x = batch.x.view(batch_size, n_nodes, 2)
    attr = batch.edge_attr.view(batch_size, n_nodes * n_nodes, 1)
    capcity = batch.capcity.view(batch_size, 1)
    demand = batch.demand.view(batch_size, n_nodes, 1)

    # Rollout with old policy (sampling mode, same as training)
    agent.old_polic.to(device)
    actions, log_p = agent.old_polic.act(
        batch, 0, cfg.training.timestep, batch_size, device, False, False
    )
    rewards = reward1(batch.x, actions, n_nodes)

    actions = actions.detach()
    log_p = log_p.detach()
    rewards = rewards.detach()

    # Populate memory per-sample (extend + unbind, same as PPO_train)
    memory = Memory()
    memory.input_x.extend(list(torch.unbind(x, dim=0)))
    memory.input_attr.extend(list(torch.unbind(attr, dim=0)))
    memory.actions.extend(list(torch.unbind(actions, dim=0)))
    memory.log_probs.extend(list(torch.unbind(log_p, dim=0)))
    memory.rewards.extend(list(torch.unbind(rewards, dim=0)))
    memory.capcity.extend(list(torch.unbind(capcity, dim=0)))
    memory.demand.extend(list(torch.unbind(demand, dim=0)))

    # One PPO update
    agent.update(memory, epoch=0)

    return agent.policy, cfg, n_nodes


@pytest.fixture(scope="module")
def trained_model():
    """Module-scoped: train once, reuse across all tests in this file."""
    policy, cfg, n_nodes = train_minimal_model()
    return policy, cfg, n_nodes


# ---------------------------------------------------------------------------
# GatConv
# ---------------------------------------------------------------------------
class TestGatConv:
    def test_classical_forward(self):
        gc = GatConv(64, 64, 16)
        x = torch.randn(16, 64)
        edge_attr = torch.randn(64, 16)
        edge_index = torch.randint(0, 16, (2, 64))
        out = gc(x, edge_index, edge_attr)
        assert out.shape == (16, 64)

    def test_output_no_nan(self):
        gc = GatConv(32, 32, 8)
        x = torch.randn(8, 32)
        ei = torch.randint(0, 8, (2, 32))
        ea = torch.randn(32, 8)
        out = gc(x, ei, ea)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class TestEncoder:
    def test_encoder_instantiation(self):
        enc = Encoder(3, 64, 1, 16, conv_layers=2)
        assert enc is not None
        assert hasattr(enc, "fc_node")
        assert hasattr(enc, "fc_edge")


# ---------------------------------------------------------------------------
# Attention / ProbAttention
# ---------------------------------------------------------------------------
class TestAttention:
    def test_prob_attention_temperature(self):
        pa = ProbAttention(8, 64, 64)
        pa.set_temperature(1.5)
        assert pa.temperature == 1.5

    def test_attention1_forward(self):
        attn = Attention1(4, 1, 64, 64)
        state_t = torch.randn(2, 1, 64)
        context = torch.randn(2, 11, 64)
        mask = torch.zeros(2, 11, dtype=torch.bool)
        out = attn(state_t, context, mask)
        assert out.shape == (2, 64)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
class TestDecoder:
    def test_decoder_instantiation(self):
        dec = Decoder1(64, 64)
        assert dec is not None


# ---------------------------------------------------------------------------
# Model (PPO actor) — forward test uses trained weights
# ---------------------------------------------------------------------------
class TestModel:
    def test_classical_instantiation(self):
        model = Model(3, 128, 1, 16, 4, quantum_cfg={}, dropout_cfg={})
        assert model is not None

    def test_forward_with_trained_model(self, trained_model):
        policy, cfg, n_nodes = trained_model
        model = policy.actor  # trained Model instance

        dl = creat_data(n_nodes, num_samples=2, batch_size=2)
        batch = next(iter(dl))
        device = torch.device("cpu")

        model.eval()
        with torch.no_grad():
            actions, log_p, _, _, x = model(
                batch, 0, cfg.training.timestep, 2, device, True, False, temperature=2.5
            )

        assert actions.shape[0] == 2
        assert log_p.shape[0] == 2
        assert x.shape == (2, n_nodes, cfg.model.hidden_node_dim)


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------
class TestCritic:
    def test_classical(self):
        c = Critic(64)
        x = torch.randn(2, 11, 64)
        val = c(x)
        assert val.shape == (2,)

    def test_apply_layer_classical(self):
        c = Critic(64)
        x_conv = torch.randn(2, 64, 11)
        out = c._apply_layer(c.fc1, x_conv, False, 2, 11, 64)
        assert out.shape[0] == 2


# ---------------------------------------------------------------------------
# Actor_critic — act test uses trained weights
# ---------------------------------------------------------------------------
class TestActorCritic:
    def test_classical_instantiation(self):
        ac = Actor_critic(3, 128, 1, 16, 4, quantum_cfg={}, dropout_cfg={})
        assert hasattr(ac, "actor")
        assert hasattr(ac, "critic")

    def test_act_with_trained_model(self, trained_model):
        policy, cfg, n_nodes = trained_model  # policy is the trained Actor_critic

        dl = creat_data(n_nodes, num_samples=2, batch_size=2)
        batch = next(iter(dl))
        device = torch.device("cpu")

        policy.eval()
        with torch.no_grad():
            actions, log_p = policy.act(
                batch, 0, cfg.training.timestep, 2, device, True, False, temperature=2.5
            )

        assert actions.shape[0] == 2
        assert log_p.shape[0] == 2


# ---------------------------------------------------------------------------
# BlockTimer / _NullTimer
# ---------------------------------------------------------------------------
class TestTimers:
    def test_block_timer_measures(self):
        bt = BlockTimer()
        with bt.measure("test_block"):
            _ = sum(range(100))
        assert "test_block" in bt.totals
        assert bt.totals["test_block"] >= 0

    def test_block_timer_reset(self):
        bt = BlockTimer()
        with bt.measure("x"):
            pass
        bt.reset()
        assert len(bt.totals) == 0

    def test_null_timer_noop(self):
        nt = _NullTimer()
        with nt.measure("anything"):
            pass
        assert len(nt.totals) == 0

    def test_null_timer_reset(self):
        nt = _NullTimer()
        nt.reset()
