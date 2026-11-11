import torch
import torch.nn as nn
import mlflow
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.distributions.categorical import Categorical
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils.rnn import pad_sequence
import time
import logging
import numpy as np
import random as _random
from DRL_Lei.vrpUpdate import update_mask, update_state
from DRL_Lei.quantum_layers import create_quantum_layer, SwitchableLinear
from DRL_Lei.utils.model_utils import log_model_summary, count_model_parameters

# Shared encoder/attention modules (canonical source)
from DRL_Lei.shared_modules import GatConv, Encoder, Attention1, ProbAttention  # noqa: F401

INIT = True
max_grad_norm = 2

model_logger = logging.getLogger(__name__)


class Decoder1(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_decoder=0.0,
                 attn_dropout=0.0, dropout_attn_output=0.0, temperature=2.5,
                 tanh_clipping=10.0, quantum_decoder_cfg=None, pool_behavior="static"):
        super(Decoder1, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_decoder = dropout_decoder
        self.pool_behavior = pool_behavior
        self.prob = ProbAttention(8, input_dim, hidden_dim,
                                  attn_dropout=attn_dropout, dropout=dropout_attn_output,
                                  temperature=temperature, tanh_clipping=tanh_clipping)

        qd = quantum_decoder_cfg or {}
        use_q_fc = qd.get('fc', False)
        use_q_fc1 = qd.get('fc1', False)
        n_qubits = qd.get('n_qubits', 4)
        n_layers_q = qd.get('n_layers', 4)
        rotation = qd.get('rotation', 'RYRZ')
        topology = qd.get('topology', 'ring')
        backend = qd.get('backend', 'lightning.qubit')
        diff_method = qd.get('diff_method', 'best')

        self.fc = SwitchableLinear(hidden_dim + 1, hidden_dim, use_quantum=use_q_fc,
                                   n_qubits=n_qubits, n_layers=n_layers_q,
                                   rotation=rotation, topology=topology, use_bias=False,
                                   backend=backend, diff_method=diff_method)
        self.fc1 = SwitchableLinear(hidden_dim, hidden_dim, use_quantum=use_q_fc1,
                                    n_qubits=n_qubits, n_layers=n_layers_q,
                                    rotation=rotation, topology=topology, use_bias=False,
                                    backend=backend, diff_method=diff_method)

        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name and len(p.size()) >= 2:
                    nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self, encoder_inputs, pool, actions_old, capcity, demand, n_steps, batch_size,
                device, greedy=False, _action=False, temperature=None):

        mask1 = encoder_inputs.new_zeros((encoder_inputs.size(0), encoder_inputs.size(1)))
        mask = encoder_inputs.new_zeros((encoder_inputs.size(0), encoder_inputs.size(1)))
        dynamic_capcity = capcity.view(encoder_inputs.size(0), -1)
        demands = demand.view(encoder_inputs.size(0), encoder_inputs.size(1))
        index = torch.zeros(encoder_inputs.size(0), device=device).long()
        if self.pool_behavior == "static":
            pool = self.fc1(pool)

        if _action:
            actions_old = actions_old.reshape(batch_size, -1)
            entropys = []
            old_actions_probs = []

            for i in range(n_steps):
                if not mask1[:, 1:].eq(0).any():
                    break
                if i == 0:
                    _input = encoder_inputs[:, 0, :]

                decoder_input = torch.cat([_input, dynamic_capcity], -1)
                decoder_input = self.fc(decoder_input)
                if self.pool_behavior == "iterative_legacy":
                    pool = self.fc1(pool)
                decoder_input = decoder_input + pool

                if i == 0:
                    mask, mask1 = update_mask(demands, dynamic_capcity, index.unsqueeze(-1), mask1, i)

                p = self.prob(decoder_input, encoder_inputs, mask, temperature)

                dist = Categorical(p)
                old_actions_prob = dist.log_prob(actions_old[:, i])
                entropy = dist.entropy()
                is_done = (mask1[:, 1:].sum(1) >= (encoder_inputs.size(1) - 1)).float()
                old_actions_prob = old_actions_prob * (1. - is_done)
                entropy = entropy * (1. - is_done)

                entropys.append(entropy.unsqueeze(1))
                old_actions_probs.append(old_actions_prob.unsqueeze(1))

                dynamic_capcity = update_state(demands, dynamic_capcity, actions_old[:, i].unsqueeze(-1),
                                               capcity[0].item())
                mask, mask1 = update_mask(demands, dynamic_capcity, actions_old[:, i].unsqueeze(-1), mask1, i)

                _input = torch.gather(encoder_inputs, 1,
                                      actions_old[:, i].unsqueeze(-1).unsqueeze(-1).expand(
                                          encoder_inputs.size(0), -1, encoder_inputs.size(2))).squeeze(1)

            entropys = torch.cat(entropys, dim=1)
            old_actions_probs = torch.cat(old_actions_probs, dim=1)
            num_e = entropys.ne(0).float().sum(1)
            entropy = entropys.sum(1) / num_e
            old_actions_probs = old_actions_probs.sum(dim=1)

            return 0, 0, entropy, old_actions_probs
        else:
            log_ps = []
            actions = []
            for i in range(n_steps):
                if not mask1[:, 1:].eq(0).any():
                    break
                if i == 0:
                    _input = encoder_inputs[:, 0, :]

                decoder_input = torch.cat([_input, dynamic_capcity], -1)
                decoder_input = self.fc(decoder_input)
                if self.pool_behavior == "iterative_legacy":
                    pool = self.fc1(pool)
                decoder_input = decoder_input + pool

                if i == 0:
                    mask, mask1 = update_mask(demands, dynamic_capcity, index.unsqueeze(-1), mask1, i)

                p = self.prob(decoder_input, encoder_inputs, mask, temperature)
                if self.dropout_decoder > 0:
                    p = F.dropout(p, p=self.dropout_decoder, training=self.training)
                    p = p.clamp(min=1e-8)
                dist = Categorical(p)
                if greedy:
                    _, index = p.max(dim=-1)
                else:
                    index = dist.sample()

                actions.append(index.data.unsqueeze(1))
                log_p = dist.log_prob(index)
                is_done = (mask1[:, 1:].sum(1) >= (encoder_inputs.size(1) - 1)).float()
                log_p = log_p * (1. - is_done)
                log_ps.append(log_p.unsqueeze(1))

                dynamic_capcity = update_state(demands, dynamic_capcity, index.unsqueeze(-1), capcity[0].item())
                mask, mask1 = update_mask(demands, dynamic_capcity, index.unsqueeze(-1), mask1, i)

                _input = torch.gather(encoder_inputs, 1,
                                      index.unsqueeze(-1).unsqueeze(-1).expand(
                                          encoder_inputs.size(0), -1, encoder_inputs.size(2))).squeeze(1)

            log_ps = torch.cat(log_ps, dim=1)
            actions = torch.cat(actions, dim=1)
            log_p = log_ps.sum(dim=1)
            return actions, log_p, 0, 0


class Model(nn.Module):
    def __init__(self, input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, conv_laysers,
                 quantum_cfg=None, dropout_cfg=None, model_cfg=None, temperature=2.5):
        super(Model, self).__init__()
        if quantum_cfg is None:
            quantum_cfg = {}
        if dropout_cfg is None:
            dropout_cfg = {}
        if model_cfg is None:
            model_cfg = {}

        enc_q = quantum_cfg.get('encoder', {})
        self.encoder = Encoder(
            input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim,
            conv_layers=conv_laysers,
            quantum_fc_node=enc_q.get('fc_node', False),
            quantum_fc_edge=enc_q.get('fc_edge', False),
            quantum_gat_layers=enc_q.get('gat_layers', []),
            quantum_gat_attn_layers=enc_q.get('gat_attn_layers', []),
            n_qubits_actor=quantum_cfg.get('n_qubits_actor', 4),
            n_layers_quantum=quantum_cfg.get('n_layers_quantum_actor', 4),
            rotation=quantum_cfg.get('rotation', 'RYRZ'),
            topology=quantum_cfg.get('topology', 'ring'),
            backend=quantum_cfg.get('backend', 'lightning.qubit'),
            diff_method=quantum_cfg.get('diff_method', 'best'),
            dropout_gat=dropout_cfg.get('gat', 0.0),
            use_ffn=model_cfg.get('use_ffn', False),
            normalization=model_cfg.get('normalization', 'batch'),
        )
        # Build actor decoder config with backend/diff_method inherited from top-level quantum_cfg
        decoder_q_cfg = quantum_cfg.get('actor_decoder', None)
        if decoder_q_cfg is not None:
            decoder_q_cfg = dict(decoder_q_cfg)  # copy to avoid mutating original
            decoder_q_cfg.setdefault('backend', quantum_cfg.get('backend', 'lightning.qubit'))
            decoder_q_cfg.setdefault('diff_method', quantum_cfg.get('diff_method', 'best'))

        self.decoder = Decoder1(
            hidden_node_dim, hidden_node_dim,
            dropout_decoder=dropout_cfg.get('decoder', 0.0),
            attn_dropout=dropout_cfg.get('attn', 0.0),
            dropout_attn_output=dropout_cfg.get('attn_output', 0.0),
            temperature=temperature,
            tanh_clipping=model_cfg.get('tanh_clipping', 10.0),
            quantum_decoder_cfg=decoder_q_cfg,
            pool_behavior=model_cfg.get('pool_behavior', 'static'),
        )

    def forward(self, datas, actions_old, n_steps, batch_size, device, greedy, _action, temperature=None):
        x = self.encoder(datas)
        pooled = x.mean(dim=1)
        demand = datas.demand
        capcity = datas.capcity
        actions, log_p, entropy, dists = self.decoder(
            x, pooled, actions_old, capcity, demand, n_steps, batch_size,
            device, greedy, _action, temperature)
        return actions, log_p, entropy, dists, x


class Critic(nn.Module):
    """Estimates problem complexity. Supports per-layer quantum selection
    and selectable decoder type (conv1d or fc)."""

    def __init__(self, hidden_node_dim, quantum_critic_cfg=None,
                 n_qubits_critic=4, n_layers_quantum_critic=2, dropout_critic=0.0,
                 rotation='RYRZ', topology='ring', critic_decoder='conv1d',
                 backend='lightning.qubit', diff_method='best'):
        super(Critic, self).__init__()
        if quantum_critic_cfg is None:
            quantum_critic_cfg = {}

        self.hidden_node_dim = hidden_node_dim
        self.dropout_critic = dropout_critic
        self.critic_decoder = critic_decoder
        quantum_layers_list = quantum_critic_cfg.get('layers', [])
        self.use_quantum_any = len(quantum_layers_list) > 0

        def _make_classical(in_f, out_f, idx):
            if idx in quantum_layers_list:
                return None  # handled below
            if critic_decoder == 'fc':
                return nn.Linear(in_f, out_f)
            return nn.Conv1d(in_f, out_f, kernel_size=1)

        if 0 in quantum_layers_list:
            self.fc1 = create_quantum_layer(hidden_node_dim, 20, n_qubits_critic, n_layers_quantum_critic,
                                            rotation=rotation, topology=topology,
                                            backend=backend, diff_method=diff_method)
            self._fc1_is_quantum = True
            model_logger.info("Critic.fc1: quantum")
        else:
            self.fc1 = _make_classical(hidden_node_dim, 20, 0)
            self._fc1_is_quantum = False

        if 1 in quantum_layers_list:
            self.fc2 = create_quantum_layer(20, 20, n_qubits_critic, n_layers_quantum_critic,
                                            rotation=rotation, topology=topology,
                                            backend=backend, diff_method=diff_method)
            self._fc2_is_quantum = True
            model_logger.info("Critic.fc2: quantum")
        else:
            self.fc2 = _make_classical(20, 20, 1)
            self._fc2_is_quantum = False

        if 2 in quantum_layers_list:
            self.fc3 = create_quantum_layer(20, 1, n_qubits_critic, n_layers_quantum_critic,
                                            rotation=rotation, topology=topology,
                                            backend=backend, diff_method=diff_method)
            self._fc3_is_quantum = True
            model_logger.info("Critic.fc3: quantum")
        else:
            self.fc3 = _make_classical(20, 1, 2)
            self._fc3_is_quantum = False

    def _apply_layer(self, layer, x, is_quantum, batch_size, n_nodes, in_features, activate=True):
        """Apply a layer that may be quantum (expects 2D) or Conv1d (expects 3D)."""
        if is_quantum:
            if x.dim() == 3:  # (batch, features, nodes) from Conv1d → flatten
                x = x.transpose(1, 2).reshape(-1, x.shape[1])
            out = layer(x)
        else:
            if x.dim() == 2:  # (batch*nodes, features) from quantum → reshape for Conv1d
                x = x.reshape(batch_size, n_nodes, -1).transpose(1, 2)
            out = layer(x)
        if activate:
            out = F.relu(out)
        return out

    def forward(self, x):
        if self.use_quantum_any:
            batch_size, n_nodes = x.shape[0], x.shape[1]
            # Prepare both formats
            x_flat = x.reshape(-1, self.hidden_node_dim)
            x_conv = x.transpose(2, 1)

            # Layer 1
            if self._fc1_is_quantum:
                output = F.relu(self.fc1(x_flat))
            else:
                output = F.relu(self.fc1(x_conv))
            output = F.dropout(output, p=self.dropout_critic, training=self.training)

            # Layer 2
            output = self._apply_layer(self.fc2, output, self._fc2_is_quantum, batch_size, n_nodes, 20)
            output = F.dropout(output, p=self.dropout_critic, training=self.training)

            # Layer 3
            output = self._apply_layer(self.fc3, output, self._fc3_is_quantum, batch_size, n_nodes, 20, activate=False)

            # Aggregate
            if output.dim() == 2:  # quantum output: (batch*nodes, 1)
                output = output.reshape(batch_size, n_nodes, -1)
                value = output.sum(dim=1).squeeze(-1)
            else:  # Conv1d output: (batch, 1, nodes)
                value = output.sum(dim=2).squeeze(-1)
            return value
        else:
            # Pure classical pathway
            if self.critic_decoder == 'fc':
                # FC path: (batch, nodes, features) → Linear
                output = F.relu(self.fc1(x))
                output = F.dropout(output, p=self.dropout_critic, training=self.training)
                output = F.relu(self.fc2(output))
                output = F.dropout(output, p=self.dropout_critic, training=self.training)
                value = self.fc3(output).sum(dim=1).squeeze(-1)
            else:
                # Conv1d path: (batch, features, nodes)
                x1 = x.transpose(2, 1)
                output = F.relu(self.fc1(x1))
                output = F.dropout(output, p=self.dropout_critic, training=self.training)
                output = F.relu(self.fc2(output))
                output = F.dropout(output, p=self.dropout_critic, training=self.training)
                value = self.fc3(output).sum(dim=2).squeeze(-1)
            return value


class Actor_critic(nn.Module):
    def __init__(self, input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, conv_laysers,
                 quantum_cfg=None, dropout_cfg=None, model_cfg=None, temperature=2.5):
        super(Actor_critic, self).__init__()
        if quantum_cfg is None:
            quantum_cfg = {}
        if dropout_cfg is None:
            dropout_cfg = {}
        if model_cfg is None:
            model_cfg = {}

        self.actor = Model(input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, conv_laysers,
                           quantum_cfg=quantum_cfg, dropout_cfg=dropout_cfg, model_cfg=model_cfg,
                           temperature=temperature)

        # Build critic via factory (supports shared_conv1d, shared_fc,
        # own_encoder_conv1d, own_encoder_mlp — see critics.py for rationale)
        from DRL_Lei.critics import build_critic, _resolve_critic_type, CriticKOOL, CriticOwnEncoderConv1d
        critic_section = quantum_cfg.get('critic', {})
        critic_type = _resolve_critic_type(critic_section)
        self.critic = build_critic(
            critic_type, hidden_node_dim,
            quantum_cfg=quantum_cfg,
            quantum_critic_cfg=critic_section,
            dropout_cfg=dropout_cfg,
            model_cfg=model_cfg,
            input_node_dim=input_node_dim,
            input_edge_dim=input_edge_dim,
            hidden_edge_dim=hidden_edge_dim,
            conv_layers=conv_laysers,
        )
        self._critic_has_own_encoder = isinstance(self.critic, (CriticKOOL, CriticOwnEncoderConv1d))

    def act(self, datas, actions, steps, batch_size, device, greedy, _action, temperature=None):
        actions, log_p, _, _, _ = self.actor(datas, actions, steps, batch_size, device, greedy, _action, temperature)
        return actions, log_p

    def evaluate(self, datas, actions, steps, batch_size, device, greedy, _action, temperature=None):
        _, _, entropy, old_log_p, x = self.actor(datas, actions, steps, batch_size, device, greedy, _action, temperature)
        # Own-encoder critics have their own encoder — feed raw data, not actor embeddings
        if self._critic_has_own_encoder:
            value = self.critic(datas)
        else:
            value = self.critic(x)
        return entropy, old_log_p, value


class Memory:
    def __init__(self):
        self.input_x = []
        self.input_attr = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.capcity = []
        self.demand = []

    def def_memory(self):
        self.input_x.clear()
        self.input_attr.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.capcity.clear()
        self.demand.clear()


class Agentppo:
    def __init__(self, steps, greedy, lr, input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim,
                 epoch=1, batch_size=32, conv_laysers=3, entropy_value=0.2, eps_clip=0.2,
                 n_nodes=21, num_workers=0, pin_memory=False, lr_decay=0.96,
                 quantum_cfg=None, dropout_cfg=None, model_cfg=None, seeds_cfg=None,
                 temperature_config=None, total_training_epochs=100,
                 verbose_level=1, device=None, loss_weights=None):
        # ===== Device =====
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        if quantum_cfg is None:
            quantum_cfg = {}
        if dropout_cfg is None:
            dropout_cfg = {}
        if model_cfg is None:
            model_cfg = {}
        if seeds_cfg is None:
            seeds_cfg = {}

        # ===== REPRODUCIBILITY =====
        self.data_seed = seeds_cfg.get('data_seed', 42)
        self.model_seed = seeds_cfg.get('model_seed', 42)
        self.random_seed = seeds_cfg.get('random_seed', 42)
        _random.seed(self.random_seed)
        np.random.seed(self.data_seed)
        torch.manual_seed(self.model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
        if verbose_level >= 1:
            model_logger.info(f"Seeds: data={self.data_seed}, model={self.model_seed}, random={self.random_seed}")

        # ===== TEMPERATURE =====
        if temperature_config is None:
            temperature_config = {'initial': 2.5, 'final': 2.5, 'schedule': 'constant', 'decay_rate': 1.0}
        self.temperature_config = temperature_config
        self.current_temperature = temperature_config['initial']
        self.total_training_epochs = total_training_epochs

        # ===== VERBOSE =====
        self.verbose_level = verbose_level

        self.n_nodes = n_nodes
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.lr_decay = lr_decay

        self.policy = Actor_critic(input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, conv_laysers,
                                   quantum_cfg=quantum_cfg, dropout_cfg=dropout_cfg, model_cfg=model_cfg,
                                   temperature=self.current_temperature)
        self.old_polic = Actor_critic(input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, conv_laysers,
                                      quantum_cfg=quantum_cfg, dropout_cfg=dropout_cfg, model_cfg=model_cfg,
                                      temperature=self.current_temperature)
        self.old_polic.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.MseLoss = nn.MSELoss()
        self.batch_size = batch_size
        self.epoch = epoch
        self.steps = steps
        self.entropy_value = entropy_value
        self.eps_clip = eps_clip

        # Loss weights (Paper Eq 23): L = λp·LCLIP + λv·LMSE − λe·LE
        if loss_weights is None:
            loss_weights = {}
        self.lambda_policy = loss_weights.get('policy', 1.0)
        self.lambda_value = loss_weights.get('value', 0.5)
        self.lambda_entropy = loss_weights.get('entropy', entropy_value)

        self.greedy = greedy
        self._action = True
        self.conv_layers = conv_laysers
        self.input_node_dim = input_node_dim
        self.input_edge_dim = input_edge_dim
        self.hidden_node_dim = hidden_node_dim
        self.hidden_edge_dim = hidden_edge_dim
        self.batch_idx = 1
        self.times, self.losses, self.rewards, self.critic_rewards = [], [], [], []

        # Pre-compute edge_index once
        edges = [[i, j] for i in range(self.n_nodes) for j in range(self.n_nodes)]
        self.edges_index = torch.LongTensor(edges).transpose(0, 1)

        # Parameter counting
        self._param_counts = log_model_summary(self.policy, verbose_level)

    def adv_normalize(self, adv):
        std = adv.std()
        assert std != 0. and not torch.isnan(std), 'Need nonzero std'
        n_advs = (adv - adv.mean()) / (adv.std() + 1e-8)
        return n_advs

    def compute_temperature(self, epoch, total_epochs):
        cfg = self.temperature_config
        schedule = cfg.get('schedule', 'constant')
        if schedule == 'constant':
            return cfg['initial']
        elif schedule == 'linear':
            progress = epoch / max(total_epochs - 1, 1)
            return cfg['initial'] + (cfg['final'] - cfg['initial']) * progress
        elif schedule == 'exponential':
            return cfg['initial'] * (cfg['decay_rate'] ** epoch)
        else:
            model_logger.warning(f"Unknown temperature schedule: {schedule}, using constant")
            return cfg['initial']

    def set_policy_temperature(self, temperature):
        self.current_temperature = temperature
        for net in [self.policy, self.old_polic]:
            if hasattr(net, 'actor') and hasattr(net.actor, 'decoder'):
                if hasattr(net.actor.decoder, 'prob'):
                    net.actor.decoder.prob.set_temperature(temperature)

    def value_loss_gae(self, val_targ, old_vs, value_od, clip_val):
        vs_clipped = old_vs + torch.clamp(old_vs - value_od, -clip_val, +clip_val)
        val_loss_mat_unclipped = self.MseLoss(old_vs, val_targ)
        val_loss_mat_clipped = self.MseLoss(vs_clipped, val_targ)
        val_loss_mat = torch.max(val_loss_mat_unclipped, val_loss_mat_clipped)
        return val_loss_mat

    def update(self, memory, epoch):
        # Temperature annealing
        temp = self.compute_temperature(epoch, self.total_training_epochs)
        self.set_policy_temperature(temp)
        if self.verbose_level >= 2 and temp != self.temperature_config['initial']:
            print(f"    [UPDATE] Temperature annealed to {temp:.4f} (epoch {epoch})")

        if self.verbose_level >= 2:
            print(f"    [UPDATE] Starting - memory has {len(memory.actions)} episodes")

        old_input_x = torch.stack(memory.input_x)
        old_input_attr = torch.stack(memory.input_attr)
        old_demand = torch.stack(memory.demand)
        old_capcity = torch.stack(memory.capcity)

        # Handle variable-length action sequences using pad_sequence
        old_action = pad_sequence(memory.actions, batch_first=True, padding_value=-1)
        old_rewards = torch.stack(memory.rewards).unsqueeze(-1)
        old_log_probs = torch.stack(memory.log_probs).unsqueeze(-1)

        datas = [
            Data(
                x=old_input_x[i],
                edge_index=self.edges_index,
                edge_attr=old_input_attr[i],
                actions=old_action[i],
                rewards=old_rewards[i],
                log_probs=old_log_probs[i],
                demand=old_demand[i],
                capcity=old_capcity[i]
            ) for i in range(old_input_x.size(0))
        ]

        self.policy.to(self.device)
        data_loader = DataLoader(datas, batch_size=self.batch_size, shuffle=False,
                                 num_workers=self.num_workers, pin_memory=self.pin_memory)
        scheduler = LambdaLR(self.optimizer, lr_lambda=lambda f: self.lr_decay ** epoch)

        for i in range(self.epoch):
            if self.verbose_level >= 2:
                print(f"      [UPDATE-EPOCH {i}] Starting inner training epoch...")

            self.policy.train()
            epoch_start = time.time()
            self.times, self.losses, self.rewards, self.critic_rewards = [], [], [], []

            for batch_idx, batch in enumerate(data_loader):
                self.batch_idx += 1
                batch = batch.to(self.device)
                entropy, log_probs, value = self.policy.evaluate(
                    batch, batch.actions, self.steps, self.batch_size,
                    self.device, self.greedy, self._action)

                rewar = batch.rewards
                rewar = self.adv_normalize(rewar)
                mse_loss = self.MseLoss(rewar, value)

                ratios = torch.exp(log_probs - batch.log_probs)
                advantages = rewar - value.detach()

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                loss = (self.lambda_policy * torch.min(surr1, surr2)
                        + self.lambda_value * mse_loss
                        - self.lambda_entropy * entropy)
                self.optimizer.zero_grad()
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
                self.optimizer.step()

                scheduler.step()
                if batch_idx % 10 == 0:
                    if mlflow.active_run() is not None:
                        ratio_mean = ratios.mean().item()
                        ratio_std = ratios.std().item()
                        clip_fraction = (torch.abs(ratios - 1.0) > self.eps_clip).float().mean().item()
                        mlflow.log_metric("ppo/loss", loss.mean().item(), step=self.batch_idx)
                        mlflow.log_metric("ppo/value_mse", mse_loss.mean().item(), step=self.batch_idx)
                        mlflow.log_metric("ppo/entropy", entropy.mean().item(), step=self.batch_idx)
                        mlflow.log_metric("ppo/ratio_mean", ratio_mean, step=self.batch_idx)
                        mlflow.log_metric("ppo/ratio_std", ratio_std, step=self.batch_idx)
                        mlflow.log_metric("ppo/clip_fraction", clip_fraction, step=self.batch_idx)
                        mlflow.log_metric("opt/lr", self.optimizer.param_groups[0]["lr"], step=self.batch_idx)

                self.rewards.append(torch.mean(rewar.detach()).item())
                self.losses.append(torch.mean(loss.detach()).item())

            if self.verbose_level >= 2:
                print(f"      [UPDATE-EPOCH {i}] Completed with {len(self.losses)} batches")

        self.old_polic.load_state_dict(self.policy.state_dict())


if __name__ == '__main__':
    raise Exception('Cannot be called from main')
