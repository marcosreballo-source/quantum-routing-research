import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from DRL_Lei.vrpUpdate import update_mask, update_state
from DRL_Lei.quantum_layers import SwitchableLinear

# Shared encoder/attention modules (canonical source)
from DRL_Lei.shared_modules import GatConv, Encoder, Attention1, ProbAttention  # noqa: F401

INIT = False
max_grad_norm = 2


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

    def forward(self, encoder_inputs, pool, capcity, demand, n_steps, device, temperature=None, greedy=False):

        mask1 = encoder_inputs.new_zeros((encoder_inputs.size(0), encoder_inputs.size(1)))
        mask = encoder_inputs.new_zeros((encoder_inputs.size(0), encoder_inputs.size(1)))
        dynamic_capcity = capcity.view(encoder_inputs.size(0), -1)
        demands = demand.view(encoder_inputs.size(0), encoder_inputs.size(1))
        index = torch.zeros(encoder_inputs.size(0), device=device).long()
        _input = encoder_inputs[:, 0, :]  # depot node; updated each step via index

        log_ps = []
        actions = []

        if self.pool_behavior == "static":
            pool = self.fc1(pool)

        for i in range(n_steps):
            if not mask1[:, 1:].eq(0).any():
                break

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
        return actions, log_p


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
        # Build decoder config with backend/diff_method inherited from top-level quantum_cfg
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

    def forward(self, datas, n_steps, device, greedy=False, temperature=None):
        x = self.encoder(datas)
        pooled = x.mean(dim=1)
        demand = datas.demand
        capcity = datas.capcity
        actions, log_p = self.decoder(x, pooled, capcity, demand, n_steps, device, temperature, greedy)
        return actions, log_p
