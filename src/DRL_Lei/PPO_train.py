import json
import time
from pathlib import Path
from collections import OrderedDict, namedtuple, defaultdict
from contextlib import contextmanager
from itertools import product

import torch
import mlflow
import numpy as np

from DRL_Lei.VRP_PPO_Model import Agentppo, Memory
from DRL_Lei.creat_vrp import creat_data, reward, reward1
from DRL_Lei.evaluate import evaluate
from DRL_Lei.utils.paths import make_eval_dir
from DRL_Lei.utils.profiler import (
    BlockTimer, _NullTimer, make_profiler, log_profiler_to_mlflow,
    log_device_info, log_quantum_summary, MemoryTracker,
    V_NONE, V_BASIC, V_MEDIUM, V_FULL,
)


# ---------------------------------------------------------------------------
# quantum param-count utilities
# ---------------------------------------------------------------------------

def _count_quantum_params(model) -> int:
    """Count trainable parameters whose name contains 'quantum'."""
    return sum(
        p.numel() for name, p in model.named_parameters()
        if p.requires_grad and 'quantum' in name
    )


def log_parameter_counts_to_mlflow(param_counts: dict, model=None, quantum_enabled: bool = False) -> None:
    """
    Log per-component parameter counts to MLflow (step=0, logged once at init).
    If quantum_enabled and model is provided, also logs quantum reduction %.
    """
    if mlflow.active_run() is None:
        return
    pc = param_counts
    mlflow.log_metric("param_count/total", pc.get('total', 0), step=0)
    mlflow.log_metric("param_count/actor_encoder", pc.get('encoder', 0), step=0)
    mlflow.log_metric("param_count/actor_decoder", pc.get('decoder', 0), step=0)
    mlflow.log_metric("param_count/critic", pc.get('critic', 0), step=0)

    if quantum_enabled and model is not None:
        total = max(pc.get('total', 1), 1)
        encoder = max(pc.get('encoder', 1), 1)
        decoder = max(pc.get('decoder', 1), 1)
        q_total = _count_quantum_params(model)
        q_encoder = sum(
            p.numel() for name, p in model.named_parameters()
            if p.requires_grad and 'quantum' in name and 'actor.encoder.' in name
        )
        q_decoder = sum(
            p.numel() for name, p in model.named_parameters()
            if p.requires_grad and 'quantum' in name and 'actor.decoder.' in name
        )
        mlflow.log_metric("quantum_reduction_pct/total", q_total / total * 100.0, step=0)
        mlflow.log_metric("quantum_reduction_pct/encoder", q_encoder / encoder * 100.0, step=0)
        mlflow.log_metric("quantum_reduction_pct/decoder", q_decoder / max(decoder, 1) * 100.0, step=0)
        # Also store raw counts for reference
        mlflow.log_metric("param_count/quantum_total", q_total, step=0)


# ---------------------------------------------------------------------------
# BestRouteTracker — tracks best validation tour + improvement history
# ---------------------------------------------------------------------------

class BestRouteTracker:
    """
    Tracks the best validation tour length across epochs and saves an
    improvement history + route JSON artifacts to MLflow.
    """

    def __init__(self):
        self.best_cost: float = float('inf')
        self.best_epoch: int = -1
        self.improvement_history: list[dict] = []  # {epoch, cost, improved}
        self.best_route_data: dict | None = None   # node coords of best tour batch

    def update(self, epoch: int, cost: float, batch_nodes=None) -> bool:
        """
        Update tracker. Returns True if a new best was found.
        batch_nodes: optional first sample from the validation batch (CPU tensor [n_nodes, 2])
        """
        improved = cost < self.best_cost
        if improved:
            self.best_cost = cost
            self.best_epoch = epoch
            if batch_nodes is not None:
                self.best_route_data = {
                    "epoch": epoch,
                    "cost": cost,
                    "nodes": batch_nodes.cpu().tolist() if hasattr(batch_nodes, 'cpu') else batch_nodes,
                }
                if mlflow.active_run() is not None:
                    try:
                        mlflow.log_text(
                            json.dumps(self.best_route_data, indent=2),
                            f"routes/best_train_route_epoch_{epoch}.json",
                        )
                    except Exception:
                        pass  # non-critical
        self.improvement_history.append({
            "epoch": epoch,
            "cost": round(cost, 6),
            "best_so_far": round(self.best_cost, 6),
            "improved": improved,
        })
        return improved

    def log_final_artifacts(self) -> None:
        """Log improvement history and final summary to MLflow."""
        if mlflow.active_run() is None:
            return
        try:
            mlflow.log_text(
                json.dumps(self.improvement_history, indent=2),
                "routes/improvement_history.json",
            )
            summary = {
                "best_epoch": self.best_epoch,
                "best_train_cost": round(self.best_cost, 6),
                "total_improvements": sum(1 for h in self.improvement_history if h["improved"]),
            }
            if self.best_route_data is not None:
                summary["best_route_nodes"] = self.best_route_data.get("nodes")
            mlflow.log_text(json.dumps(summary, indent=2), "routes/best_route_summary.json")
            mlflow.log_metric("best_epoch", self.best_epoch)
            mlflow.log_metric("best_train_cost", self.best_cost)
        except Exception:
            pass  # non-critical


# ---------------------------------------------------------------------------
# quantum config validation
# ---------------------------------------------------------------------------
def _validate_quantum_layer_indices(conv_layers, enc_q_cfg, critic_q_cfg=None):
    """Raise ValueError if any quantum layer index is out of bounds."""
    gat_layers = enc_q_cfg.get('gat_layers', [])
    gat_attn_layers = enc_q_cfg.get('gat_attn_layers', [])
    all_enc = list(gat_layers) + list(gat_attn_layers)
    if all_enc and max(all_enc) >= conv_layers:
        raise ValueError(
            f"Quantum layer index out of bounds: max(gat_layers + gat_attn_layers) = "
            f"{max(all_enc)} >= conv_layers = {conv_layers}. "
            f"Valid indices: 0 to {conv_layers - 1}."
        )
    if critic_q_cfg:
        critic_layers = critic_q_cfg.get('layers', [])
        crit_enc = critic_q_cfg.get('encoder', {})
        crit_gat = list(crit_enc.get('gat_layers', [])) + list(crit_enc.get('gat_attn_layers', []))
        if critic_layers and max(critic_layers) >= conv_layers:
            raise ValueError(
                f"Critic quantum layer index out of bounds: max(critic.layers) = "
                f"{max(critic_layers)} >= conv_layers = {conv_layers}."
            )
        if crit_gat and max(crit_gat) >= conv_layers:
            raise ValueError(
                f"Critic encoder quantum layer index out of bounds: max = "
                f"{max(crit_gat)} >= conv_layers = {conv_layers}."
            )


# ---------------------------------------------------------------------------
# rollout helper
# ---------------------------------------------------------------------------
def rollout(model, dataset, batch_size, steps, n_nodes, device):
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model.act(bat, 0, steps, batch_size, device, True, False)
            cost = reward1(bat.x, cost.detach(), n_nodes)
        return cost.cpu()

    totall_cost = torch.cat([eval_model_bat(bat.to(device)) for bat in dataset], 0)
    return totall_cost


# ---------------------------------------------------------------------------
# TrainPPO
# ---------------------------------------------------------------------------
class TrainPPO:
    def __init__(self, steps, greedy, lr, input_node_dim, hidden_node_dim,
                 input_edge_dim, hidden_edge_dim,
                 epoch=40, batch_size=32, conv_laysers=3,
                 entropy_value=0.01, eps_clip=0.2, timestep=4, ppo_epoch=2,
                 n_nodes=21, num_workers=0, pin_memory=False, lr_decay=0.96,
                 quantum_cfg=None, dropout_cfg=None, model_cfg=None, seeds_cfg=None,
                 temperature_config=None, verbose_level=1, device=None,
                 loss_weights=None):

        self.steps = steps
        self.greedy = greedy
        self.batch_size = batch_size
        self.update_timestep = timestep
        self.epoch = epoch
        self.memory = Memory()
        self.n_nodes = n_nodes
        self.verbose_level = verbose_level

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.agent = Agentppo(
            steps, greedy, lr, input_node_dim, hidden_node_dim,
            input_edge_dim, hidden_edge_dim, ppo_epoch, batch_size,
            conv_laysers, entropy_value, eps_clip,
            n_nodes=n_nodes, num_workers=num_workers, pin_memory=pin_memory,
            lr_decay=lr_decay,
            quantum_cfg=quantum_cfg, dropout_cfg=dropout_cfg, model_cfg=model_cfg,
            seeds_cfg=seeds_cfg,
            temperature_config=temperature_config, total_training_epochs=epoch,
            verbose_level=verbose_level, device=device,
            loss_weights=loss_weights,
        )

    def run_train(self, data_loader, batch_size, valid_loder, run_dir,
                  cfg=None, profiler_enabled=False):
        memory = Memory()
        profiler = make_profiler(profiler_enabled)
        device = self.device

        self.agent.old_polic.to(device)
        quantum_enabled = bool(getattr(cfg, 'quantum', None) and getattr(cfg.quantum, 'enabled', False)) if cfg else False

        costs = []
        best_cost = float('inf')
        route_tracker = BestRouteTracker()
        for i in range(self.epoch):
            print('old_epoch:', i, '***************************************')
            self.agent.old_polic.train()
            times, losses, rewards2, critic_rewards = [], [], [], []
            epoch_start = time.time()
            start = epoch_start
            profiler.reset()

            for batch_idx, batch in enumerate(data_loader):
                if self.verbose_level >= 2:
                    print(f"  [Batch {batch_idx}] Starting batch processing...")

                with profiler.measure("1_data_transfer"):
                    x, attr, capcity, demand = batch.x, batch.edge_attr, batch.capcity, batch.demand
                    batch = batch.to(device)
                    x = x.view(batch_size, self.n_nodes, 2)
                    attr = attr.view(batch_size, self.n_nodes * self.n_nodes, 1)
                    capcity = capcity.view(batch_size, 1)
                    demand = demand.view(batch_size, self.n_nodes, 1)

                with profiler.measure("2_actor_forward"):
                    actions, log_p = self.agent.old_polic.act(
                        batch, 0, self.steps, batch_size, device, self.greedy, False)

                with profiler.measure("3_reward_compute"):
                    rewards = reward1(batch.x, actions, self.n_nodes)

                with profiler.measure("4_detach_and_unbind"):
                    actions = actions.detach()
                    log_p = log_p.detach()
                    rewards = rewards.detach()

                    memory.input_x.extend(list(torch.unbind(x, dim=0)))
                    memory.input_attr.extend(list(torch.unbind(attr, dim=0)))
                    memory.actions.extend(list(torch.unbind(actions, dim=0)))
                    memory.log_probs.extend(list(torch.unbind(log_p, dim=0)))
                    memory.rewards.extend(list(torch.unbind(rewards, dim=0)))
                    memory.capcity.extend(list(torch.unbind(capcity, dim=0)))
                    memory.demand.extend(list(torch.unbind(demand, dim=0)))

                if (batch_idx + 1) % self.update_timestep == 0:
                    with profiler.measure("5_ppo_update_backprop"):
                        self.agent.update(memory, i)
                    memory.def_memory()

                rewards2.append(torch.mean(rewards.detach()).item())
                time_Space = max(1, len(data_loader) // 10)
                if (batch_idx + 1) % time_Space == 0:
                    end = time.time()
                    times.append(end - start)
                    start = end
                    mean_reward = np.mean(rewards2[-time_Space:])
                    print('  Batch %d/%d, reward: %2.3f,took: %2.4fs' %
                          (batch_idx, len(data_loader), mean_reward, times[-1]))
                    if mlflow.active_run() is not None:
                        global_step = i * len(data_loader) + batch_idx
                        mlflow.log_metric("train/mean_reward", mean_reward, step=global_step)
                        mlflow.log_metric("time/batch_seconds", times[-1], step=global_step)

            cost = rollout(self.agent.policy, valid_loder, batch_size, self.steps, self.n_nodes, device)
            cost = cost.mean()
            costs.append(cost.item())
            if cost.item() < best_cost:
                best_cost = cost.item()
            route_tracker.update(i, cost.item())

            if mlflow.active_run() is not None:
                mlflow.log_metric("val/mean_cost", cost.item(), step=i)
                mlflow.log_metric("val/best_mean_cost", best_cost, step=i)
                mlflow.log_metric("train/mean_reward_epoch", float(np.mean(rewards2)), step=i)
                mlflow.log_metric("time/epoch_seconds", time.time() - epoch_start, step=i)

                # Profiler metrics
                if profiler_enabled and profiler.totals:
                    log_profiler_to_mlflow(profiler, step=i, prefix="profiler")
                    if self.verbose_level >= V_BASIC:
                        print(profiler.summary(prefix=f"  [Epoch {i}] "))

                # Quantum gradient norms
                if quantum_enabled:
                    q_grads_actor, q_grads_critic = [], []
                    for name, param in self.agent.policy.actor.named_parameters():
                        if 'quantum' in name and param.grad is not None:
                            q_grads_actor.append(param.grad.norm().item())
                    for name, param in self.agent.policy.critic.named_parameters():
                        if 'quantum' in name and param.grad is not None:
                            q_grads_critic.append(param.grad.norm().item())
                    if q_grads_actor:
                        mlflow.log_metric("quantum/grad_norm_actor", np.mean(q_grads_actor), step=i)
                    if q_grads_critic:
                        mlflow.log_metric("quantum/grad_norm_critic", np.mean(q_grads_critic), step=i)

            print('Problem:VRP''%s' % self.n_nodes, '/ Average distance:', cost.item())
            print(costs)

            save_path = run_dir / 'checkpoints' / 'last.pt'
            torch.save(self.agent.old_polic.state_dict(), save_path)
            if i % 10 == 0 or i == self.epoch - 1:
                if i < self.epoch - 1:
                    save_path = run_dir / 'checkpoints' / f'actor_{i}.pt'
                    torch.save(self.agent.old_polic.state_dict(), save_path)
                elif i == self.epoch - 1:
                    save_path = run_dir / 'checkpoints' / 'final.pt'
                    torch.save(self.agent.old_polic.state_dict(), save_path)
                    if mlflow.active_run() is not None:
                        mlflow.log_artifact(str(save_path))
                    route_tracker.log_final_artifacts()
                    if cfg is not None:
                        eval_cfg = getattr(cfg, 'evaluation', None)
                        eval_enabled = getattr(eval_cfg, 'enabled', True) if eval_cfg else True
                        if eval_enabled:
                            eval_dir = make_eval_dir(save_path)
                            evaluate(cfg, save_path, eval_dir)
                        else:
                            print("  [eval] Evaluation disabled via config (evaluation.enabled=false)")


# ---------------------------------------------------------------------------
# train() entry point — called by main.py
# ---------------------------------------------------------------------------
def train(cfg, run_dir: Path):
    class RunBuilder:
        @staticmethod
        def get_runs(params):
            Run = namedtuple('Run', params.keys())
            runs = []
            for v in product(*params.values()):
                runs.append(Run(*v))
            return runs

    # ----- runtime resolved by config.py: resolve_mode() -----
    runtime = getattr(cfg, 'runtime', None)
    device = getattr(runtime, 'device', None) if runtime else None
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    verbose_level = getattr(runtime, 'verbose_level', 1) if runtime else 1
    profiler_enabled = getattr(runtime, 'profiler_enabled', False) if runtime else False

    print(f"\nDevice: {device}  |  verbose={verbose_level}  |  profiler={profiler_enabled}")
    log_device_info(device, verbose_level)
    log_quantum_summary(cfg, verbose_level)

    n_nodes = cfg.problem.n_nodes + 1  # Adding depot node
    greedy = True if cfg.training.policy_mode == 'greedy' else False

    # ----- build quantum_cfg dict from new granular config -----
    q_section = getattr(cfg, 'quantum', None)
    quantum_cfg = {}
    if q_section and getattr(q_section, 'enabled', False):
        enc = getattr(q_section, 'encoder', None)
        crit = getattr(q_section, 'critic', None)
        dec = getattr(q_section, 'actor_decoder', None)
        # quantum.critic.enabled gates per-critic quantum (consistent with master switch)
        crit_enabled = getattr(crit, 'enabled', True) if crit else True
        quantum_cfg = {
            'n_qubits_actor': getattr(q_section, 'n_qubits_actor', 4),
            'n_layers_quantum_actor': getattr(q_section, 'n_layers_quantum_actor', 4),
            'n_qubits_critic': getattr(q_section, 'n_qubits_critic', 4),
            'n_layers_quantum_critic': getattr(q_section, 'n_layers_quantum_critic', 2),
            'rotation': getattr(q_section, 'rotation', 'RYRZ'),
            'topology': getattr(q_section, 'topology', 'ring'),
            'backend': getattr(q_section, 'backend', 'lightning.qubit'),
            'diff_method': getattr(q_section, 'diff_method', 'best'),
            'encoder': {
                'fc_node': getattr(enc, 'fc_node', False) if enc else False,
                'fc_edge': getattr(enc, 'fc_edge', False) if enc else False,
                'gat_layers': list(getattr(enc, 'gat_layers', [])) if enc else [],
                'gat_attn_layers': list(getattr(enc, 'gat_attn_layers', [])) if enc else [],
            },
            'critic': {
                'layers': list(getattr(crit, 'layers', [])) if (crit and crit_enabled) else [],
                'type': getattr(crit, 'type', 'shared_conv1d') if crit else 'shared_conv1d',
            },
            'actor_decoder': {
                'fc': getattr(dec, 'fc', False) if dec else False,
                'fc1': getattr(dec, 'fc1', False) if dec else False,
                'n_qubits': getattr(dec, 'n_qubits', 4) if dec else 4,
                'n_layers': getattr(dec, 'n_layers', 4) if dec else 4,
            },
        }
        print(f"Quantum ENABLED: {quantum_cfg}")
        _validate_quantum_layer_indices(
            cfg.model.conv_layers,
            quantum_cfg.get('encoder', {}),
            quantum_cfg.get('critic', {}),
        )

    # ----- build dropout_cfg -----
    d_section = getattr(cfg, 'dropout', None)
    dropout_cfg = {}
    if d_section:
        dropout_cfg = {
            'gat': getattr(d_section, 'gat', 0.0),
            'attn': getattr(d_section, 'attn', 0.0),
            'attn_output': getattr(d_section, 'attn_output', 0.0),
            'decoder': getattr(d_section, 'decoder', 0.0),
            'critic': getattr(d_section, 'critic', 0.0),
        }

    # ----- seeds -----
    s_section = getattr(cfg, 'seeds', None)
    seeds_cfg = {}
    if s_section:
        seeds_cfg = {
            'data_seed': getattr(s_section, 'data_seed', 42),
            'model_seed': getattr(s_section, 'model_seed', 42),
            'random_seed': getattr(s_section, 'random_seed', 42),
        }

    # ----- temperature -----
    t_section = getattr(cfg, 'temperature', None)
    temperature_config = {
        'initial': getattr(t_section, 'initial', 2.5) if t_section else 2.5,
        'final': getattr(t_section, 'final', 2.5) if t_section else 2.5,
        'schedule': getattr(t_section, 'schedule', 'constant') if t_section else 'constant',
        'decay_rate': getattr(t_section, 'decay_rate', 1.0) if t_section else 1.0,
    }

    # ----- hardware -----
    hw_section = getattr(cfg, 'hardware', None)
    num_workers = getattr(hw_section, 'num_workers', 0) if hw_section else 0
    pin_memory = getattr(hw_section, 'pin_memory', False) if hw_section else False
    lr_decay = getattr(cfg.training, 'lr_decay', 0.96) if hasattr(cfg.training, 'lr_decay') else 0.96

    # ----- model architecture -----
    m_section = getattr(cfg, 'model', None)
    model_cfg = {}
    if m_section:
        model_cfg = {
            'use_ffn': getattr(m_section, 'use_ffn', False),
            'normalization': getattr(m_section, 'normalization', 'batch'),
            'tanh_clipping': getattr(m_section, 'tanh_clipping', 10.0),
            'pool_behavior': getattr(m_section, 'pool_behavior', 'static'),
        }

    params = OrderedDict(
        lr=[cfg.training.lr],
        hidden_node_dim=[cfg.model.hidden_node_dim],
        hidden_edge_dim=[cfg.model.hidden_edge_dim],
        epoch=[cfg.training.epochs],
        batch_size=[cfg.training.batch_size],
        conv_laysers=[cfg.model.conv_layers],
        entropy_value=[cfg.training.entropy_value],
        eps_clip=[cfg.training.eps_clip],
        timestep=[cfg.training.timestep],
        ppo_epoch=[cfg.training.ppo_epoch],
        data_size=[cfg.dataset.data_size],
        valid_size=[cfg.dataset.val_size],
    )
    runs = RunBuilder.get_runs(params)

    for (lr, hidden_node_dim, hidden_edge_dim, epoch, batch_size, conv_laysers,
         entropy_value, eps_clip, timestep, ppo_epoch, data_size, valid_size) in runs:

        print('lr', 'batch_size', 'hidden_node_dim', 'hidden_edge_dim', 'conv_laysers',
              'epoch,batch_size', 'entropy_value', 'eps_clip', 'timestep:', 'data_size', 'valid_size',
              lr, hidden_node_dim, hidden_edge_dim, epoch, batch_size, conv_laysers,
              entropy_value, eps_clip, timestep, data_size, valid_size)

        distribution = getattr(cfg.dataset, 'distribution', 'uniform')
        data_loder = creat_data(n_nodes, data_size, batch_size, distribution=distribution)
        valid_loder = creat_data(n_nodes, valid_size, batch_size, distribution=distribution)
        print('DATA CREATED/Problem size:', n_nodes)

        # ----- loss weights -----
        lw_section = getattr(cfg.training, 'loss_weights', None)
        loss_weights = {}
        if lw_section:
            loss_weights = {
                'policy': getattr(lw_section, 'policy', 1.0),
                'value': getattr(lw_section, 'value', 0.5),
                'entropy': getattr(lw_section, 'entropy', 0.01),
            }

        trainppo = TrainPPO(
            n_nodes * 2, greedy, lr, 3, hidden_node_dim, 1, hidden_edge_dim,
            epoch, batch_size, conv_laysers, entropy_value, eps_clip, timestep, ppo_epoch,
            n_nodes=n_nodes, num_workers=num_workers, pin_memory=pin_memory,
            lr_decay=lr_decay,
            quantum_cfg=quantum_cfg, dropout_cfg=dropout_cfg, model_cfg=model_cfg,
            seeds_cfg=seeds_cfg,
            temperature_config=temperature_config, verbose_level=verbose_level,
            device=device,
            loss_weights=loss_weights,
        )

        # Log parameter counts to MLflow (with quantum reduction % if applicable)
        if mlflow.active_run() is not None and hasattr(trainppo.agent, '_param_counts'):
            pc = trainppo.agent._param_counts
            policy_model = trainppo.agent.policy if hasattr(trainppo.agent, 'policy') else None
            log_parameter_counts_to_mlflow(pc, model=policy_model, quantum_enabled=bool(quantum_cfg))

        trainppo.run_train(data_loder, batch_size, valid_loder, run_dir,
                           cfg=cfg, profiler_enabled=profiler_enabled)
