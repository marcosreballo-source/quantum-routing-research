import math
import numpy as np
import torch
import time
from pathlib import Path
import mlflow
import torch.optim as optim
from collections import OrderedDict, namedtuple
from itertools import product
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader as TorchDataLoader

from DRL_Lei.VRP_Actor import Model
from DRL_Lei.creat_vrp import creat_data, reward1
from DRL_Lei.evaluate import evaluate
from DRL_Lei.utils.paths import make_eval_dir
from DRL_Lei.utils.profiler import (
    make_profiler, log_profiler_to_mlflow,
    log_device_info, log_quantum_summary,
    V_NONE, V_BASIC, V_MEDIUM, V_FULL,
)
from DRL_Lei.reinforce_baselines import (
    NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline,
    BaselineDataset,
)

max_grad_norm = 2


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


def rollout(model, dataset, batch_size, n_nodes, device):
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(bat, n_nodes * 2, device, True)
            cost = reward1(bat.x, cost.detach(), n_nodes)
        return cost.cpu()

    totall_cost = torch.cat([eval_model_bat(bat.to(device)) for bat in dataset], 0)
    return totall_cost


def clip_grad_norms(param_groups, max_norm=math.inf):
    """Clip gradient norms for all param groups and return norms."""
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'], max_norm if max_norm > 0 else math.inf, norm_type=2
        )
        for group in param_groups
    ]
    return grad_norms


def adv_normalize(adv):
    std = adv.std()
    assert std != 0. and not torch.isnan(std), 'Need nonzero std'
    n_advs = (adv - adv.mean()) / (adv.std() + 1e-8)
    return n_advs


def _collate_baseline_batch(samples):
    """Collate function for BaselineDataset.

    Each sample is {'data': pyg_Data, 'baseline': tensor(1,)}.
    We need to batch the PyG Data objects and stack the baseline values.
    """
    from torch_geometric.data import Batch
    data_list = [s['data'] for s in samples]
    baselines = torch.cat([s['baseline'] for s in samples], dim=0)
    return {'data': Batch.from_data_list(data_list), 'baseline': baselines}


def _build_baseline(cfg, model, val_loader, n_steps, n_nodes, device):
    """Build a baseline from config. Supports: none, exponential, rollout, critic."""
    bl_type = getattr(cfg.training, 'baseline', None) or 'rollout'
    bl_warmup = getattr(cfg.training, 'bl_warmup_epochs', 1)
    bl_alpha = getattr(cfg.training, 'bl_alpha', 0.05)
    exp_beta = getattr(cfg.training, 'exp_beta', 0.8)

    if bl_type == 'none':
        print("  [baseline] type=none (raw REINFORCE)")
        return NoBaseline()
    elif bl_type == 'exponential':
        print(f"  [baseline] type=exponential (beta={exp_beta})")
        return ExponentialBaseline(beta=exp_beta)
    elif bl_type == 'rollout':
        print(f"  [baseline] type=rollout (alpha={bl_alpha}, warmup={bl_warmup})")
        bl = RolloutBaseline(model, val_loader, n_steps=n_steps,
                             n_nodes=n_nodes, device=device, bl_alpha=bl_alpha)
        if bl_warmup > 0:
            return WarmupBaseline(bl, n_epochs=bl_warmup,
                                  warmup_exp_beta=exp_beta)
        return bl
    elif bl_type == 'critic':
        from DRL_Lei.critics import build_critic, _resolve_critic_type
        from DRL_Lei.utils.config import namespace_to_dict
        q_section = getattr(cfg, 'quantum', None)
        # Build full quantum_cfg dict for the factory
        q_dict = namespace_to_dict(q_section) if q_section and getattr(q_section, 'enabled', False) else {}
        crit_section = q_dict.get('critic', {})
        critic_type = _resolve_critic_type(crit_section)
        # Shared-encoder critics cannot work as standalone baselines in the
        # rollout trainer because VRP_Actor does not expose encoder embeddings.
        if critic_type in ('shared_conv1d', 'shared_fc'):
            raise ValueError(
                f"baseline=critic with critic type '{critic_type}' is not "
                f"supported in the rollout trainer. Shared-encoder critics "
                f"require the actor's encoder output, which VRP_Actor does "
                f"not expose.  Use own_encoder_conv1d, own_encoder_mlp, "
                f"or baseline=rollout instead."
            )
        d_section = getattr(cfg, 'dropout', None)
        dropout_dict = namespace_to_dict(d_section) if d_section else {}
        m_section = getattr(cfg, 'model', None)
        model_dict = namespace_to_dict(m_section) if m_section else {}
        hidden_dim = cfg.model.hidden_node_dim
        critic = build_critic(
            critic_type, hidden_dim,
            quantum_cfg=q_dict,
            quantum_critic_cfg=crit_section,
            dropout_cfg=dropout_dict,
            model_cfg=model_dict,
            input_node_dim=3,
            input_edge_dim=1,
            hidden_edge_dim=getattr(cfg.model, 'hidden_edge_dim', 16),
            conv_layers=getattr(cfg.model, 'conv_layers', 4),
        ).to(device)
        print(f"  [baseline] type=critic, critic_type={critic_type}")
        return CriticBaseline(critic)
    else:
        raise ValueError(f"Unknown baseline type: {bl_type}")


def train(cfg, run_dir: Path):
    class RunBuilder:
        @staticmethod
        def get_runs(params):
            Run = namedtuple('Run', params.keys())
            runs = []
            for v in product(*params.values()):
                runs.append(Run(*v))
            return runs

    # ----- runtime -----
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
    steps = n_nodes

    # ----- quantum_cfg -----
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
        _validate_quantum_layer_indices(
            cfg.model.conv_layers,
            quantum_cfg.get('encoder', {}),
            quantum_cfg.get('critic', {}),
        )

    # ----- dropout_cfg -----
    d_section = getattr(cfg, 'dropout', None)
    dropout_cfg = {}
    if d_section:
        dropout_cfg = {
            'gat': getattr(d_section, 'gat', 0.0),
            'attn': getattr(d_section, 'attn', 0.0),
            'attn_output': getattr(d_section, 'attn_output', 0.0),
            'decoder': getattr(d_section, 'decoder', 0.0),
        }

    # ----- model_cfg -----
    m_section = getattr(cfg, 'model', None)
    model_cfg = {}
    if m_section:
        model_cfg = {
            'use_ffn': getattr(m_section, 'use_ffn', False),
            'normalization': getattr(m_section, 'normalization', 'batch'),
            'tanh_clipping': getattr(m_section, 'tanh_clipping', 10.0),
            'pool_behavior': getattr(m_section, 'pool_behavior', 'static'),
        }

    # ----- temperature -----
    t_section = getattr(cfg, 'temperature', None)
    temperature = getattr(t_section, 'initial', 2.5) if t_section else 2.5

    params = OrderedDict(
        lr=[cfg.training.lr],
        lr_decay=[cfg.training.lr_decay],
        batch_size=[cfg.training.batch_size],
        hidden_node_dim=[cfg.model.hidden_node_dim],
        hidden_edge_dim=[cfg.model.hidden_edge_dim],
        conv_laysers=[cfg.model.conv_layers],
        data_size=[cfg.dataset.data_size],
        valid_size=[cfg.dataset.val_size],
    )
    runs = RunBuilder.get_runs(params)

    for lr, lr_decay, batch_size, hidden_node_dim, hidden_edge_dim, conv_laysers, data_size, valid_size in runs:
        print('lr', 'batch_size', 'hidden_node_dim', 'hidden_edge_dim', 'conv_laysers:',
              lr, batch_size, hidden_node_dim, hidden_edge_dim, conv_laysers)

        distribution = getattr(cfg.dataset, 'distribution', 'uniform')
        data_loder = creat_data(n_nodes, data_size, batch_size=batch_size, distribution=distribution)
        valid_loder = creat_data(n_nodes, valid_size, batch_size=batch_size, distribution=distribution)
        print('Data creation completed')

        actor = Model(3, hidden_node_dim, 1, hidden_edge_dim, conv_laysers=conv_laysers,
                      quantum_cfg=quantum_cfg, dropout_cfg=dropout_cfg,
                      model_cfg=model_cfg,
                      temperature=temperature).to(device)

        baseline = _build_baseline(cfg, actor, valid_loder, steps * 2, steps, device)

        # Multi-param-group optimizer: separate LR for critic baseline if present
        lr_critic = getattr(cfg.training, 'lr_critic', lr)
        bl_params = baseline.get_learnable_parameters()
        if bl_params:
            actor_optim = optim.Adam([
                {'params': actor.parameters(), 'lr': lr},
                {'params': bl_params, 'lr': lr_critic},
            ])
        else:
            actor_optim = optim.Adam(actor.parameters(), lr=lr)
        scheduler = LambdaLR(actor_optim, lr_lambda=lambda epoch: lr_decay ** epoch)

        costs = []
        from DRL_Lei.PPO_train import BestRouteTracker, log_parameter_counts_to_mlflow
        route_tracker = BestRouteTracker()

        # Log parameter counts (rollout model)
        if mlflow.active_run() is not None:
            from DRL_Lei.utils.model_utils import count_model_parameters
            pc = count_model_parameters(actor)
            q_enabled = bool(quantum_cfg)
            log_parameter_counts_to_mlflow(pc, model=actor, quantum_enabled=q_enabled)

        for epoch in range(cfg.training.epochs):
            print("epoch:", epoch, "------------------------------------------------")
            actor.train()
            profiler = make_profiler(profiler_enabled)

            # Per-epoch dataset wrapping (KOOL pattern): pre-compute baselines
            wrapped = baseline.wrap_dataset(data_loder)
            if isinstance(wrapped, BaselineDataset):
                # Wrapped dataset: iterate via standard DataLoader
                epoch_loader = TorchDataLoader(
                    wrapped, batch_size=batch_size, shuffle=False,
                    num_workers=0, collate_fn=_collate_baseline_batch,
                )
                use_precomputed_bl = True
            else:
                # No wrapping (NoBaseline, Exponential, CriticBaseline)
                epoch_loader = data_loder
                use_precomputed_bl = False

            times, losses, rewards = [], [], []
            epoch_start = time.time()
            start = epoch_start

            for batch_idx, batch in enumerate(epoch_loader):
                if use_precomputed_bl:
                    raw_batch, bl_precomputed = batch['data'], batch['baseline']
                    raw_batch = raw_batch.to(device)
                    bl_precomputed = bl_precomputed.to(device).view(-1)
                else:
                    raw_batch = batch
                    bl_precomputed = None
                    with profiler.measure("1_data_transfer"):
                        raw_batch = raw_batch.to(device)

                with profiler.measure("2_model_forward"):
                    tour_indices, tour_logp = actor(raw_batch, steps * 2, device)

                with profiler.measure("3_reward_compute"):
                    cost = reward1(raw_batch.x, tour_indices.detach(), n_nodes)

                with profiler.measure("4_baseline_eval"):
                    if bl_precomputed is not None:
                        bl_val, bl_loss = bl_precomputed, 0
                    else:
                        bl_val, bl_loss = baseline.eval(raw_batch, cost)

                with profiler.measure("5_update_backprop"):
                    advantage = cost - bl_val
                    if not advantage.ne(0).any():
                        print("advantage==0.")
                        continue
                    advantage = adv_normalize(advantage)

                    reinforce_loss = (advantage.detach() * tour_logp).mean()
                    loss = reinforce_loss + bl_loss

                    actor_optim.zero_grad()
                    loss.backward()
                    clip_grad_norms(actor_optim.param_groups, max_grad_norm)
                    actor_optim.step()

                rewards.append(cost.mean().item())
                losses.append(loss.item())

                if mlflow.active_run() is not None:
                    global_step = epoch * len(data_loder) + batch_idx
                    mlflow.log_metric("rollout/baseline_mean",
                                      bl_val.mean().item() if torch.is_tensor(bl_val) else float(bl_val),
                                      step=global_step)

                log_step = 200
                if (batch_idx + 1) % log_step == 0:
                    end = time.time()
                    times.append(end - start)
                    start = end
                    mean_loss = np.mean(losses[-log_step:])
                    mean_reward = np.mean(rewards[-log_step:])
                    print('  Batch %d/%d, reward: %2.3f, loss: %2.4f, took: %2.4fs' %
                          (batch_idx, len(data_loder), mean_reward, mean_loss, times[-1]))
                    if mlflow.active_run() is not None:
                        mlflow.log_metric("train/mean_reward", mean_reward, step=global_step)
                        mlflow.log_metric("train/mean_loss", mean_loss, step=global_step)

            # LR decay per epoch (not per batch — fixed bug)
            scheduler.step()

            baseline.epoch_callback(actor, epoch)

            # Profiler metrics
            if profiler_enabled and profiler.totals:
                log_profiler_to_mlflow(profiler, step=epoch, prefix="profiler")
                if verbose_level >= V_BASIC:
                    print(profiler.summary(prefix=f"  [Epoch {epoch}] "))

            # Checkpointing
            save_path = run_dir / 'checkpoints' / 'last.pt'
            torch.save({
                'model': actor.state_dict(),
                'optimizer': actor_optim.state_dict(),
                'baseline': baseline.state_dict(),
                'epoch': epoch,
            }, save_path)

            if epoch % 10 == 0 or epoch == cfg.training.epochs - 1:
                if epoch < cfg.training.epochs - 1:
                    save_path = run_dir / 'checkpoints' / f'actor_{epoch}.pt'
                    torch.save(actor.state_dict(), save_path)
                elif epoch == cfg.training.epochs - 1:
                    save_path = run_dir / 'checkpoints' / 'final.pt'
                    torch.save(actor.state_dict(), save_path)
                    if mlflow.active_run() is not None:
                        mlflow.log_artifact(str(save_path))
                    route_tracker.log_final_artifacts()
                    # Evaluation guard
                    eval_cfg = getattr(cfg, 'evaluation', None)
                    eval_enabled = getattr(eval_cfg, 'enabled', True) if eval_cfg else True
                    if eval_enabled:
                        eval_dir = make_eval_dir(save_path)
                        evaluate(cfg, save_path, eval_dir)
                    else:
                        print("  [eval] Evaluation disabled via config (evaluation.enabled=false)")

            cost_val = rollout(actor, valid_loder, batch_size, steps, device)
            cost_mean = cost_val.mean().item()
            costs.append(cost_mean)
            route_tracker.update(epoch, cost_mean)

            print(f'VRP{n_nodes} / Average distance: {cost_mean:.4f}')
            if mlflow.active_run() is not None:
                mlflow.log_metric("val/mean_cost", cost_mean, step=epoch)
                mlflow.log_metric("val/best_mean_cost", route_tracker.best_cost, step=epoch)
                mlflow.log_metric("time/epoch_seconds", time.time() - epoch_start, step=epoch)
                mlflow.log_metric("opt/lr", actor_optim.param_groups[0]["lr"], step=epoch)
            print(costs)
