import yaml
import time
import torch
import mlflow
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from DRL_Lei.creat_vrp import creat_data, reward1
from DRL_Lei.beam_search import beam_search_decode
from DRL_Lei.VRP_Actor import Model as RolloutModel
from DRL_Lei.VRP_PPO_Model import Actor_critic as PPOModel
from DRL_Lei.utils.paths import get_path_suffix
from DRL_Lei.utils.profiler import (
    make_profiler, log_profiler_to_mlflow, log_device_info, log_quantum_summary,
    V_NONE, V_BASIC, V_MEDIUM, V_FULL,
)


def _build_quantum_cfg(cfg):
    """Extract granular quantum_cfg dict from config namespace."""
    q = getattr(cfg, 'quantum', None)
    if q is None or not getattr(q, 'enabled', False):
        return {}
    enc = getattr(q, 'encoder', None)
    crit = getattr(q, 'critic', None)
    return {
        'n_qubits_actor': getattr(q, 'n_qubits_actor', 4),
        'n_layers_quantum_actor': getattr(q, 'n_layers_quantum_actor', 4),
        'n_qubits_critic': getattr(q, 'n_qubits_critic', 4),
        'n_layers_quantum_critic': getattr(q, 'n_layers_quantum_critic', 2),
        'backend': getattr(q, 'backend', 'lightning.qubit'),
        'diff_method': getattr(q, 'diff_method', 'best'),
        'encoder': {
            'fc_node': getattr(enc, 'fc_node', False) if enc else False,
            'fc_edge': getattr(enc, 'fc_edge', False) if enc else False,
            'gat_layers': list(getattr(enc, 'gat_layers', [])) if enc else [],
        },
        'critic': {
            'layers': list(getattr(crit, 'layers', [])) if crit else [],
            'type': getattr(crit, 'type', 'shared_conv1d') if crit else 'shared_conv1d',
            'encoder': {
                'fc_node': getattr(getattr(crit, 'encoder', None), 'fc_node', False) if crit else False,
                'fc_edge': getattr(getattr(crit, 'encoder', None), 'fc_edge', False) if crit else False,
                'gat_layers': list(getattr(getattr(crit, 'encoder', None), 'gat_layers', [])) if crit else [],
            } if crit and hasattr(crit, 'encoder') else {},
        },
    }


def _build_dropout_cfg(cfg):
    d = getattr(cfg, 'dropout', None)
    if d is None:
        return {}
    return {
        'gat': getattr(d, 'gat', 0.0),
        'attn': getattr(d, 'attn', 0.0),
        'attn_output': getattr(d, 'attn_output', 0.0),
        'decoder': getattr(d, 'decoder', 0.0),
        'critic': getattr(d, 'critic', 0.0),
    }


def load_model(cfg, checkpoint_path: Path):
    runtime = getattr(cfg, 'runtime', None)
    device = getattr(runtime, 'device', None) if runtime else None
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_nodes = cfg.problem.n_nodes + 1
    quantum_cfg = _build_quantum_cfg(cfg)
    dropout_cfg = _build_dropout_cfg(cfg)

    t_section = getattr(cfg, 'temperature', None)
    temperature = getattr(t_section, 'initial', 2.5) if t_section else 2.5

    if cfg.training.method == 'rollout':
        model = RolloutModel(
            3, cfg.model.hidden_node_dim, 1,
            cfg.model.hidden_edge_dim,
            conv_laysers=cfg.model.conv_layers,
            quantum_cfg=quantum_cfg, dropout_cfg=dropout_cfg,
            temperature=temperature,
        ).to(device)
    else:  # ppo
        model = PPOModel(
            3, cfg.model.hidden_node_dim, 1,
            cfg.model.hidden_edge_dim,
            cfg.model.conv_layers,
            quantum_cfg=quantum_cfg, dropout_cfg=dropout_cfg,
            temperature=temperature,
        ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model, n_nodes, device


def _call_model(model, bat, n_nodes, greedy, T, cfg, device):
    with torch.no_grad():
        if cfg.training.method == 'rollout':
            actions, _ = model(bat, n_nodes * 2, device, greedy=greedy, temperature=T)
        else:  # ppo
            actions, _ = model.act(bat, 0, n_nodes * 2,
                                   bat.num_graphs, device, greedy, False, T)
    return actions


def evaluate(cfg, checkpoint_path: Path, eval_dir: Path):
    eval_start = time.time()
    runtime = getattr(cfg, 'runtime', None)
    verbose = getattr(runtime, 'verbose_level', 1) if runtime else 1
    profiler_enabled = getattr(runtime, 'profiler_enabled', False) if runtime else False
    profiler = make_profiler(profiler_enabled)

    n_nodes = cfg.problem.n_nodes + 1
    temperatures = {int(k): v for k, v in
                    vars(cfg.evaluation.temperatures).items()}
    T = temperatures.get(n_nodes)
    if T is None:
        # Fallback: nearest key or default 1.5
        if temperatures:
            nearest_key = min(temperatures.keys(), key=lambda k: abs(k - n_nodes))
            T = temperatures[nearest_key]
            print(f"  [eval] WARNING: No temperature for n_nodes={n_nodes}, "
                  f"using T={T} from nearest key={nearest_key}")
        else:
            T = 1.5
            print(f"  [eval] WARNING: No temperatures configured, using default T={T}")
    n_samples = cfg.evaluation.n_samples
    eval_size = cfg.evaluation.eval_size
    eval_modes = getattr(cfg.evaluation, 'modes', ['greedy', 'sampling'])

    if verbose >= V_BASIC:
        print(f"\n{'='*60}")
        print(f"EVALUATION: CVRP{cfg.problem.n_nodes}, checkpoint={checkpoint_path.name}")
        print(f"  eval_size={eval_size}, n_samples={n_samples}, T={T}")

    with profiler.measure("load_model"):
        model, n_nodes, device = load_model(cfg, checkpoint_path)
    if verbose >= V_BASIC:
        log_device_info(device, verbose)
        log_quantum_summary(cfg, verbose)

    with profiler.measure("create_dataset"):
        distribution = getattr(cfg.dataset, 'distribution', 'uniform')
        dataset = creat_data(n_nodes, eval_size, batch_size=cfg.training.batch_size, distribution=distribution)

    # --- Greedy ---
    greedy_all = None
    greedy_mean = None
    if 'greedy' in eval_modes:
        with profiler.measure("greedy_eval"):
            greedy_costs = []
            for bat_idx, bat in enumerate(dataset):
                bat = bat.to(device)
                actions = _call_model(model, bat, n_nodes, greedy=True, T=1, cfg=cfg, device=device)
                greedy_costs.append(reward1(bat.x, actions.detach(), n_nodes).cpu())
                if verbose >= V_FULL and (bat_idx + 1) % 10 == 0:
                    print(f"  [greedy] batch {bat_idx+1}/{len(dataset)}")
        greedy_all = torch.cat(greedy_costs)
        greedy_mean = greedy_all.mean().item()
        if verbose >= V_BASIC:
            print(f"  Greedy mean: {greedy_mean:.4f}")
    else:
        if verbose >= V_BASIC:
            print("  [eval] Greedy mode disabled")

    # --- Sampling: N draws per instance, keep best ---
    sampling_all = None
    sampling_mean = None
    if 'sampling' in eval_modes:
        with profiler.measure("sampling_eval"):
            sampling_costs = []
            for bat_idx, bat in enumerate(dataset):
                bat = bat.to(device)
                all_costs = []
                for _ in range(n_samples):
                    actions = _call_model(model, bat, n_nodes,
                                          greedy=False, T=T, cfg=cfg, device=device)
                    all_costs.append(reward1(bat.x, actions.detach(), n_nodes))
                best = torch.stack(all_costs, dim=0).min(dim=0).values
                sampling_costs.append(best.cpu())
                if verbose >= V_FULL and (bat_idx + 1) % 10 == 0:
                    print(f"  [sampling] batch {bat_idx+1}/{len(dataset)}")
        sampling_all = torch.cat(sampling_costs)
        sampling_mean = sampling_all.mean().item()
        if verbose >= V_BASIC:
            print(f"  Sampling mean (n={n_samples}, T={T}): {sampling_mean:.4f}")
    else:
        if verbose >= V_BASIC:
            print("  [eval] Sampling mode disabled")

    # --- Beam search (rollout models only) ---
    beam_mean = None
    beam_all = None
    beam_width = getattr(cfg.evaluation, 'beam_width', 0)
    if beam_width > 0 and cfg.training.method == 'rollout' and 'beam' in eval_modes:
        with profiler.measure("beam_eval"):
            beam_costs = []
            for bat_idx, bat in enumerate(dataset):
                bat = bat.to(device)
                actions, _ = beam_search_decode(model, bat, n_nodes * 2, device,
                                                beam_width=beam_width)
                beam_costs.append(reward1(bat.x, actions.detach(), n_nodes).cpu())
                if verbose >= V_FULL and (bat_idx + 1) % 10 == 0:
                    print(f"  [beam] batch {bat_idx+1}/{len(dataset)}")
        beam_all = torch.cat(beam_costs)
        beam_mean = beam_all.mean().item()
        if verbose >= V_BASIC:
            print(f"  Beam mean (w={beam_width}): {beam_mean:.4f}")

    print(f"CVRP{cfg.problem.n_nodes}"
          + (f" | Greedy: {greedy_mean:.4f}" if greedy_mean is not None else "")
          + (f" | Sampling (n={n_samples}, T={T}): {sampling_mean:.4f}" if sampling_mean is not None else "")
          + (f" | Beam (w={beam_width}): {beam_mean:.4f}" if beam_mean is not None else ""))

    if mlflow.active_run() is not None:
        if greedy_all is not None:
            mlflow.log_metric("eval/greedy_mean_cost", greedy_mean)
            mlflow.log_metric("eval/greedy_std_cost", greedy_all.std().item())
            mlflow.log_metric("eval/greedy_min_cost", greedy_all.min().item())
            mlflow.log_metric("eval/greedy_max_cost", greedy_all.max().item())
        if sampling_all is not None:
            mlflow.log_metric("eval/sampling_mean_cost", sampling_mean)
            mlflow.log_metric("eval/sampling_std_cost", sampling_all.std().item())
            mlflow.log_metric("eval/sampling_min_cost", sampling_all.min().item())
            mlflow.log_metric("eval/sampling_max_cost", sampling_all.max().item())
        mlflow.log_metric("eval/temperature", T)
        mlflow.log_metric("eval/n_samples", n_samples)
        if beam_mean is not None:
            mlflow.log_metric("eval/beam_mean_cost", beam_mean)
            mlflow.log_metric("eval/beam_std_cost", beam_all.std().item())
            mlflow.log_metric("eval/beam_width", beam_width)
            mlflow.log_metric("eval/beam_width", beam_width)

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(
                                f"Greedy (mean={greedy_mean:.4f})" if greedy_mean is not None else "Greedy (skipped)",
                                f"Sampling n={n_samples}, T={T} (mean={sampling_mean:.4f})" if sampling_mean is not None else "Sampling (skipped)"
                            ))
        if greedy_all is not None:
            fig.add_trace(go.Histogram(x=greedy_all.numpy(), nbinsx=30, name="Greedy"), row=1, col=1)
        if sampling_all is not None:
            fig.add_trace(go.Histogram(x=sampling_all.numpy(), nbinsx=30, name="Sampling"), row=1, col=2)
        fig.update_layout(title=f"CVRP{cfg.problem.n_nodes} cost distribution")
        mlflow.log_figure(fig, "eval/cost_histogram.html")

    results = {
        "greedy_mean_cost": greedy_mean,
        "greedy_std_cost": greedy_all.std().item() if greedy_all is not None else None,
        "greedy_min_cost": greedy_all.min().item() if greedy_all is not None else None,
        "greedy_max_cost": greedy_all.max().item() if greedy_all is not None else None,
        "greedy_costs": greedy_all.tolist() if greedy_all is not None else None,
        "sampling_mean_cost": sampling_mean,
        "sampling_std_cost": sampling_all.std().item() if sampling_all is not None else None,
        "sampling_min_cost": sampling_all.min().item() if sampling_all is not None else None,
        "sampling_max_cost": sampling_all.max().item() if sampling_all is not None else None,
        "sampling_costs": sampling_all.tolist() if sampling_all is not None else None,
        "n_samples": n_samples,
        "temperature": T,
        "beam_width": beam_width,
        "beam_mean_cost": beam_mean,
        "eval_size": eval_size,
        "eval_modes": eval_modes,
        "checkpoint": str(checkpoint_path),
    }
    with open(eval_dir / f"results_{get_path_suffix()}.yaml", "w") as f:
        yaml.safe_dump(results, f)

    # Profiler summary
    total_eval_time = time.time() - eval_start
    if profiler_enabled and profiler.totals:
        log_profiler_to_mlflow(profiler, step=0, prefix="eval_profiler")
        if verbose >= V_BASIC:
            print(profiler.summary(prefix="  [eval] "))
    if verbose >= V_BASIC:
        print(f"  Total evaluation time: {total_eval_time:.2f}s")
        print(f"{'='*60}\n")
    if mlflow.active_run() is not None:
        mlflow.log_metric("eval/total_time_seconds", total_eval_time)

    return greedy_mean, sampling_mean
