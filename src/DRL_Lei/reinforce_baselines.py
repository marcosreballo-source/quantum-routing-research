"""
REINFORCE baselines for VRP-QGAT.

Six baseline types ported from KOOL (attention-learn-to-route):
  NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
  + BaselineDataset (helper for per-instance rollout baselines)

Adapted for PyTorch Geometric DataLoader / VRP_Actor.Model interface.
"""
import copy
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.stats import ttest_rel

from DRL_Lei.creat_vrp import reward1


def _greedy_rollout(model, data_loader, n_steps, n_nodes, device):
    """Evaluate model greedily on a DataLoader, returning (batch,) costs."""
    model.eval()
    costs = []
    start = time.time()
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            actions, _ = model(batch, n_steps, device, greedy=True)
            cost = reward1(batch.x, actions.detach(), n_nodes)
            costs.append(cost.cpu())
    elapsed = time.time() - start
    print(f"  [baseline] _greedy_rollout: {len(data_loader)} batches, {elapsed:.2f}s")
    return torch.cat(costs, 0)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
class Baseline:

    def wrap_dataset(self, dataset):
        return dataset

    def unwrap_batch(self, batch):
        return batch, None

    def eval(self, x, c):
        raise NotImplementedError

    def get_learnable_parameters(self):
        return []

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


# ---------------------------------------------------------------------------
# Warmup: convex blend of warmup_baseline and inner baseline
# ---------------------------------------------------------------------------
class WarmupBaseline(Baseline):

    def __init__(self, baseline, n_epochs=1, warmup_exp_beta=0.8):
        self.baseline = baseline
        assert n_epochs > 0
        self.warmup_baseline = ExponentialBaseline(warmup_exp_beta)
        self.alpha = 0
        self.n_epochs = n_epochs

    def wrap_dataset(self, dataset):
        if self.alpha > 0:
            return self.baseline.wrap_dataset(dataset)
        return self.warmup_baseline.wrap_dataset(dataset)

    def unwrap_batch(self, batch):
        if self.alpha > 0:
            return self.baseline.unwrap_batch(batch)
        return self.warmup_baseline.unwrap_batch(batch)

    def eval(self, x, c):
        if self.alpha == 1:
            return self.baseline.eval(x, c)
        if self.alpha == 0:
            return self.warmup_baseline.eval(x, c)
        v, l = self.baseline.eval(x, c)
        vw, lw = self.warmup_baseline.eval(x, c)
        return self.alpha * v + (1 - self.alpha) * vw, self.alpha * l + (1 - self.alpha) * lw

    def epoch_callback(self, model, epoch):
        self.baseline.epoch_callback(model, epoch)
        if epoch < self.n_epochs:
            self.alpha = (epoch + 1) / float(self.n_epochs)
            print(f"  [warmup] Set warmup alpha = {self.alpha}")

    def get_learnable_parameters(self):
        # Delegate to inner baseline (e.g. CriticBaseline)
        return self.baseline.get_learnable_parameters()

    def state_dict(self):
        return self.baseline.state_dict()

    def load_state_dict(self, state_dict):
        self.baseline.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# No baseline (raw REINFORCE)
# ---------------------------------------------------------------------------
class NoBaseline(Baseline):

    def eval(self, x, c):
        return 0, 0


# ---------------------------------------------------------------------------
# Exponential moving average
# ---------------------------------------------------------------------------
class ExponentialBaseline(Baseline):

    def __init__(self, beta=0.8):
        self.beta = beta
        self.v = None

    def eval(self, x, c):
        if self.v is None:
            v = c.mean()
        else:
            v = self.beta * self.v + (1.0 - self.beta) * c.mean()
        self.v = v.detach()
        return self.v, 0

    def state_dict(self):
        return {'v': self.v}

    def load_state_dict(self, state_dict):
        self.v = state_dict.get('v')


# ---------------------------------------------------------------------------
# Critic baseline (learned value function)
# ---------------------------------------------------------------------------
class CriticBaseline(Baseline):

    def __init__(self, critic):
        self.critic = critic

    def eval(self, x, c):
        v = self.critic(x)
        return v.detach(), F.mse_loss(v, c.detach())

    def get_learnable_parameters(self):
        return list(self.critic.parameters())

    def state_dict(self):
        return {'critic': self.critic.state_dict()}

    def load_state_dict(self, state_dict):
        sd = state_dict.get('critic', {})
        if not isinstance(sd, dict):
            sd = sd.state_dict()
        self.critic.load_state_dict({**self.critic.state_dict(), **sd})


# ---------------------------------------------------------------------------
# Rollout baseline with t-test challenge
# ---------------------------------------------------------------------------
class RolloutBaseline(Baseline):

    def __init__(self, model, val_loader, n_steps, n_nodes, device,
                 bl_alpha=0.05, epoch=0):
        self.n_steps = n_steps
        self.n_nodes = n_nodes
        self.device = device
        self.bl_alpha = bl_alpha
        self._update_model(model, val_loader, epoch)

    def _update_model(self, model, val_loader, epoch):
        self.model = copy.deepcopy(model)
        self.val_loader = val_loader
        self.bl_vals = _greedy_rollout(
            self.model, val_loader, self.n_steps, self.n_nodes, self.device
        ).numpy()
        self.mean = self.bl_vals.mean()
        self.epoch = epoch

    def wrap_dataset(self, dataset):
        """Pre-compute per-instance baseline values for a DataLoader.

        Accepts a PyG DataLoader. Returns a BaselineDataset that pairs each
        training instance with its greedy rollout cost under the baseline model.
        This is the KOOL pattern for strong per-instance variance reduction.
        """
        print("  [rollout] Evaluating baseline on training dataset...")
        bl_costs = _greedy_rollout(
            self.model, dataset, self.n_steps, self.n_nodes, self.device
        )  # shape (total_samples,)
        # Extract the underlying list of Data objects from the PyG DataLoader
        raw_dataset = dataset.dataset
        return BaselineDataset(raw_dataset, bl_costs.view(-1, 1))

    def unwrap_batch(self, batch):
        return batch['data'], batch['baseline'].view(-1)

    def eval(self, x, c):
        """Fallback when pre-computed bl_val is not available.

        Runs greedy rollout of the baseline model on the current batch input.
        Prefer using wrap_dataset + unwrap_batch for per-instance baselines.
        """
        with torch.no_grad():
            self.model.eval()
        return torch.tensor(self.mean, device=c.device), 0

    def epoch_callback(self, model, epoch):
        cb_start = time.time()
        candidate_vals = _greedy_rollout(
            model, self.val_loader, self.n_steps, self.n_nodes, self.device
        ).numpy()
        candidate_mean = candidate_vals.mean()

        if candidate_mean - self.mean < 0:
            t, p = ttest_rel(candidate_vals, self.bl_vals)
            p_val = p / 2  # one-sided
            if t < 0 and p_val < self.bl_alpha:
                print(f'  [rollout] Baseline updated: {self.mean:.4f} -> {candidate_mean:.4f} (p={p_val:.4f})')
                self._update_model(model, self.val_loader, epoch)
            else:
                print(f"  [rollout] Candidate not significant: p={p_val:.4f} >= alpha={self.bl_alpha}")
        else:
            print(f"  [rollout] Candidate not better: {candidate_mean:.4f} >= {self.mean:.4f}")
        print(f"  [rollout] epoch_callback: {time.time() - cb_start:.2f}s (candidate_mean={candidate_mean:.4f})")

    def state_dict(self):
        return {
            'model_state_dict': self.model.state_dict(),
            'epoch': self.epoch,
        }

    def load_state_dict(self, state_dict):
        load_model = copy.deepcopy(self.model)
        load_model.load_state_dict(state_dict['model_state_dict'])
        self._update_model(load_model, self.val_loader, state_dict.get('epoch', 0))


# ---------------------------------------------------------------------------
# Dataset wrapper for pre-computed per-instance baselines
# ---------------------------------------------------------------------------
class BaselineDataset(Dataset):
    """Wraps a DataLoader's dataset with pre-computed baseline values.

    Used by RolloutBaseline.wrap_dataset() to pair each training instance
    with its greedy rollout cost under the current baseline model.
    """

    def __init__(self, dataset, baseline):
        super().__init__()
        self.dataset = dataset
        self.baseline = baseline
        assert len(self.dataset) == len(self.baseline)

    def __getitem__(self, item):
        return {
            'data': self.dataset[item],
            'baseline': self.baseline[item],
        }

    def __len__(self):
        return len(self.dataset)
