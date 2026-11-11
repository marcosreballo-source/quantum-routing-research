"""Training method registry.

Usage in main.py:
    from DRL_Lei.training import TRAINERS
    TRAINERS[cfg.training.method](cfg, run_dir)
"""
from DRL_Lei.PPO_train import train as _train_ppo
from DRL_Lei.VRP_Rollout_train import train as _train_rollout

TRAINERS = {
    'ppo': _train_ppo,
    'rollout': _train_rollout,
}

REQUIRED_PARAMS = {
    'ppo': ['policy_mode', 'lr', 'batch_size', 'epochs', 'timestep',
            'ppo_epoch', 'entropy_value', 'eps_clip'],
    'rollout': ['lr', 'batch_size', 'epochs', 'lr_decay', 'baseline'],
}
