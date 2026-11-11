"""
Model parameter counting and architecture summary utilities.

Provides dynamic parameter counting per component (encoder, decoder, critic)
and formatted model summaries compatible with verbose_level settings.
"""
import logging

model_logger = logging.getLogger(__name__)


def count_model_parameters(model):
    """
    Count trainable parameters by component for an Actor_critic model.

    Returns dict with keys:
        total, actor_total, encoder, decoder, critic, by_module
    """
    counts = {
        'total': 0,
        'actor_total': 0,
        'encoder': 0,
        'decoder': 0,
        'critic': 0,
        'by_module': {},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n = param.numel()
        counts['total'] += n

        if name.startswith('actor.encoder.'):
            counts['encoder'] += n
            counts['actor_total'] += n
        elif name.startswith('actor.decoder.'):
            counts['decoder'] += n
            counts['actor_total'] += n
        elif name.startswith('critic.'):
            counts['critic'] += n
        else:
            counts['actor_total'] += n

        # Top-level module grouping (e.g. actor.encoder.gat_layers.0.fc_node)
        parts = name.split('.')
        top = '.'.join(parts[:2]) if len(parts) >= 2 else parts[0]
        counts['by_module'][top] = counts['by_module'].get(top, 0) + n

    return counts


def get_model_summary(model, verbose_level=1):
    """
    Generate formatted model summary string.

    verbose_level 0: silent (returns empty string)
    verbose_level 1: compact (total + per-component)
    verbose_level 2: detailed (per-module breakdown)
    """
    if verbose_level < 1:
        return ""

    counts = count_model_parameters(model)
    total = counts['total']
    lines = []

    lines.append("============ MODEL ARCHITECTURE ============")
    lines.append(f"  TOTAL TRAINABLE PARAMS: {total:,}")
    lines.append(f"  Encoder params:  {counts['encoder']:,}"
                 f"  ({counts['encoder']/max(total,1)*100:.1f}%)")
    lines.append(f"  Decoder params:  {counts['decoder']:,}"
                 f"  ({counts['decoder']/max(total,1)*100:.1f}%)")
    lines.append(f"  Critic params:   {counts['critic']:,}"
                 f"  ({counts['critic']/max(total,1)*100:.1f}%)")

    if verbose_level >= 2 and counts['by_module']:
        lines.append("")
        lines.append("  Per-module breakdown:")
        for mod, n in sorted(counts['by_module'].items()):
            lines.append(f"    {mod}: {n:,}")

    lines.append("============================================")
    return "\n".join(lines)


def log_model_summary(model, verbose_level=1):
    """Log model summary via model_logger and return counts dict."""
    counts = count_model_parameters(model)
    summary = get_model_summary(model, verbose_level)
    if summary:
        for line in summary.split("\n"):
            model_logger.info(line)
    return counts
