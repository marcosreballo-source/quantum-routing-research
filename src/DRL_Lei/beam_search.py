"""
Beam search decoder for VRP-QGAT.

Operates at the per-step level of Decoder1, maintaining multiple beams
of partial solutions and selecting the top-k at each expansion.
"""
import torch
from DRL_Lei.vrpUpdate import update_mask, update_state


def beam_search_decode(model, datas, n_steps, device, beam_width=10):
    """
    Beam search decoding for the VRP Actor model.

    Args:
        model: VRP_Actor.Model (encoder + decoder)
        datas: PyG batch
        n_steps: max decode steps
        device: torch device
        beam_width: number of beams to track

    Returns:
        best_actions: (batch, seq_len) best tour indices
        best_cost: None (caller should compute cost)
    """
    model.eval()
    decoder = model.decoder
    B = datas.num_graphs  # batch size
    N = datas.x.size(0) // B  # nodes per graph

    # Encode once
    with torch.no_grad():
        encoder_out = model.encoder(datas)  # (B, N, H)
    pooled_orig = encoder_out.mean(dim=1)   # (B, H)
    demand = datas.demand.view(B, N)
    capcity_val = datas.capcity[0].item()
    capcity = datas.capcity.view(B, -1)[:, :1]  # (B, 1)

    H = encoder_out.size(2)
    W = beam_width

    # Expand everything to (B*W, ...)
    def expand_beam(t, dim=0):
        """Repeat tensor along dim for beam_width copies."""
        shape = list(t.shape)
        shape.insert(dim + 1, W)
        return t.unsqueeze(dim + 1).expand(shape).reshape(
            [-1] + list(t.shape[dim + 1:]) if dim == 0
            else list(t.shape[:dim]) + [-1] + list(t.shape[dim + 1:])
        )

    # Start: single beam per batch item
    enc = encoder_out  # (B, N, H)
    pool = pooled_orig  # (B, H)
    mask1 = enc.new_zeros(B, N)
    mask = enc.new_zeros(B, N)
    dynamic_cap = capcity.expand(B, 1).clone()  # (B, 1)
    index = torch.zeros(B, device=device, dtype=torch.long)
    _input = enc[:, 0, :]  # (B, H) - start at depot

    # Beam storage
    beam_log_p = torch.zeros(B, 1, device=device)  # (B, cur_beams)
    beam_actions = []  # list of (B, cur_beams) tensors per step
    # State per beam: all tensors have shape (B, cur_beams, ...)
    # We flatten to (B*cur_beams, ...) for decoder ops then reshape

    all_actions_list = []  # will store per-step selected actions

    for step in range(n_steps):
        cur_beams = beam_log_p.size(1)
        BW = B * cur_beams

        # Flatten beam dimension into batch
        enc_flat = enc.unsqueeze(1).expand(B, cur_beams, N, H).reshape(BW, N, H) if cur_beams > 1 else enc
        pool_flat = pool.unsqueeze(1).expand(B, cur_beams, H).reshape(BW, H) if step == 0 and cur_beams == 1 else pool.reshape(BW, H) if cur_beams > 1 else pool
        _input_flat = _input.reshape(BW, H)
        dynamic_cap_flat = dynamic_cap.reshape(BW, 1)
        mask_flat = mask.reshape(BW, N)
        mask1_flat = mask1.reshape(BW, N)
        demands_flat = demand.unsqueeze(1).expand(B, cur_beams, N).reshape(BW, N) if cur_beams > 1 else demand

        # Check if all done
        if not mask1_flat[:, 1:].eq(0).any():
            break

        # Decoder step
        decoder_input = torch.cat([_input_flat, dynamic_cap_flat], -1)
        decoder_input = decoder.fc(decoder_input)
        pool_flat = decoder.fc1(pool_flat)
        decoder_input = decoder_input + pool_flat

        if step == 0:
            index_flat = index
            mask_flat, mask1_flat = update_mask(
                demands_flat, dynamic_cap_flat, index_flat.unsqueeze(-1), mask1_flat, step)

        p = decoder.prob(decoder_input, enc_flat, mask_flat)  # (BW, N)

        # For beam search use log probs
        log_p = torch.log(p.clamp(min=1e-8))  # (BW, N)
        log_p = log_p.view(B, cur_beams, N)

        # Expand: score of each candidate = beam_score + log_p
        candidate_scores = beam_log_p.unsqueeze(-1) + log_p  # (B, cur_beams, N)
        candidate_scores = candidate_scores.view(B, -1)  # (B, cur_beams * N)

        # Select top-W
        k = min(W, candidate_scores.size(1))
        topk_scores, topk_idx = candidate_scores.topk(k, dim=-1)  # (B, k)

        # Decode parent beam and action from flat index
        parent_beam = topk_idx // N  # (B, k) which beam
        action = topk_idx % N        # (B, k) which node

        beam_log_p = topk_scores  # (B, k)
        new_beams = k

        # Gather states from parent beams
        # pool: (B, cur_beams, H) -> select by parent_beam -> (B, new_beams, H)
        pool_shaped = pool_flat.view(B, cur_beams, H)
        pool = torch.gather(pool_shaped, 1, parent_beam.unsqueeze(-1).expand(B, new_beams, H))

        mask1_shaped = mask1_flat.view(B, cur_beams, N)
        mask1 = torch.gather(mask1_shaped, 1, parent_beam.unsqueeze(-1).expand(B, new_beams, N))

        dynamic_cap_shaped = dynamic_cap_flat.view(B, cur_beams, 1)
        dynamic_cap = torch.gather(dynamic_cap_shaped, 1, parent_beam.unsqueeze(-1).expand(B, new_beams, 1))

        # Store actions
        beam_actions.append(action)  # (B, new_beams)

        # Also reindex previous actions by parent
        new_all_actions = []
        for prev_act in all_actions_list:
            # prev_act: (B, prev_beams)
            prev_beams = prev_act.size(1)
            if prev_beams < cur_beams:
                # Pad if needed (shouldn't normally happen after first expansion)
                pad = prev_act.new_zeros(B, cur_beams - prev_beams)
                prev_act = torch.cat([prev_act, pad], dim=1)
            new_all_actions.append(torch.gather(prev_act, 1, parent_beam))
        new_all_actions.append(action)
        all_actions_list = new_all_actions

        # Update states for new beams (flatten to BW for update functions)
        BW_new = B * new_beams
        action_flat = action.reshape(BW_new)
        demands_new = demand.unsqueeze(1).expand(B, new_beams, N).reshape(BW_new, N)
        dynamic_cap_flat = dynamic_cap.reshape(BW_new, 1)
        mask1_flat = mask1.reshape(BW_new, N)

        dynamic_cap_flat = update_state(
            demands_new, dynamic_cap_flat, action_flat.unsqueeze(-1), capcity_val)
        dynamic_cap = dynamic_cap_flat.view(B, new_beams, 1)

        mask_flat = mask.reshape(BW_new, N) if mask.dim() == 3 else mask.new_zeros(BW_new, N)
        mask_flat, mask1_flat = update_mask(
            demands_new, dynamic_cap_flat, action_flat.unsqueeze(-1), mask1_flat, step + 1)
        mask = mask_flat.view(B, new_beams, N)
        mask1 = mask1_flat.view(B, new_beams, N)

        # Update _input for next step
        enc_expanded = enc.unsqueeze(1).expand(B, new_beams, N, H)
        action_idx = action.unsqueeze(-1).unsqueeze(-1).expand(B, new_beams, 1, H)
        _input = torch.gather(enc_expanded, 2, action_idx).squeeze(2)  # (B, new_beams, H)

    # Select best beam per batch
    if len(all_actions_list) == 0:
        return torch.zeros(B, 0, device=device, dtype=torch.long), None

    best_beam = beam_log_p.argmax(dim=1)  # (B,)
    best_actions = torch.stack(all_actions_list, dim=-1)  # (B, W, steps)
    best_actions = best_actions[torch.arange(B, device=device), best_beam]  # (B, steps)

    return best_actions, None
