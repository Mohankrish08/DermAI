# ==========================================
# secure_aggregation.py
# Secure Aggregation for Federated Learning
# ==========================================
#
# Implements a simplified secure aggregation protocol
# inspired by Bonawitz et al. (2017). Each client masks
# its model update with pairwise random masks so the
# server never sees raw individual updates.
# ==========================================

import copy
import torch
import hashlib
import os


# ==========================================
# 1. Generate Pairwise Masks
# ==========================================

def _generate_seed(client_i, client_j, round_num):
    """
    Deterministic seed derived from client pair + round.
    Both parties can independently compute the same seed.
    """
    raw = f"pair-{min(client_i, client_j)}-{max(client_i, client_j)}-round-{round_num}"
    return int(hashlib.sha256(raw.encode()).hexdigest(), 16) % (2**32)


def _generate_mask(state_dict, seed, device="cpu"):
    """
    Generate a pseudo-random mask tensor dict from a seed.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    mask = {}
    for key, param in state_dict.items():
        mask[key] = torch.randn(
            param.shape,
            generator=gen,
            device=device
        ) * 0.001          # small scale to avoid numerical issues
    return mask


# ==========================================
# 2. Mask / Unmask Client Weights
# ==========================================

def mask_client_weights(client_id, state_dict, num_clients, round_num, device="cpu"):
    """
    Mask a single client's weights using pairwise masks.
    For every other client j:
        if client_id < j  => ADD mask
        if client_id > j  => SUBTRACT mask
    After all clients' masked weights are summed, masks cancel out.
    """
    masked = copy.deepcopy(state_dict)

    for j in range(num_clients):
        if j == client_id:
            continue

        seed = _generate_seed(client_id, j, round_num)
        mask = _generate_mask(state_dict, seed, device)

        sign = 1.0 if client_id < j else -1.0

        for key in masked:
            masked[key] = masked[key] + sign * mask[key]

    return masked


# ==========================================
# 3. Secure Federated Averaging
# ==========================================

def secure_federated_average(masked_client_weights):
    """
    Average the masked client weights.
    Because pairwise masks cancel in summation, the result
    is identical to plain FedAvg — but the server never
    observes any individual client's raw weights.
    """
    num_clients = len(masked_client_weights)
    avg_weights = copy.deepcopy(masked_client_weights[0])

    for key in avg_weights:
        for i in range(1, num_clients):
            avg_weights[key] += masked_client_weights[i][key]
        avg_weights[key] = avg_weights[key] / num_clients

    return avg_weights


# ==========================================
# 4. Convenience wrapper
# ==========================================

def secure_aggregate(client_state_dicts, round_num, device="cpu"):
    """
    End-to-end secure aggregation.
    1. Each client masks its own weights.
    2. Server averages the masked weights.
    3. Masks cancel — result equals FedAvg.
    """
    num_clients = len(client_state_dicts)
    masked_weights = []

    for client_id, sd in enumerate(client_state_dicts):
        masked = mask_client_weights(
            client_id=client_id,
            state_dict=sd,
            num_clients=num_clients,
            round_num=round_num,
            device=device
        )
        masked_weights.append(masked)

    return secure_federated_average(masked_weights)
