# ==========================================
# dp_utils.py
# Differential Privacy Utilities (Opacus)
# ==========================================

import torch
from opacus import PrivacyEngine
import config


# ==========================================
# Make Model Private
# ==========================================

def make_model_private(model, optimizer, dataloader):
    """
    Wrap model, optimizer, and dataloader with Opacus PrivacyEngine
    """

    # 🔥 IMPORTANT: Model MUST be in training mode for Opacus
    model.train()

    # Optional: secure RNG (set in config if needed)
    privacy_engine = PrivacyEngine(
        secure_mode=getattr(config, "SECURE_MODE", False)
    )

    model, optimizer, private_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=config.NOISE_MULTIPLIER,
        max_grad_norm=config.MAX_GRAD_NORM,
    )

    return model, optimizer, private_loader, privacy_engine


# ==========================================
# Get Epsilon
# ==========================================

def get_epsilon(privacy_engine):
    """
    Returns current epsilon value
    """
    epsilon = privacy_engine.get_epsilon(delta=config.DELTA)
    return epsilon