"""
==========
private.py
==========
Privatize PyTorch training objects for DPSGD-LR using Opacus
"""

from opacus import PrivacyEngine

def privatize(
    model, optimizer, dataloader,
    epochs: int, norm_clip: float,
    epsilon: float, delta: float,
) -> tuple:
    """
    Opacus replaces the model, optimizer, and dataloader with DP-compatible versions that:
    1. use Poisson sampling from dataloader to satisfy DP guarantees
    2. clip per-sample gradients to norm_clip
    3. add calibrated Gaussian noise scaled to meet target epsilon

    Args:
        model, optimizer, dataloader: training objects to privatize
        epochs, norm_clip: relevant hyperparameters
        epsilon, delta: DP privacy budget and failure probability (privacy params)
        
    Returns:
        (model, optimizer, dataloader): DP-compatible versions
    """

    privacy_engine = PrivacyEngine()
    model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
        module=model, optimizer=optimizer, data_loader=dataloader,
        epochs=epochs, max_grad_norm=norm_clip,
        target_epsilon=epsilon,
        target_delta=delta,
    )

    return model, optimizer, dataloader