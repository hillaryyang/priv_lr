"""
Use Opacus to privatize PyTorch training objects
"""

from opacus import PrivacyEngine

def privatize(
    model, optimizer, dataloader,
    epochs: int, norm_clip: float,
    epsilon: float, delta: float,
) -> tuple:
    """
    Opacus updates training objects with DP-compatible versions that:
    1. Poisson sample from dataloader
    2. clip per-sample gradients to norm_clip
    3. calculate/add Gaussian noise scaled to meet target epsilon

    Args:
        model, optimizer, dataloader: training objects to privatize
        epochs, norm_clip: relevant hyperparameters
        epsilon, delta: DP privacy budget and failure probability (privacy params)
        
    Returns:
        model, optimizer, dataloader: Privatized versions of training objects
    """

    privacy_engine = PrivacyEngine()
    model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
        module=model, optimizer=optimizer, data_loader=dataloader,
        epochs=epochs, max_grad_norm=norm_clip,
        target_epsilon=epsilon,
        target_delta=delta,
    )

    return model, optimizer, dataloader