"""
==========
private.py
==========
Privatize PyTorch training objects for DPSGD using Opacus.
"""

from opacus import PrivacyEngine

def privatize(
    model,
    optimizer,
    dataloader,
    epochs: int,
    norm_clip: float,
    epsilon: float,
    delta: float,
) -> tuple:
    """
    Privatize PyTorch training objects. Opacus replaces the model, optimizer,
    and dataloader with DP-compatible versions that:
    1. use Poisson sampling from dataloader to satisfy DP guarantees
    2. clip per-sample gradients to norm_clip
    3. inject calibrated Gaussian noise scaled to meet target epsilon

    Args:
        model: PyTorch model to privatize
        optimizer: optimizer to privatize
        dataloader: dataloader to privatize
        epochs: number of training epochs
        norm_clip: per-sample gradient clipping threshold
        epsilon: DP privacy budget
        delta: DP failure probability

    Returns:
        (model, optimizer, dataloader) DP-compatible counterparts
    """
    privacy_engine = PrivacyEngine()
    model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        epochs=epochs,
        target_epsilon=epsilon,
        target_delta=delta,
        max_grad_norm=norm_clip,
    )

    return model, optimizer, dataloader