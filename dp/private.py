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
):
    """
    Privatize PyTorch training objects. Opacus replaces the model, optimizer, 
    and dataloader with DP-compatible versions that:
      - use Poisson sampling from dataloader to satisfy DP guarantees
      - clip per-sample gradients to norm_clip
      - inject calibrated Gaussian noise

    Noise is computed automatically to meet specified ε
    over the given number of epochs.

    Args:
        PyTorch training objects: model, optimizer, dataloader
        Relevant hyperparameters: epochs, norm_clip
        epsilon: DP privacy budject
        delta: DP failure probability

    Returns:
        (model, optimizer, dataloader) with DP counterparts
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