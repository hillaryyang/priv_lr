"""
==========
private.py
==========
Privatize training objects for differentially private stochastic gradient descent using Opacus.
Opacus adds DP capabilities to PyTorch training objects (model, optimizer, data loader)
and tracks privacy expenditures at each training step.

We use Opacus' make_private_with_epsilon() module that calculates privacy parameters 
based on a given privacy budget. This function allows us to pass in our desired privacy level.
"""
from opacus import PrivacyEngine

def privatize(model, optimizer, dataloader, epochs, epsilon, delta, norm_clip):
    # PrivacyEngine() wraps PyTorch training objects with private versions
    privacy_engine = PrivacyEngine() # initialize Opacus PrivacyEngine() 
    model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
        module=model, # model can now compute per sample gradients
        optimizer=optimizer, # optimizer can now clip gradients and add noise
        data_loader=dataloader, # dataloader can now perform Poisson sampling
        epochs=epochs,
        target_epsilon=epsilon,
        target_delta=delta,
        max_grad_norm=norm_clip
    )

    return model, optimizer, dataloader