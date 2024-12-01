from opacus import PrivacyEngine

def privatize(model, optimizer, dataloader, epochs, epsilon, delta, norm_clip):
    privacy_engine = PrivacyEngine()
    model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        epochs=epochs,
        target_epsilon=epsilon,
        target_delta=delta,
        max_grad_norm=norm_clip
    )

    return model, optimizer, dataloader