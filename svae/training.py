"""Sub-module to define training protocols"""
import torch.nn.functional as F
import torch

# Custom imports
from svae.vae import SVAE, GaussianVAE

# Global variable
SUPPORTED_CLASSES = [SVAE, GaussianVAE]

def training_svae(dataloader: torch.utils.data.DataLoader,
                  model_svae: SVAE, 
                  optimizer: torch.optim.Optimizer, 
                  epochs: int = 50,
                  beta_kl : float = 0.1,
                  scheduler: torch.optim.lr_scheduler.LRScheduler = None):
    """
    Training protocol for SVAE models.
    """
    assert isinstance(model_svae, SVAE), f"This training loop is tailored for SVAE modules"

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            
            x_recon, mu, kappa = model_svae(x)
            
            # reconstruction loss
            recon_loss = F.mse_loss(x_recon, x)
            
            # # kl divergence
            # kl_loss = model_svae.kl_vmf(mu, kappa)

            # kl grad
            kl_grad = model_svae.kl_grad(kappa)
            kl_grad = beta_kl * kl_grad

            loss = recon_loss #+ beta_kl * kl_loss
            loss.backward()
            learning_rate = optimizer.lr
            manual_grad_update(model_svae, kl_grad, learning_rate)
            
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            epoch_loss += loss.item()
        
        losses.append(epoch_loss / len(dataloader))
        if epoch % 10 == 0:
            print(f"epoch {epoch}: {losses[-1]:.4f}")
    return model_svae, losses


def training_nvae(dataloader: torch.utils.data.DataLoader,
                  model_nvae: GaussianVAE, 
                  optimizer: torch.optim.Optimizer, 
                  epochs: int = 50,
                  beta_kl : float = 0.001,
                  scheduler: torch.optim.lr_scheduler.LRScheduler = None):
    """
    Training protocol for NVAE models.
    """
    assert isinstance(model_nvae, GaussianVAE), f"This training loop is tailored for SVAE modules"

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            
            x_recon, mu, logvar = model_nvae(x)
            
            recon_loss = F.mse_loss(x_recon, x)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = recon_loss + beta_kl * kl_loss
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        
        losses.append(epoch_loss / len(dataloader))
        if epoch % 10 == 0:
            print(f"epoch {epoch}: {losses[-1]:.4f}")
    return model_nvae, losses

def training_loop(dataloader: torch.utils.data.DataLoader,
                  model: torch.nn.Module, 
                  optimizer: torch.optim.Optimizer, 
                  epochs: int = 50,
                  beta_kl : float = 0.001,
                  scheduler: torch.optim.lr_scheduler.LRScheduler = None):
    if isinstance(model, SVAE):
        return training_svae(dataloader,
                            model, 
                            optimizer, 
                            epochs,
                            beta_kl,
                            scheduler)
    elif isinstance(model, GaussianVAE):
        return training_nvae(dataloader,
                            model, 
                            optimizer, 
                            epochs,
                            beta_kl,
                            scheduler)
    else:
        raise ValueError(f"Unsupported model class: Only support {SUPPORTED_CLASSES} but got {model.__class__}")