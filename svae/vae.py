"""Sub-module to define architectures of Variational Auto-Encoders"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import ive
from torch.distributions.gamma import Gamma

# Custome imports
from.sampling import sample_vmf

def torch_gamma_func(val):
    """
    The gamma function in the PyTorch API is not directly accessible.
    To do so we need to use lgamma which computes ln(|gamma(val)|).
    Thus to access the gamma value we need to compose with the exponential.
    """
    return torch.exp(torch.lgamma(val))

class SVAE(nn.Module):
    """implémentation du s-vae avec distribution vmf"""
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        
        # encodeur
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_kappa = nn.Linear(hidden_dim // 2, 1)
        # >> RAPH: I stand corrected, a Gaussian in Rd needs mu and sigma 
        # to be in Rd but that's not the case for the vMF: a vMF 
        # on Sd-1 (Rd) require mu in Rd but kappa in R ! So you were right :D
        
        # décodeur
        self.fc3 = nn.Linear(latent_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        
        # mu normalisé sur la sphère
        mu = self.fc_mu(h)
        mu = mu / (torch.norm(mu, dim=-1, keepdim=True) + 1e-8)
        
        # kappa positif
        kappa = F.softplus(self.fc_kappa(h)) + 0.1 
        # >> RAPH: Why +0.1 ? the authors used +1
        
        return mu, kappa
    
    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        return self.fc5(h)
    
    def kl_vmf(self, mu, kappa):
        # >> RAPH: One of the main remarks in the calculations 
        # of the KL is that it does not depend on mu => remove unused mu
        # >> RAPH: The other main remark is that we cannot autograd this term
        # so we need to specify the gradient by hand for kappa see kl_grad
        """calcul de la divergence kl"""
        m = self.latent_dim
        
        # utilisation de ive pour stabilité numérique
        iv = ive(m/2, kappa) * torch.exp(torch.abs(kappa)) 
        # >> RAPH: Why go back to the iv instead of directly using 
        # iv(m/2, kappa) ?
        iv_prev = ive(m/2 - 1, kappa) * torch.exp(torch.abs(kappa))
        
        kl = kappa * (iv / (iv_prev + 1e-8))
        
        # terme log c_m(kappa)
        log_cm = (m/2 - 1) * torch.log(kappa + 1e-8) - (m/2) * torch.log(2 * torch.pi) - torch.log(iv_prev + 1e-8)
        
        # terme constant
        const = -torch.log(torch_gamma_func(m/2) / (2 * torch.pi**(m/2)))

        return (kl + log_cm + const).mean()
    
    def kl_grad(self, kappa):
        """
        Computes the average gradient wrt kappa over the batch
        """
        m = self.latent_dim
        kappa = kappa.detach().numpy()

        iv_1 = ive(m/2 + 1, kappa)
        iv_2 = ive(m/2 - 1, kappa)
        iv_3 = ive(m/2, kappa)
        iv_4 = ive(m/2 - 2, kappa)
        grad = 0.5 * kappa * (iv_1/iv_2 - (iv_3 * (iv_4 + iv_3))/iv_2**2 + 1)
        return grad.mean()
    
    def forward(self, x):
        # B x input_dim
        mu, kappa = self.encode(x)
        # B x latent_dim, B x 1

        # échantillonnage
        batch_size = x.shape[0]
        z = sample_vmf(mu, kappa, batch_size)
        # B x latent_dim
        
        # reconstruction
        x_recon = self.decode(z)
        # B x input_dim
        
        return x_recon, mu, kappa
    

class GaussianVAE(nn.Module):
    """vae standard avec prior gaussien"""
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        
        # encodeur
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # décodeur
        self.fc3 = nn.Linear(latent_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        return self.fc5(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar