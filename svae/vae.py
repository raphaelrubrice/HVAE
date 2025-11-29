"""Sub-module to define architectures of Variational Auto-Encoders"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.gamma import Gamma
import numpy as np
from tqdm.auto import tqdm

# Custome imports
from.sampling import sample_vmf, batch_sample_vmf, sample_gaussian
from.utils import Ive, Iv

def torch_gamma_func(val):
    """
    The gamma function in the PyTorch API is not directly accessible.
    To do so we need to use lgamma which computes ln(|gamma(val)|).
    Thus to access the gamma value we need to compose with the exponential.
    """
    return torch.exp(torch.lgamma(torch.tensor(val)))

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
        # on Sd-1 (Rd) require mu in Rd but kappa in R ! 
        # So you were right :D
        
        # >> RAPH: When kappa grows too high, the probability of acceptance drops
        # in Ulrich. Another paper from Nicols de Cao shows the unstability
        # see https://arxiv.org/pdf/2006.04437 Fig. 2
        # The relation Kappa vs Dim is not log linear but almost
        # By looking at it, for dim < 10000 a heuristic could be to cap the maximum
        # Kappa to 100 * dim
        self.max_kappa = latent_dim * 100

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
        kappa = F.softplus(self.fc_kappa(h)) +1
        # >> RAPH: Why +0.1 ? the authors used +1 => corrected
        
        # >> RAPH: When kappa grows too high, the probability of acceptance drops
        # clipping to prevent getting stuck at sampling
        # not normalizing because that would completely change the resulting distribution we sample from !!
        kappa = torch.clip(kappa, max=self.max_kappa)

        return mu, kappa
    
    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        return self.fc5(h)
    
    def kl_vmf(self, kappa):
        # >> RAPH: One of the main remarks in the calculations 
        # of the KL is that it does not depend on mu => removed unused mu
        """calcul de la divergence kl pour chaque sample du batch"""
        m = self.latent_dim
        
        # utilisation de ive pour stabilité numérique
        ive = Ive.apply(m/2, kappa)
        ive_prev = Ive.apply(m/2 - 1, kappa)
        # >> RAPH: Why go back to the iv instead of directly using ive
        # authors only use ive since exponentials will cancel out due to division
        # I removed the exponentials

        # >> RAPH: ive is not differentiable natively by PyTorch so it 
        # means calling ive on a detached tensor (no grad history) so that does not work
        # to do so we need to specify the backward ourselves (see p.14 equation 16)
        # implemented as Ive in utils.py 

        bessel_ratio = kappa * (ive / (ive_prev + 1e-8))
        # print("ive, ive_prev, kappa", ive, ive_prev, kappa)
        # terme log c_m(kappa)
        # >> RAPH: a true Iv term is necessary (not Ive because there is no ratio here, see Eq.4 p.3)
        ive = Ive.apply(m/2 - 1, kappa)
        # log iv = log ive + kappa
        log_iv = torch.log(ive + 1e-8) + kappa
        
        log_cm = (m/2 - 1) * torch.log(kappa + 1e-8) - (m/2) * torch.log(2 * torch.tensor(torch.pi)) - log_iv
        # >> RAPH: there was a kappa missing, I added it

        # terme constant
        const = (m/2) * torch.log(torch.tensor(torch.pi)) + torch.log(torch.tensor(2)) - torch.log(torch_gamma_func(m/2))
        return bessel_ratio + log_cm + const # see Eq.14 and Eq. 15 p.13
    
    def forward(self, x):
        # B x input_dim
        mu, kappa = self.encode(x)
        # B x latent_dim, B x 1

        # échantillonnage
        batch_size = x.shape[0]
        z = self.sample(mu, kappa)
        # B x latent_dim
        
        # reconstruction
        x_recon = self.decode(z)
        # B x input_dim
        
        return x_recon, mu, kappa

    def reconstruction_loss(self, x_recon, x):
        # >> RAPH: The original x (not latent) are assumed to be Gaussian 
        # so MSE is good here (its implicit 
        # in the paper because they never speak about the prior on 
        # the input only the prior on the latent space)
        # By looking at their code we can see that they 
        # never actually code grep + gcor but only use BCE (they had a binary task
        # since they use the Binarized MNIST = their model predict logits for each 
        # pixel in the binary image)

        # dim squared error
        se = (x_recon - x) ** 2  
        # sum over dimensions, mean over batch
        return se.view(se.size(0), -1).sum(dim=1).mean()
    
    def full_step(self, x, beta_kl):
        x_recon, mu, kappa = self.forward(x)
            
        # reconstruction loss
        recon_loss = self.reconstruction_loss(x_recon, x)
        
        # kl divergence
        # >> RAPH: We average the kl loss over the batch (the usual 
        # sum used in the per term kl of the gaussian comes from an analytical 
        # formula that requires to sum over dimensions, but the overal KL loss
        # should be averaged for final loss computation)
        # Note: Sum and Means are both okay since its fndamentally a sum in both cases
        # however, a mean allows a KL term that is not dependent on the batch size whereas 
        # a sum is sensitive to this. Using a mean allows comparison across experiments of losses
        # whereas the sum does not.
        kl_loss = self.kl_vmf(kappa).mean()

        # LOSS = - ELBO = - (Recon - beta * KL)
        # BUT the MSE is already the (- recon) term when using Gaussian as the input prior
        loss = recon_loss + beta_kl * kl_loss
        return loss, dict(recon=recon_loss.item(),
                          kl=kl_loss.item())
    
    def marginal_ll_sample(self, x, N: int = 500):
        """
        This implements the Importance Weighted Auto Encoder Estimator (Burda et al., 2016)
        for one sample.
        This estimator allows an approximation of the marginal log likelihood for a given sample
        Because vMF can be unstable, we compute in log space then use the logsumexp trick
        """
        self.eval()
        with torch.no_grad():
            # x should be one sample
            mu, kappa = self.encode(x)
            mu = torch.cat([mu for _ in range(N)])
            kappa = torch.cat([kappa for _ in range(N)])
            z = self.sample(mu, kappa)
            x_z = self.decode(z)

        dim = x.size(1)
        if dim > 1:
            normal = torch.distributions.MultivariateNormal(x_z, 
                                                            torch.diag(torch.tensor([1.0]*dim)))
        else:
            normal = torch.distributions.Normal(x_z, 
                                                torch.tensor([1.0]))

        # posterior input: p(x|z)
        log_p_x_z = normal.log_prob(x)
        
        # prior latent: p(z)
        # the uniform probability on the sphere of dimension dim-1
        log_p_z = torch.lgamma(torch.tensor(dim/2.0)) - (dim/2.0)*torch.log(torch.tensor(2*torch.pi))

        # approximate posterior latent: q(z|x)
        ive = Ive.apply(dim/2 - 1, kappa)
        # log iv = log ive + kappa
        log_cm = (dim/2 - 1) * torch.log(kappa + 1e-8) - (dim/2) * torch.log(2 * torch.tensor(torch.pi)) - (torch.log(ive + 1e-8) + kappa)
        # vMF probability
        dot = torch.sum((kappa * mu) * z, dim=1)
        log_q_z_x = log_cm.ravel() + dot

        log_w = log_p_x_z + log_p_z - log_q_z_x
        approx_log_px = torch.logsumexp(log_w, dim=0).item() - np.log(N)
        return approx_log_px
    
    def marginal_ll_batch(self, x, N: int = 500):
        """
        Importance Weighted Autoencoder (Burda et al., 2016) estimator
        for a batch of samples. 
        """
        self.eval()
        with torch.no_grad():
            batch_size, data_dim = x.shape

            # Encode once for the whole batch
            mu, kappa = self.encode(x)   # (batch, latent_dim)

            latent_dim = mu.size(1)

            # Expand to (batch, N, latent_dim)
            mu = mu.unsqueeze(1).expand(batch_size, N, latent_dim)
            kappa = kappa.unsqueeze(1).expand(batch_size, N, 1)

            # Sample z for each datapoint and each draw
            print(f"Sampling {x.size(0)*N} samples..")
            z = self.sample(mu.reshape(-1, latent_dim), # (batch*N, latent_dim)
                            kappa.reshape(-1, 1)) 
            z = z.view(batch_size, N, latent_dim)

            # Decode to get distribution parameters for p(x|z)
            recon_mu = self.decode(z.reshape(-1, latent_dim))   # (batch*N, data_dim)
            recon_mu = recon_mu.view(batch_size, N, data_dim)

            # Likelihood term: log p(x|z)
            # Assume Gaussian with unit variance
            recon_dist = torch.distributions.Normal(loc=recon_mu, scale=1.0)
            log_p_x_z = recon_dist.log_prob(x.unsqueeze(1))     # (batch, 1, data_dim)
            log_p_x_z = log_p_x_z.sum(dim=2)                    # sum over data_dim => (batch, 1)

            # Prior term: log p(z) for uniform on sphere S^{latent_dim-1}
            log_p_z = (torch.lgamma(torch.tensor(latent_dim/2.0)) 
                    - (latent_dim/2.0)*torch.log(torch.tensor(2*torch.pi)))
            log_p_z = log_p_z.expand(batch_size, N)             # broadcast

            # Posterior term: log q(z|x) for vMF
            v = latent_dim/2 - 1
            ive = Ive.apply(v, kappa.reshape(-1, latent_dim))   # scaled Bessel
            log_iv = torch.log(ive + 1e-12) + kappa.reshape(-1, latent_dim)
            log_cm = (v * torch.log(kappa.reshape(-1, latent_dim) + 1e-12)
                    - (latent_dim/2) * torch.log(torch.tensor(2*torch.pi))
                    - log_iv)
            log_cm = log_cm.view(batch_size, N, -1).sum(dim=2)  # reduce over latent_dim

            dot = torch.sum(mu * z, dim=2) * kappa.squeeze(-1)  # (batch, N)
            log_q_z_x = log_cm + dot

            # IWAE weights
            log_w = log_p_x_z + log_p_z - log_q_z_x  # (batch, N)

            # Marginal log likelihood per datapoint
            log_px = torch.logsumexp(log_w, dim=1) - torch.log(torch.tensor(N))  # (batch,)

        return log_px  # size (batch,)


    def total_marginal_ll(self, tensor, N: int = 500, reduced: str = 'mean'):
        """
        Compute the total marginal log likelihood over a given dataset
        """
        datasetsize = tensor.size(0)
        print(f"[SVAE] Computing total marginal LL for {datasetsize} samples with {N} draws each..")
        LL = self.marginal_ll_batch(tensor, N=N)
        if reduced == 'sum':
            return LL.sum()
        return LL.mean()

    def sample(self, mu, kappa):
        return batch_sample_vmf(mu, kappa, mu.size(0))
        # old, non vectorized
        # svae_latent_samples = []
        # for i in range(mu.size(0)):
        #     z = sample_vmf(mu[i:i+1,:], kappa[i:i+1,:], 1)
        #     svae_latent_samples.append(z)
        # return torch.cat(svae_latent_samples, dim=0)
    
    def get_latent_samples(self, data_tensor):
        self.eval()
        with torch.no_grad():
            print("[SVAE] Encoding dataset..")
            mu_all, kappa_all = self.encode(data_tensor)

            print("[SVAE] Sampling from latent space..")
            # For each input's latent distribution, sample 1 element
            svae_latent_samples = self.sample(mu_all, kappa_all)
            svae_latent_samples = svae_latent_samples.cpu().numpy()
            return svae_latent_samples, mu_all, kappa_all
    
    def get_latent_distributions(self, data_tensor):
        self.eval()
        with torch.no_grad():
            print("[SVAE] Encoding dataset..")
            mu_all, kappa_all = self.encode(data_tensor)
            svae_latent_dists = torch.cat([mu_all, kappa_all], dim=1)
            svae_latent_dists = svae_latent_dists.cpu().numpy()
            return svae_latent_dists, mu_all, kappa_all
    
    def get_latent(self, data_tensor, mode="sample"):
        if mode == "sample":
            return self.get_latent_samples(data_tensor)
        elif mode == "dist":
            return self.get_latent_distributions(data_tensor)
        else:
            raise ValueError(f"Unrecognized mode {mode}. Should either be 'sample' or 'dist'.")
    

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
        return sample_gaussian(mu, std)
    
    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        return self.fc5(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def reconstruction_loss(self, x_recon, x):
        # dim squared error
        se = (x_recon - x) ** 2  
        # sum over dimensions, mean over batch
        return se.view(se.size(0), -1).sum(dim=1).mean()
    
    def full_step(self, x, beta_kl):
        x_recon, mu, logvar = self.forward(x)
            
        recon_loss = self.reconstruction_loss(x_recon, x)
        kl_loss = 0.5 * (-1 - logvar + mu.pow(2) + logvar.exp()) 
        # >> RAPH: for each input we compute the formula for log q/p when q and p are gaussians
        kl_loss = kl_loss.mean()
        # >> RAPH: we then take the expected value (estimated over the batch)
        
        # LOSS = - ELBO = - (recon - beta * KL)
        # BUT the MSE is already the (- recon) term when using Gaussian as the input prior
        loss = recon_loss + beta_kl * kl_loss
        return loss, dict(recon=recon_loss.item(),
                          kl=kl_loss.item())

    def marginal_ll_sample(self, x, N: int = 500):
        """
        This implements the Importance Weighted Auto Encoder Estimator (Burda et al., 2016)
        for one sample.
        This estimator allows an approximation of the marginal log likelihood for a given sample
        """
        self.eval()
        with torch.no_grad():
            # x should be one sample
            mu, logvar = self.encode(x)
            std = torch.exp(0.5 * logvar)
            mu = torch.cat([mu for _ in range(N)])
            std = torch.cat([std for _ in range(N)])
            z = self.sample(mu, std)
            x_z = self.decode(z)

        dim = x.size(1)
        if dim > 1:
            normal = torch.distributions.MultivariateNormal(x_z, 
                                                            torch.diag(torch.tensor([1.0]*dim)))
            normal_prior_latent = torch.distributions.MultivariateNormal(torch.tensor([0.0]*dim), 
                                                                        std)
            normal_approx_latent = torch.distributions.MultivariateNormal(mu, 
                                                                        std)
        else:
            normal = torch.distributions.Normal(x_z, 
                                                torch.tensor([1.0]))
            normal_prior_latent = torch.distributions.MultivariateNormal(torch.tensor([0.0]), 
                                                                        std)
            normal_approx_latent = torch.distributions.MultivariateNormal(mu, 
                                                                            std)

        # posterior input: p(x|z)
        log_p_x_z = normal.log_prob(x)
        
        # prior latent: p(z)
        log_p_z = normal_prior_latent.log_prob(z)

        # approximate posterior latent: q(z|x)
        log_q_z_x = normal_approx_latent.log_prob(z)

        log_w = log_p_x_z + log_p_z - log_q_z_x

        approx_px = torch.logsumexp(log_w, dim=0).item() - np.log(N)
        return approx_px
    
    def marginal_ll_batch(self, x, N: int = 500):
        """
        Importance Weighted Autoencoder (Burda et al., 2016) estimator
        for a batch of samples.
        """
        self.eval()
        with torch.no_grad():
            batch_size, data_dim = x.shape

            # Encode once for whole batch
            mu, logvar = self.encode(x)   # (batch, latent_dim)
            std = torch.exp(0.5 * logvar)

            latent_dim = mu.size(1)

            # Expand to (batch, N, latent_dim)
            mu = mu.unsqueeze(1).expand(batch_size, N, latent_dim)
            std = std.unsqueeze(1).expand(batch_size, N, latent_dim)

            # Sample z for each datapoint and each draw
            print(f"Sampling {x.size(0)*N} samples..")
            z = self.sample(mu.reshape(-1, latent_dim),
                        std.reshape(-1, latent_dim))   # (batch*N, latent_dim)
            z = z.view(batch_size, N, latent_dim)

            # Decode to get distribution parameters for p(x|z)
            recon_mu = self.decode(z.reshape(-1, latent_dim))   # (batch*N, data_dim)
            recon_mu = recon_mu.view(batch_size, N, data_dim)

            # Likelihood term: log p(x|z)
            # Assume Gaussian with unit variance
            recon_dist = torch.distributions.Normal(loc=recon_mu, scale=1.0)
            log_p_x_z = recon_dist.log_prob(x.unsqueeze(1))     # (batch, N, data_dim)
            log_p_x_z = log_p_x_z.sum(dim=2)                    # sum over data_dim => (batch, N)

            # Prior term: log p(z) under standard normal
            prior_dist = torch.distributions.Normal(
                loc=torch.zeros(latent_dim, device=x.device),
                scale=torch.ones(latent_dim, device=x.device)
            )
            log_p_z = prior_dist.log_prob(z)                    # (batch, N, latent_dim)
            log_p_z = log_p_z.sum(dim=2)                        # (batch, N)

            # Posterior term: log q(z|x) under encoder Gaussian
            approx_post = torch.distributions.Normal(loc=mu, scale=std)
            log_q_z_x = approx_post.log_prob(z)                 # (batch, N, latent_dim)
            log_q_z_x = log_q_z_x.sum(dim=2)                    # (batch, N)

            # IWAE weights
            log_w = log_p_x_z + log_p_z - log_q_z_x             # (batch, N)

            # Marginal log likelihood per datapoint
            log_px = torch.logsumexp(log_w, dim=1) - torch.log(torch.tensor(N))  # (batch,)

        return log_px  # size (batch,)


    def total_marginal_ll(self, tensor, N: int = 500, reduced: str = 'mean'):
        """
        Compute the total marginal log likelihood over a given dataset
        """
        datasetsize = tensor.size(0)
        print(f"[NVAE] Computing total marginal LL for {datasetsize} samples with {N} draws each..")
        LL = self.marginal_ll_batch(tensor, N=N)
        if reduced == 'sum':
            return LL.sum()
        return LL.mean()
    
    def sample(self, mu, std):
        return sample_gaussian(mu, std)
    
    def get_latent_samples(self, data_tensor):
        with torch.no_grad():
            print("[NVAE] Encoding dataset..")
            mu_all, logvar_all = self.encode(data_tensor)
            std_all = torch.exp(0.5 * logvar_all)

            print("[NVAE] Sampling from latent space..")
            # For each input's latent distribution, sample 1 element
            nvae_latent_samples = self.sample(mu_all, std_all)
            nvae_latent_samples = nvae_latent_samples.cpu().numpy()
            return nvae_latent_samples, mu_all, std_all
    
    def get_latent_distributions(self, data_tensor):
        with torch.no_grad():
            print("[NVAE] Encoding dataset..")
            mu_all, logvar_all = self.encode(data_tensor)
            std_all = torch.exp(0.5 * logvar_all)

            nvae_latent_dists = torch.cat([mu_all, std_all], dim=1)
            nvae_latent_dists = nvae_latent_dists.cpu().numpy()
            return nvae_latent_dists, mu_all, std_all
    
    def get_latent(self, data_tensor, mode="sample"):
        if mode == "sample":
            return self.get_latent_samples(data_tensor)
        elif mode == "dist":
            return self.get_latent_distributions(data_tensor)
        else:
            raise ValueError(f"Unrecognized mode {mode}. Should either be 'sample' or 'dist'.")