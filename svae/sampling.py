"""Sub-module to define sampling procedures and utility functions"""
import torch

# fonctions utilitaires pour vmf
def sample_vmf(mu, kappa, batch_size):
    """échantillonnage depuis vmf en dimension m"""
    m = mu.shape[-1]
    
    # cas 2d simplifié
    if m == 2:
        angles = torch.randn(batch_size, 1) * 2 * torch.pi
        x = torch.cos(angles)
        y = torch.sin(angles)
        samples = torch.cat([x, y], dim=1)
        
        # concentration autour de mu
        for i in range(batch_size):
            if kappa[i] > 0:
                mu_angle = torch.atan2(mu[i,1], mu[i,0])
                concentrated_angles = mu_angle + torch.randn(1, 1) / (kappa[i] + 1e-8)
                xi = torch.cos(concentrated_angles)
                yi = torch.sin(concentrated_angles)
                samples[i] = torch.cat([xi, yi], dim=1)
        return samples
    
    # cas général - algorithme d'ulrich
    b = -2 * kappa + torch.sqrt(4 * kappa**2 + (m-1)**2)
    b = b / (m - 1)
    a = (m - 1 + 2 * kappa + torch.sqrt(4 * kappa**2 + (m-1)**2)) / 4
    d = 4 * a * b / (1 + b) - (m - 1) * torch.log(torch.tensor(m - 1))
    Beta_dist = torch.distributions.Beta((m-1)/2, (m-1)/2)
    
    samples = []
    for i in range(batch_size): # for each sample
        while True:
            epsilon = Beta_dist.sample()
            omega = (1 - (1 + b[i]) * epsilon) / (1 - (1 - b[i]) * epsilon) # >> RAPH: Corrected parenthises of the numerator
            t = 2 * a[i] * b[i] / (1 - (1 - b[i]) * epsilon)
            u = torch.rand(1)
            if (m - 1) * torch.log(t) - t + d[i] >= torch.log(u):
                break
             
        # échantillonnage sur s^{m-2}
        v = torch.randn(m - 1)
        v = v / torch.norm(v)

        # construction du sample
        z = torch.cat([omega, 
                       torch.sqrt(1 - omega**2) * v])
        
        # transformation householder pour aligner avec mu
        e1 = torch.zeros(m)
        e1[0] = 1
        u = e1 - mu[i,:]
        u = u / (torch.norm(u) + 1e-8)
        
        householder = torch.eye(m) - 2 * torch.outer(u, u)
        z = householder @ z
        samples.append(z)
    
    return torch.stack(samples)

def batch_sample_vmf(mu, kappa, batch_size):
    """
    Sample from a von Mises–Fisher distribution in dimension m with batch.
    """
    device = mu.device
    dtype = mu.dtype

    assert mu.dim() == 2, "Expected mu of shape (batch_size, m)"
    B, m = mu.shape
    assert B == batch_size, "batch_size must match mu.shape[0]"

    kappa = kappa.view(B).to(device=device, dtype=dtype)

    # 2D special case
    if m == 2:
        eps = 1e-8
        mu_angle = torch.atan2(mu[:, 1], mu[:, 0])
        noise = torch.randn(B, device=device, dtype=dtype) / (kappa + eps)

        uniform_angles = 2 * torch.pi * torch.rand(B, device=device, dtype=dtype)
        concentrated_angles = mu_angle + noise

        use_concentrated = kappa > 0
        angles = torch.where(use_concentrated, concentrated_angles, uniform_angles)

        x = torch.cos(angles)
        y = torch.sin(angles)
        return torch.stack([x, y], dim=-1)

    # Ulrich algorithm
    m_minus_1 = m - 1.0
    uniform_mask = (kappa <= 1e-3)
    pos_mask = ~uniform_mask

    samples = torch.empty(B, m, device=device, dtype=dtype)

    # uniform case
    if uniform_mask.any():
        v = torch.randn(uniform_mask.sum(), m, device=device, dtype=dtype)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
        samples[uniform_mask] = v

    if pos_mask.any():
        mu_pos = mu[pos_mask]
        kappa_pos = kappa[pos_mask]
        B_pos = mu_pos.shape[0]

        k2 = 4 * kappa_pos ** 2
        sqrt_term = torch.sqrt(k2 + m_minus_1**2)
        b = (-2 * kappa_pos + sqrt_term) / m_minus_1
        a = (m_minus_1 + 2 * kappa_pos + sqrt_term) / 4.0
        d = 4 * a * b / (1 + b) - m_minus_1 * torch.log(torch.tensor(m_minus_1, device=device, dtype=dtype))

        Beta_dist = torch.distributions.Beta(
            torch.tensor(m_minus_1 / 2, device=device, dtype=dtype),
            torch.tensor(m_minus_1 / 2, device=device, dtype=dtype)
        )

        omega = torch.empty(B_pos, device=device, dtype=dtype)
        accepted = torch.zeros(B_pos, dtype=torch.bool, device=device)

        n_iter = 0
        while not torch.all(accepted):
            idx = (~accepted).nonzero(as_tuple=False).squeeze(-1)
            n_rem = idx.numel()
            eps = Beta_dist.sample((n_rem,)).to(device=device, dtype=dtype)

            b_r, a_r, d_r = b[idx], a[idx], d[idx]
            num = 1 - (1 + b_r) * eps
            den = 1 - (1 - b_r) * eps
            omega_cand = num / den

            t = 2 * a_r * b_r / (1 - (1 - b_r) * eps)
            u = torch.rand(n_rem, device=device, dtype=dtype)
            log_accept = m_minus_1 * torch.log(t) - t + d_r - torch.log(u)
            new_accept = log_accept >= 0

            omega[idx[new_accept]] = omega_cand[new_accept]
            accepted[idx] = new_accept
            n_iter += 1

        v = torch.randn(B_pos, int(m_minus_1), device=device, dtype=dtype)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)

        omega_col = omega.unsqueeze(-1)
        sqrt_term = torch.sqrt(1 - omega ** 2).unsqueeze(-1)
        z = torch.cat([omega_col, sqrt_term * v], dim=-1)

        e1 = torch.zeros(1, m, device=device, dtype=dtype)
        e1[0, 0] = 1.0
        e1 = e1.expand(B_pos, m)

        u = e1 - mu_pos
        u = u / (u.norm(dim=-1, keepdim=True) + 1e-8)

        proj = (z * u).sum(dim=-1, keepdim=True)
        z_reflected = z - 2 * u * proj

        samples[pos_mask] = z_reflected
    return samples


def sample_gaussian(mu, std):
    """
    Sampling from a gaussian
    """
    return mu + std * torch.randn_like(std)