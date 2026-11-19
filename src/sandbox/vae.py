"""Generative Core: Variational Autoencoder per dati tabellari (PyTorch).

Encoder e decoder sono composti da strati densi (fully connected).
L'encoder produce media e log-varianza della distribuzione latente,
la reparameterization trick permette di campionare z mantenendo il
gradiente, il decoder ricostruisce il vettore originale.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _mlp(in_dim: int, hidden_dims: tuple[int, ...]) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.2)]
        prev = h
    return nn.Sequential(*layers), prev


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], latent_dim: int):
        super().__init__()
        self.body, last = _mlp(input_dim, hidden_dims)
        self.fc_mu = nn.Linear(last, latent_dim)
        self.fc_logvar = nn.Linear(last, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: tuple[int, ...], output_dim: int):
        super().__init__()
        self.body, last = _mlp(latent_dim, tuple(reversed(hidden_dims)))
        self.out = nn.Linear(last, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Sigmoid: sia le numeriche (MinMax) sia le one-hot stanno in [0, 1]
        return torch.sigmoid(self.out(self.body(z)))


class TabularVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        hidden_dims: tuple[int, ...] = (64, 32),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = tuple(hidden_dims)
        self.encoder = Encoder(input_dim, self.hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, self.hidden_dims, input_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + sigma * eps, con eps ~ N(0, I)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    @torch.no_grad()
    def sample(self, n: int, device: str | torch.device = "cpu") -> torch.Tensor:
        """Generazione: l'encoder viene "spento", si campiona z ~ N(0, I)
        e lo si passa al solo decoder."""
        self.eval()
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z)

    def config(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden_dims": list(self.hidden_dims),
        }


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Loss = Reconstruction Loss (MSE) + beta * KL Divergence.

    Entrambi i termini sono sommati sulle feature e mediati sul batch,
    cosi' hanno una scala confrontabile.
    """
    recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl, recon_loss, kl
