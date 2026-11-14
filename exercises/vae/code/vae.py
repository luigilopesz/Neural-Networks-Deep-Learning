"""Shared code for the VAE exercise: data, model, loss, training loop and evaluation.

Everything VAE-specific is written by hand in this file: the reparameterization trick, both terms
of the loss (Bernoulli/BCE reconstruction summed over the 784 pixels and the closed-form Gaussian KL
summed over the latent dimensions) and the sampling. PyTorch supplies only autograd, nn.Linear and
Adam; torchvision only downloads MNIST. No library VAE class and no distribution objects are used
(the statement forbids them).

Units: every loss value in this exercise is in nats per image (summed over pixels / latent
dimensions, averaged over the images).

Run `python vae.py` for a tiny self-check of the hand-written pieces.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torchvision import datasets, transforms

# 4 threads keeps the CPU shareable, and it is the setting the report's numbers were produced with. Training is
# chaotic over the first epochs, so another thread count (or CPU) gives slightly different numbers (see the report).
torch.set_num_threads(4)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"          # MNIST download cache + d=16 checkpoint (gitignored)
FIG_DIR = HERE.parent / "figures"

SEED, EPOCHS, BATCH, LR, BETA = 42, 20, 128, 1e-3, 1.0


# ----------------------------------------------------------------------------------------------
# Part 1A -- data
# ----------------------------------------------------------------------------------------------
def load_data() -> dict:
    """MNIST as flat float32 tensors in [0, 1], split 50 000 / 10 000 / 10 000 (train / val / test)."""
    train_full = datasets.MNIST(str(DATA_DIR), train=True, download=True, transform=transforms.ToTensor())
    test = datasets.MNIST(str(DATA_DIR), train=False, download=True, transform=transforms.ToTensor())
    train, val = random_split(train_full, [50_000, 10_000],
                              generator=torch.Generator().manual_seed(42))   # the statement's split

    # ToTensor() is just uint8 / 255; indexing the raw tensor gives the same numbers without
    # decoding 70 000 PIL images one by one.
    flat = lambda ds: ds.data.float().div(255).flatten(1)
    X, X_test = flat(train_full), flat(test)
    assert torch.equal(train_full[0][0].flatten(), X[0])          # identical to the ToTensor pipeline
    return dict(X_train=X[train.indices], y_train=train_full.targets[train.indices],
                X_val=X[val.indices], y_val=train_full.targets[val.indices],
                X_test=X_test, y_test=test.targets)


# ----------------------------------------------------------------------------------------------
# Part 1B -- model and loss
# ----------------------------------------------------------------------------------------------
class VAE(nn.Module):
    """784 -> 512 -> 256 -> (mu, log sigma^2) in R^d  ->  256 -> 512 -> 784 (sigmoid)."""

    def __init__(self, d: int):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.mu = nn.Linear(256, d)          # head 1: mean of q(z|x)
        # head 2: log sigma^2, not sigma. The layer output is any real number, and sigma = exp(logvar / 2)
        # is positive by construction (no clamp or softplus needed); log sigma^2 is also what the KL contains.
        self.logvar = nn.Linear(256, d)
        self.dec = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(),
                                 nn.Linear(512, 784), nn.Sigmoid())   # pixel probabilities in (0, 1)

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        """z = mu + sigma * eps, eps ~ N(0, I): the noise is an input, so gradients reach mu and logvar."""
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)      # one sample per image
        return self.decode(z), mu, logvar


def recon_term(x_hat, x):
    """-log p(x|z) under a Bernoulli decoder: BCE summed over the 784 pixels, averaged over the batch."""
    # "sum" / batch, never "mean": mean would also divide by the 784 pixels, which shrinks this term
    # 784-fold against the KL, i.e. silently sets beta ~ 784 and pushes q(z|x) onto the prior (collapse).
    return F.binary_cross_entropy(x_hat, x, reduction="sum") / x.size(0)


def kl_term(mu, logvar):
    """KL( N(mu, sigma^2) || N(0, I) ) = 1/2 sum_i (mu_i^2 + sigma_i^2 - 1 - log sigma_i^2), batch average."""
    # Closed form for a diagonal Gaussian against the standard normal prior (no sampling, no library
    # helper). Summed over the d dimensions, averaged over the images: nats per image.
    return 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum() / mu.size(0)


# ----------------------------------------------------------------------------------------------
# Part 1C -- evaluation and training
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, X):
    """(reconstruction, KL) in nats per image, one sampled z per image: a Monte-Carlo estimate of the ELBO."""
    model.eval()
    x_hat, mu, logvar = model(X)
    return recon_term(x_hat, X).item(), kl_term(mu, logvar).item()


@torch.no_grad()
def encode_all(model, X):
    """(mu, logvar) of every image; no sampling."""
    model.eval()
    mu, logvar = model.encode(X)
    return mu, logvar


def train_vae(d, X_train, X_val, epochs=EPOCHS, batch=BATCH, lr=LR, beta=BETA):
    """Adam, minibatches of 128. Returns the model and one dict per epoch with both loss terms,
    on the training set (average over the epoch) and on the validation set (end of the epoch).
    The test set never enters this function."""
    torch.manual_seed(SEED)                       # seed BEFORE building the model: it draws the init
    model = VAE(d)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(X_train))      # fresh shuffle every epoch (global RNG)
        s_rec = s_kl = 0.0
        for i in range(0, len(X_train), batch):
            x = X_train[order[i:i + batch]]
            x_hat, mu, logvar = model(x)
            rec, kl = recon_term(x_hat, x), kl_term(mu, logvar)
            opt.zero_grad()
            (rec + beta * kl).backward()          # L = reconstruction + beta * KL
            opt.step()
            s_rec += rec.item() * len(x)          # weight by batch size (the last batch has 80 images)
            s_kl += kl.item() * len(x)
        val_rec, val_kl = evaluate(model, X_val)
        history.append(dict(epoch=epoch, train_rec=s_rec / len(X_train), train_kl=s_kl / len(X_train),
                            val_rec=val_rec, val_kl=val_kl))
        print(f"d={d:<2d} epoch {epoch:2d}  train {history[-1]['train_rec']:7.2f} + {history[-1]['train_kl']:5.2f}"
              f"   val {val_rec:7.2f} + {val_kl:5.2f} = {val_rec + val_kl:7.2f}", flush=True)
    return model, history


# ----------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------
def tile(imgs, rows, cols):
    """(rows*cols, 784) -> one (rows*28, cols*28) picture, filled row by row."""
    return imgs.view(rows, cols, 28, 28).permute(0, 2, 1, 3).reshape(rows * 28, cols * 28)


def selfcheck():
    z = torch.zeros(3, 4)
    assert kl_term(z, z).item() == 0.0                                          # q = prior -> KL = 0
    assert abs(kl_term(torch.ones(1, 1), torch.zeros(1, 1)).item() - 0.5) < 1e-6  # mu=1, sigma=1 -> 1/2
    lv = torch.full((1, 1), 2 * np.log(2.0))                                    # sigma = 2
    assert abs(kl_term(torch.zeros(1, 1), lv).item() - 0.5 * (4 - 1 - 2 * np.log(2))) < 1e-6
    x, p = torch.rand(5, 784), torch.rand(5, 784).clamp(0.01, 0.99)
    by_hand = -(x * p.log() + (1 - x) * (1 - p).log()).sum() / 5
    assert torch.allclose(recon_term(p, x), by_hand, rtol=1e-5)                 # BCE formula, summed over pixels
    mu, lv = torch.zeros(2, 3, requires_grad=True), torch.zeros(2, 3, requires_grad=True)
    VAE(3).reparameterize(mu, lv).sum().backward()
    assert mu.grad is not None and lv.grad is not None                          # gradients reach both heads
    model = VAE(16)
    assert sum(p.numel() for p in model.parameters()) == 1_079_600
    assert tile(torch.arange(4 * 784.0).view(4, 784), 2, 2).shape == (56, 56)
    print("selfcheck ok")


if __name__ == "__main__":
    selfcheck()
