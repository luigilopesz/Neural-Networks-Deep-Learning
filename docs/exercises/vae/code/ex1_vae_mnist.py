"""Exercise 1 -- a VAE on MNIST (d = 16).

Trains the model, evaluates it once on the test set, and writes
    ../figures/fig1.png  fig2.png  fig3.png      the three figures of the exercise
    results_ex1.json                              every number quoted in the report
    data/vae_d16.pt                               checkpoint reused by Exercise 2 (gitignored)

Run from anywhere:  python ex1_vae_mnist.py      (about a minute on a laptop CPU)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from vae import DATA_DIR, FIG_DIR, HERE, SEED, VAE, encode_all, evaluate, load_data, recon_term, tile, train_vae

D = 16
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"axes.grid": True, "grid.color": "#000000", "grid.alpha": 0.08,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10.5})


def figure1(hist):
    """Reconstruction x epoch and KL x epoch, each with the training and the validation curve."""
    epochs = [h["epoch"] for h in hist]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panels = [("rec", "Reconstruction term (BCE summed over 784 pixels)"),
              ("kl", f"KL term (closed form, summed over d = {D} dimensions)")]
    for ax, (key, title) in zip(axes, panels):
        ax.plot(epochs, [h[f"train_{key}"] for h in hist], "o-", ms=3.5, color=BLUE, label="train (average over the epoch)")
        ax.plot(epochs, [h[f"val_{key}"] for h in hist], "s--", ms=3.5, color=ORANGE, label="validation (end of the epoch)")
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("epoch")
        ax.set_ylabel("nats per image")
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.legend(frameon=False)
    # label the first and last validation KL; the first label sits right of the steep training curve, not on it
    for e, offset in ((0, (24, -6)), (-1, (4, -14))):
        v = hist[e]["val_kl"]
        axes[1].annotate(f"{v:.2f}", (hist[e]["epoch"], v), textcoords="offset points", xytext=offset, color=ORANGE)
    fig.suptitle("Figure 1 -- Training curves of the d = 16 VAE", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1.png", dpi=150)
    plt.close(fig)


def figure2(model, X_te, first):
    """First test image of each digit (top) and its reconstruction (bottom), decoded from mu."""
    with torch.no_grad():
        x = X_te[first]
        mu, _ = model.encode(x)
        # decode mu, do not sample z: mu is the centre of q(z|x), a fixed code per image. A sampled z adds
        # noise that says nothing about the image (the KL term is what pays for it) and changes at every run.
        recon = model.decode(mu)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.imshow(tile(torch.cat([x, recon]), 2, 10).numpy(), cmap="gray", vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(14 + 28 * np.arange(10)); ax.set_xticklabels(range(10))
    ax.set_yticks([14, 42]); ax.set_yticklabels(["input", "reconstruction"])
    ax.set_xlabel("digit class"); ax.set_ylabel("row")
    ax.set_title("Figure 2 -- First test image of each digit and its reconstruction (decoding mu)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2.png", dpi=150, bbox_inches="tight", pad_inches=0.15)   # crop the blank band the 5:1 image leaves
    plt.close(fig)


def figure3(model):
    """8 x 8 images decoded from z ~ N(0, I), drawn right after torch.manual_seed(0)."""
    torch.manual_seed(0)
    z = torch.randn(64, D)                              # nothing else touches the RNG between seed and draw
    with torch.no_grad():
        samples = model.decode(z)
    fig, ax = plt.subplots(figsize=(7.4, 7.6))
    ax.imshow(tile(samples, 8, 8).numpy(), cmap="gray", vmin=0, vmax=1)
    ax.grid(False)
    ticks = 14 + 28 * np.arange(8)
    ax.set_xticks(ticks); ax.set_xticklabels(range(8)); ax.set_yticks(ticks); ax.set_yticklabels(range(8))
    ax.set_xlabel("sample column"); ax.set_ylabel("sample row")
    ax.set_title(f"Figure 3 -- 64 images decoded from z ~ N(0, I), d = {D}, seed 0", fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3.png", dpi=150)
    plt.close(fig)


@torch.no_grad()
def validation_diagnostics(model, data):
    """The numbers behind the 1E answers. Validation set only: the test set is kept for the headline numbers."""
    X_tr, X_va = data["X_train"], data["X_val"]
    mu_va, lv_va = encode_all(model, X_va)
    mu_tr, _ = encode_all(model, X_tr)
    out = {}

    # (a) which latent dimensions carry information
    kl_dim = (0.5 * (mu_va ** 2 + lv_va.exp() - 1 - lv_va)).mean(0)         # KL per dimension, nats
    active = (mu_va.var(0) > 0.01).nonzero().flatten()                       # mu varies across images
    out["kl_per_dim"] = kl_dim.tolist()
    out["active_units"] = len(active)
    out["mean_sigma_active"] = (0.5 * lv_va).exp()[:, active].mean().item()

    # (b) decode(mu) against decode(sampled z), same validation images
    torch.manual_seed(1)
    out["val_rec_sampled_z"] = evaluate(model, X_va)[0]
    out["val_rec_mu"] = recon_term(model.decode(mu_va), X_va).item()

    # (c) blur: pixel entropy of what the decoder outputs, sum_j H(p_j) = recon_term(p, p), nats per image
    torch.manual_seed(2)
    recon, prior = model.decode(mu_va), model.decode(torch.randn(10_000, D))
    gray = lambda p: ((p > 0.1) & (p < 0.9)).float().mean().item()           # share of "gray" pixels
    out["pixel_entropy"] = dict(data=recon_term(X_va, X_va).item(), recon_mu=recon_term(recon, recon).item(),
                                prior_samples=recon_term(prior, prior).item())
    out["gray_fraction"] = dict(data=gray(X_va), recon_mu=gray(recon), prior_samples=gray(prior))

    # (d) holes: distance of a code to the nearest encoded TRAINING image (active dimensions only).
    # If the aggregate posterior matched the prior, 5% of prior samples would exceed the 95th percentile
    # of the distances of codes drawn from the encoder itself.
    ref = mu_tr[:, active]
    torch.manual_seed(3)
    from_encoder = (mu_va + (0.5 * lv_va).exp() * torch.randn_like(mu_va))[:, active]
    from_prior = torch.randn(10_000, D)[:, active]
    nearest = lambda q: torch.cat([torch.cdist(q[i:i + 2000], ref).min(1).values for i in range(0, len(q), 2000)])
    d_enc, d_pri = nearest(from_encoder), nearest(from_prior)
    p95 = torch.quantile(d_enc, 0.95)
    out["holes"] = dict(median_encoder=d_enc.median().item(), median_prior=d_pri.median().item(),
                        p95_encoder=p95.item(), frac_prior_beyond_p95=(d_pri > p95).float().mean().item())

    # (e) two reference points for "why does the KL rise": the untrained network (same initialisation as the
    # trained one) and the best decoder that ignores z, a constant image (KL = 0 by construction). The
    # constant that minimises the summed BCE is the mean training image; the clamp only guards log(0).
    torch.manual_seed(SEED)
    untrained = VAE(D)
    torch.manual_seed(4)
    out["untrained_val"] = dict(zip(("rec", "kl"), evaluate(untrained, X_va)))
    mean_image = X_tr.mean(0).clamp(1e-4, 1 - 1e-4)
    out["mean_image_val_rec"] = recon_term(mean_image.expand_as(X_va), X_va).item()
    return out


def main():
    FIG_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    # A -- data. Train and validation feed the training loop; the test set is not touched until below.
    data = load_data()
    print(f"split: train {len(data['X_train'])}, validation {len(data['X_val'])}, test {len(data['X_test'])}")

    # B + C -- build (torch.manual_seed(42) inside train_vae) and train for 20 epochs
    model, hist = train_vae(D, data["X_train"], data["X_val"])
    torch.save(model.state_dict(), DATA_DIR / "vae_d16.pt")
    n_params = sum(p.numel() for p in model.parameters())

    # --- the test set is used for the first time here: final numbers and figures only ---
    X_te, y_te = data["X_test"], data["y_test"]
    test_rec, test_kl = evaluate(model, X_te)
    print(f"TEST d={D}: reconstruction {test_rec:.2f}  KL {test_kl:.2f}  negative ELBO {test_rec + test_kl:.2f}  ({n_params} parameters)")
    print(f"validation KL: epoch 1 = {hist[0]['val_kl']:.2f}, epoch 20 = {hist[-1]['val_kl']:.2f}")

    first = [int((y_te == k).nonzero()[0]) for k in range(10)]               # first test image of each digit
    figure1(hist)
    figure2(model, X_te, first)
    figure3(model)

    diag = validation_diagnostics(model, data)
    print(json.dumps(diag, indent=1))

    # the numbers are bit-exact only for this torch build and thread count (see the report): record them
    results = dict(env=dict(torch=torch.__version__, threads=torch.get_num_threads()),
                   split=dict(train=50_000, val=10_000, test=10_000), n_params=n_params, history=hist,
                   test=dict(rec=test_rec, kl=test_kl, neg_elbo=test_rec + test_kl),
                   val_kl_first=hist[0]["val_kl"], val_kl_last=hist[-1]["val_kl"],
                   first_test_index_per_digit=first, diagnostics=diag)
    (HERE / "results_ex1.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
