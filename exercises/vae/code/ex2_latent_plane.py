"""Exercise 2 -- a latent space you can see (d = 2), and interpolation with the d = 16 model.

Trains the d = 2 VAE (same code, data, seed, beta and budget as Exercise 1) and reuses the d = 16
model of Exercise 1 (data/vae_d16.pt; Exercise 1 is run first if the checkpoint is missing). Writes
    ../figures/fig4.png  fig5.png  fig6.png
    results_ex2.json                              every number quoted in the report

Run from anywhere:  python ex2_latent_plane.py   (about a minute on a laptop CPU)
"""
import itertools
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from vae import DATA_DIR, FIG_DIR, HERE, VAE, encode_all, evaluate, load_data, tile, train_vae

plt.rcParams.update({"axes.grid": True, "grid.color": "#000000", "grid.alpha": 0.08,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10.5})

# 15 evenly spaced quantiles of the prior, mapped back to z: equal prior mass per step along each axis
Q = np.linspace(0.05, 0.95, 15)
ZG = norm.ppf(Q) + 0.0                                    # -1.645 ... +1.645, denser near 0
fmt = lambda v: f"{v:.2f}".replace("-0.00", "0.00")

# Figure 6 pair, PINNED: test image 8 (a 5) and test image 0 (a 7), two of the ten images of Figure 2. In the committed
# run it is also the pair with the largest gap computed in main(), but that argmax is float-sensitive (another thread
# count picks another pair), so the script prints the diagnostic and does not let it move Figure 6.
IA, IB = 8, 0


def figure4(mu, y):
    """Scatter of mu for all 10 000 test images, colored by digit."""
    assert len(mu) == len(y) == 10_000                    # every test image, no subsampling
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    for k in range(10):
        m = (y == k)
        ax.scatter(mu[m, 0], mu[m, 1], s=5, alpha=0.6, linewidths=0, color=plt.get_cmap("tab10")(k), label=str(k))
    lo, hi = ZG[0], ZG[-1]                                # the square decoded in Figure 5
    ax.add_patch(plt.Rectangle((lo, lo), hi - lo, hi - lo, fill=False, ls="--", lw=1, ec="black"))
    ax.text(hi, lo - 0.06, "Figure 5 grid", ha="right", va="top", fontsize=8.5,    # a label, not an 11th legend entry
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\mu_1$"); ax.set_ylabel(r"$\mu_2$")
    ax.set_title("Figure 4 -- Encoded test set, d = 2 (10 000 images, colored by digit)", fontweight="bold")
    ax.legend(title="digit", markerscale=3, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4.png", dpi=150)
    plt.close(fig)


def figure5(m2):
    """15 x 15 decoded plane. Columns: z1 increasing to the right; rows: z2 increasing UPWARD (as in Figure 4)."""
    Z = torch.tensor([[z1, z2] for z2 in ZG[::-1] for z1 in ZG], dtype=torch.float32)     # row-major, top row first
    with torch.no_grad():
        imgs = m2.decode(Z)
    fig, ax = plt.subplots(figsize=(8.4, 8.6))
    ax.imshow(tile(imgs, 15, 15).numpy(), cmap="gray", vmin=0, vmax=1)
    ax.grid(False)
    ticks = 14 + 28 * np.arange(15)
    ax.set_xticks(ticks); ax.set_xticklabels([fmt(v) for v in ZG], rotation=90, fontsize=7.5)
    ax.set_yticks(ticks); ax.set_yticklabels([fmt(v) for v in ZG[::-1]], fontsize=7.5)
    ax.set_xlabel(r"$z_1 = \Phi^{-1}(q)$"); ax.set_ylabel(r"$z_2 = \Phi^{-1}(q)$")
    # cell index 1-15 (column from the left, row from the top) on the top and right axes: the addresses the report uses
    top, right = ax.secondary_xaxis("top"), ax.secondary_yaxis("right")
    top.set_xticks(ticks); top.set_xticklabels(range(1, 16), fontsize=7.5); top.set_xlabel("column", fontsize=8.5)
    right.set_yticks(ticks); right.set_yticklabels(range(1, 16), fontsize=7.5); right.set_ylabel("row", fontsize=8.5)
    ax.set_title("Figure 5 -- Decoded 15 x 15 latent plane, d = 2 (q evenly spaced in [0.05, 0.95])", fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5.png", dpi=150)
    plt.close(fig)


def rows(m16, X_te, ia, ib, t):
    """Pixel-space row and latent-space row (d = 16 model, decoding the interpolated mu) for the pair (ia, ib)."""
    pix = (1 - t) * X_te[ia] + t * X_te[ib]
    with torch.no_grad():
        mu, _ = m16.encode(X_te[[ia, ib]])
        lat = m16.decode((1 - t) * mu[0] + t * mu[1])
    return pix, lat


def figure6(pix, lat, ia, ib, ya, yb, t):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.imshow(tile(torch.cat([pix, lat]), 2, 10).numpy(), cmap="gray", vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(14 + 28 * np.arange(10)); ax.set_xticklabels([f"{v:.2f}" for v in t.flatten()])
    ax.set_yticks([14, 42]); ax.set_yticklabels(["pixel space", "latent space (d = 16)"])
    ax.set_xlabel("t"); ax.set_ylabel("interpolation")
    ax.set_title(f"Figure 6 -- Interpolation from test image {ia} (digit {ya}) to test image {ib} (digit {yb})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6.png", dpi=150, bbox_inches="tight", pad_inches=0.15)   # crop the blank band the 5:1 image leaves
    plt.close(fig)


def main():
    FIG_DIR.mkdir(exist_ok=True)
    data = load_data()
    X_tr, y_tr, X_va, X_te, y_te = data["X_train"], data["y_train"], data["X_val"], data["X_test"], data["y_test"]

    # the d = 16 model of Exercise 1 (checkpoint written by ex1_vae_mnist.py) and its test numbers
    ckpt = DATA_DIR / "vae_d16.pt"
    if not ckpt.exists():
        import ex1_vae_mnist
        ex1_vae_mnist.main()
    m16 = VAE(16)
    m16.load_state_dict(torch.load(ckpt))
    m16.eval()
    ex1 = json.loads((HERE / "results_ex1.json").read_text())["test"]

    # A -- the 2-D model: nothing changes except d
    m2, hist2 = train_vae(2, X_tr, X_va)
    rec2, kl2 = evaluate(m2, X_te)                        # test set: final numbers only from here on
    print(f"TEST d=2 : reconstruction {rec2:.2f}  KL {kl2:.2f}  negative ELBO {rec2 + kl2:.2f}")
    print(f"TEST d=16: reconstruction {ex1['rec']:.2f}  KL {ex1['kl']:.2f}  negative ELBO {ex1['neg_elbo']:.2f}   (from Exercise 1)")
    per_digit = {k: dict(d16=evaluate(m16, X_te[y_te == k])[0], d2=evaluate(m2, X_te[y_te == k])[0]) for k in range(10)}

    # B -- encoded test set
    mu_te, lv_te = encode_all(m2, X_te)
    mu_tr, _ = encode_all(m2, X_tr)
    figure4(mu_te.numpy(), y_te.numpy())
    cent = np.stack([mu_te[y_te == k].mean(0).numpy() for k in range(10)])             # digit centroids in the plane

    # overlap, measured: 15-nearest-neighbour vote in the 2-D code (fit on the training codes)
    pred2 = KNeighborsClassifier(n_neighbors=15).fit(mu_tr.numpy(), y_tr.numpy()).predict(mu_te.numpy())
    cm = confusion_matrix(y_te.numpy(), pred2)
    mu16_te, _ = encode_all(m16, X_te)
    mu16_tr, _ = encode_all(m16, X_tr)
    pred16 = KNeighborsClassifier(n_neighbors=15).fit(mu16_tr.numpy(), y_tr.numpy()).predict(mu16_te.numpy())
    cm16 = confusion_matrix(y_te.numpy(), pred16)
    cell = lambda p: (14 - int(np.abs(ZG - p[1]).argmin()) + 1, int(np.abs(ZG - p[0]).argmin()) + 1)   # 1-based (row from top, col from left)
    top = sorted(((int(cm[a, b] + cm[b, a]), a, b) for a, b in itertools.combinations(range(10), 2)), reverse=True)[:5]
    overlap = [dict(pair=[a, b], confused=tot, rate_a_as_b=float(cm[a, b] / cm[a].sum()), rate_b_as_a=float(cm[b, a] / cm[b].sum()),
                    centroid_distance=float(np.linalg.norm(cent[a] - cent[b])), midpoint=((cent[a] + cent[b]) / 2).tolist(),
                    figure5_cell_row_col=cell((cent[a] + cent[b]) / 2)) for tot, a, b in top]

    # C -- decoded plane
    figure5(m2)
    dq = float(Q[1] - Q[0])

    # D -- interpolation between the pinned pair (IA, IB), two different digits. The gap diagnostic is computed for the
    # 45 pairs of Figure 2's images (first test image of each digit): how much farther from real digits the in-between
    # pixel images are than the latent ones. It is only reported, never used to pick the pair.
    first = json.loads((HERE / "results_ex1.json").read_text())["first_test_index_per_digit"]
    t = torch.linspace(0, 1, 10)[:, None]
    nearest = lambda imgs: torch.cat([torch.cdist(imgs[i:i + 1000], X_va).min(1).values
                                      for i in range(0, len(imgs), 1000)]).mean().item()   # mean L2 distance to the closest validation image
    gaps = {}                                             # (test index a, test index b) -> (gap, pixel row, latent row)
    for a, b in itertools.combinations(range(10), 2):
        pix, lat = rows(m16, X_te, first[a], first[b], t)
        dp, dl = nearest(pix[1:-1]), nearest(lat[1:-1])
        gaps[first[a], first[b]] = (dp - dl, dp, dl)
    best = max(gaps, key=lambda k: gaps[k][0])            # the pair this run's model would pick
    gap, d_pix, d_lat = gaps[IA, IB]                      # (KeyError unless IA, IB are two of Figure 2's images, a < b)
    ia, ib, a, b = IA, IB, int(y_te[IA]), int(y_te[IB])
    assert a != b                                         # two different digits (the statement)
    print(f"Figure 6 pair (pinned): test {ia} (digit {a}) -> test {ib} (digit {b}), gap {gap:.2f}; "
          f"largest gap of this run: test {best[0]} -> test {best[1]}, gap {gaps[best][0]:.2f}"
          + ("" if best == (IA, IB) else "   <-- differs from the pinned pair (float-sensitive argmax)"))
    pix, lat = rows(m16, X_te, ia, ib, t)
    figure6(pix, lat, ia, ib, a, b, t)

    # Scale for those distances, same reference set (validation images): a real test digit, and the t = 0.5 image
    # of 500 random pairs of test images of different digits (pixel blend vs decoded latent midpoint, d = 16).
    pa, pb = torch.randint(0, len(X_te), (2, 1000), generator=torch.Generator().manual_seed(0))
    ok = y_te[pa] != y_te[pb]
    pa, pb = pa[ok][:500], pb[ok][:500]
    with torch.no_grad():
        mid_lat = m16.decode(0.5 * (m16.encode(X_te[pa])[0] + m16.encode(X_te[pb])[0]))
    scale = dict(real_test_digit=nearest(X_te), pixel_blend_t05=nearest(0.5 * (X_te[pa] + X_te[pb])),
                 latent_midpoint=nearest(mid_lat), n_random_pairs=len(pa))

    results = dict(
        env=dict(torch=torch.__version__, threads=torch.get_num_threads()),   # bit-exact only for this build and thread count
        d2=dict(history=hist2, n_params=sum(p.numel() for p in m2.parameters()), test=dict(rec=rec2, kl=kl2, neg_elbo=rec2 + kl2),
                val_kl_first=hist2[0]["val_kl"], val_kl_last=hist2[-1]["val_kl"],
                mu_var=mu_te.var(0).tolist(), mean_sigma=(0.5 * lv_te).exp().mean(0).tolist(),
                mu_range=[mu_te.min(0).values.tolist(), mu_te.max(0).values.tolist()]),
        d16_test=ex1,
        diff_d2_minus_d16=dict(rec=rec2 - ex1["rec"], kl=kl2 - ex1["kl"], neg_elbo=rec2 + kl2 - ex1["neg_elbo"]),
        kl_bits=dict(d16=ex1["kl"] / np.log(2), d2=kl2 / np.log(2), class_label=np.log2(10)),   # nats -> bits
        per_digit_test_rec=per_digit,
        knn15_test_accuracy=dict(d2=float((pred2 == y_te.numpy()).mean()), d16=float((pred16 == y_te.numpy()).mean())),
        knn15_recall_per_digit=dict(d2=(cm.diagonal() / cm.sum(1)).tolist(), d16=(cm16.diagonal() / cm16.sum(1)).tolist()),
        centroids=cent.tolist(), overlap_top5=overlap,
        grid=dict(z=ZG.tolist(), q=Q.tolist(), dq=dq, mass_per_cell=dq ** 2, inner_mass=(14 * dq) ** 2),
        interpolation=dict(index_a=ia, digit_a=a, index_b=ib, digit_b=b, t=t.flatten().tolist(),
                           nn_dist_pixel_row=d_pix, nn_dist_latent_row=d_lat, gap=gap,
                           largest_gap_pair_this_run=dict(index_a=best[0], index_b=best[1], gap=gaps[best][0]),
                           mean_gap_all_45_pairs=float(np.mean([g[0] for g in gaps.values()])),
                           min_gap=min(g[0] for g in gaps.values()), max_gap=max(g[0] for g in gaps.values()),
                           n_pairs_latent_closer=sum(g[0] > 0 for g in gaps.values()), scale=scale))
    (HERE / "results_ex2.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({k: v for k, v in results.items() if k not in ("d2",)}, indent=1))


if __name__ == "__main__":
    main()
