"""Sub-module to define vizualisation functions"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
import umap
from typing import NoReturn
import numpy as np
import os
from pathlib import Path

from.vae import M1_M2, SVAE, SVAE_M2, predict_classes_loader, arccos, arccos_with_grad

def save_plot(save_path):
    if "/" in save_path:
        parent_path = "/".join(save_path.split("/")[:-1])
        os.makedirs(parent_path, exist_ok=True)
    plot_ext = os.path.splitext(save_path)[-1][1:]
    plt.savefig(save_path, 
                bbox_inches='tight', 
                format=plot_ext)
    print(f"\nPlot saved at: {save_path}.")

def plot_latent_space(true_2d_latent: np.ndarray, 
                      learned_2d_latent: np.ndarray, 
                      labels: np.ndarray, 
                      losses: list,
                      save_path: str = None,
                      ) -> NoReturn:
    """
    Function to plot and compare true vs learned latent spaces in 2D.
    """
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.scatter(true_2d_latent[:, 0], true_2d_latent[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.title('données originales s1')
    plt.axis('equal')

    plt.subplot(1, 3, 2)
    plt.scatter(learned_2d_latent[:, 0], learned_2d_latent[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.title('espace latent s-vae')
    plt.axis('equal')

    plt.subplot(1, 3, 3)
    plt.plot(losses)
    plt.title('loss')
    plt.xlabel('epoch')

    plt.tight_layout()
    if save_path is not None:
        save_plot(save_path)
    else:
        plt.show()

def compare_latent_space(true_2d_latent: np.ndarray, 
                      svae_learned_2d_latent: np.ndarray, 
                      nvae_learned_2d_latent: np.ndarray,
                      labels: np.ndarray, 
                      svae_losses: dict,
                      nvae_losses: dict,
                      save_path: str = None,
                      addon_svae: str = '',
                      addon_nvae: str = '',
                      ) -> NoReturn:
    """
    Function to plot and compare true vs learned latent spaces in 2D.
    """
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 3, 1)
    plt.scatter(true_2d_latent[:, 0], 
                true_2d_latent[:, 1], 
                c=labels, cmap='viridis', alpha=0.5)
    plt.title('données originales s1')
    plt.axis('equal')

    plt.subplot(2, 3, 2)
    plt.scatter(svae_learned_2d_latent[:, 0], 
                svae_learned_2d_latent[:, 1], 
                c=labels, cmap='viridis', alpha=0.5)
    plt.title('espace latent S-VAE')
    plt.axis('equal')

    plt.subplot(2, 3, 3)
    plt.plot(svae_losses['train'], label='train')
    plt.plot(svae_losses['val'], label='val')
    plt.legend()
    plt.title(f'S-VAE training & validation loss\n{addon_svae}')
    plt.xlabel('epoch')

    plt.subplot(2, 3, 4)
    plt.scatter(true_2d_latent[:, 0], 
                true_2d_latent[:, 1], 
                c=labels, cmap='viridis', alpha=0.5)
    plt.title('données originales s1')
    plt.axis('equal')

    plt.subplot(2, 3, 5)
    plt.scatter(nvae_learned_2d_latent[:, 0], 
                nvae_learned_2d_latent[:, 1], 
                c=labels, cmap='viridis', alpha=0.5)
    plt.title('espace latent N-VAE')
    plt.axis('equal')

    plt.subplot(2, 3, 6)
    plt.plot(nvae_losses['train'], label='train')
    plt.plot(nvae_losses['val'], label='val')
    plt.legend()
    plt.title(f'N-VAE training & validation loss\n{addon_nvae}')
    plt.xlabel('epoch')

    plt.tight_layout()
    if save_path is not None:
        save_plot(save_path)
    else:
        plt.show()


def _to_numpy(x):
    """Convert torch tensors / lists to numpy arrays (no-op if already numpy)."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _suffix_path(save_path: str, suffix: str) -> str:
    """Insert suffix before extension: /a/b.png + _X -> /a/b_X.png"""
    p = Path(save_path)
    return str(p.with_name(f"{p.stem}_{suffix}{p.suffix}"))


def _embed_umap(Z, mode="normal"):
    if umap is None:
        raise ImportError(
            "UMAP is not available. Install it with `pip install umap-learn`."
        )
    reducer = umap.UMAP(n_components=2, n_jobs=os.cpu_count() // 2, 
                        metric="euclidean" if mode == "normal" else arccos)
    return reducer.fit_transform(Z)


def _embed_tsne(Z, mode="normal"):
    # TSNE defaults can be slow; keep them reasonable and robust.
    n = Z.shape[0]
    perplexity = min(30, max(5, (n - 1) // 3))  # safe-ish heuristic
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        n_jobs=os.cpu_count() // 2,
        metric="euclidean" if mode == "normal" else arccos,
    )
    return tsne.fit_transform(Z)


def _plot_2x2_embeddings(Z, y_true, y_pred, title_prefix: str, mode):
    """
    2 rows: UMAP, t-SNE
    2 cols: True labels, Pred labels
    """
    Z = _to_numpy(Z)
    y_true = _to_numpy(y_true).reshape(-1)
    y_pred = _to_numpy(y_pred).reshape(-1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=False, sharey=False)

    addon = "arccos" if mode != "normal" else "euclidean"

    # --- UMAP row ---
    umap_xy = _embed_umap(Z, mode)
    for j, (lab, name) in enumerate([(y_true, "True"), (y_pred, "Pred")]):
        ax = axes[0, j]
        sns.scatterplot(
            x=umap_xy[:, 0],
            y=umap_xy[:, 1],
            hue=lab,
            ax=ax,
            s=18,
            linewidth=0,
            alpha=0.9,
            palette="tab10",
        )
        ax.set_title(f"{title_prefix} UMAP ({name},{addon})")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.legend(title="Class", loc="best", fontsize=9, title_fontsize=10)

    # --- t-SNE row ---
    tsne_xy = _embed_tsne(Z, mode)
    for j, (lab, name) in enumerate([(y_true, "True"), (y_pred, "Pred")]):
        ax = axes[1, j]
        sns.scatterplot(
            x=tsne_xy[:, 0],
            y=tsne_xy[:, 1],
            hue=lab,
            ax=ax,
            s=18,
            linewidth=0,
            alpha=0.9,
            palette="tab10",
        )
        ax.set_title(f"{title_prefix} t-SNE ({name},{addon})")
        ax.set_xlabel("tSNE-1")
        ax.set_ylabel("tSNE-2")
        ax.legend(title="Class", loc="best", fontsize=9, title_fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_latent_classes(model, loader, mode, save_path=None):
    assert isinstance(model, M1_M2), (
        f"Unsupported model class: Only supports {M1_M2} but got {model.__class__}"
    )
    if isinstance(model.vae_m1, SVAE):
        M1_MODE = 'svae'
    else:
        M1_MODE = 'normal'

    if isinstance(model.vae_m2, SVAE_M2):
        M2_MODE = 'svae'
    else:
        M2_MODE = 'normal'

    Y, Y_hat, Z1, Z2 = predict_classes_loader(model, loader, mode, return_latent=True)

    acc = accuracy_score(Y, Y_hat)

    # --- M1 latent (Z1) ---
    fig_m1 = _plot_2x2_embeddings(
        Z=Z1,
        y_true=Y,
        y_pred=Y_hat,
        title_prefix="M1 latent (Z1):", 
        mode=M1_MODE,
    )
    plt.suptitle(f"M1 Latent space with M2 classification.\nTest accuracy: {acc*100:.2f}%",
                 y=0.99)
    if save_path is not None:
        out_m1 = _suffix_path(save_path, "M1_latent")
        os.makedirs(os.path.dirname(out_m1) or ".", exist_ok=True)
        fig_m1.savefig(out_m1, dpi=300, bbox_inches="tight")

    # --- M2 latent (Z2) ---
    fig_m2 = _plot_2x2_embeddings(
        Z=Z2,
        y_true=Y,
        y_pred=Y_hat,
        title_prefix="M2 latent (Z2):",
        mode=M2_MODE,
    )
    plt.suptitle(f"M2 Latent space with M2 classification.\nTest accuracy: {acc*100:.2f}%",
                 y=0.99)
    if save_path is not None:
        out_m2 = _suffix_path(save_path, "M2_latent")
        os.makedirs(os.path.dirname(out_m2) or ".", exist_ok=True)
        fig_m2.savefig(out_m2, dpi=300, bbox_inches="tight")

    return fig_m1, fig_m2


