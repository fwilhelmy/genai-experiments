import torch
import torchvision
import matplotlib.pyplot as plt
from typing import Optional
import json
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def generate_images(
    model: torch.nn.Module,
    num_images: int = 64,
    device: Optional[torch.device] = None,
    show: bool = False,
    file_path: Optional[str] = None
) -> torch.Tensor:
    """
    Sample `num_images` from N(0,I) and decode them with a trained VAE.

    Args:
        model (nn.Module): trained VAE instance (with `decode(z)`).
        num_images (int): how many images to generate.
        device (torch.device, optional): where to run the sampling. 
            If None, inferred from model parameters.
        show (bool): whether to display the generated image grid.
        file_path (str, optional): filepath to save the generated grid image.
    Returns:
        Tensor of shape (num_images, 1, 28, 28) with values in [0,1].
    """
    model.eval()
    # infer device if not provided
    if device is None:
        device = next(model.parameters()).device
    model.to(device)

    with torch.no_grad():
        latent_dim = model.fc21.out_features
        z = torch.randn(num_images, latent_dim, device=device)
        recon_flat = model.decode(z)                   # (num_images, 784)
        images = recon_flat.view(num_images, 1, 28, 28) # (num_images,1,28,28)

    # optionally display/save a grid
    if show or file_path:
        grid = torchvision.utils.make_grid(images, nrow=8, padding=2)
        plt.figure(figsize=(6, 6))
        plt.imshow(grid.permute(1, 2, 0).cpu(), cmap='gray')
        plt.axis('off')
        if file_path:
            plt.savefig(file_path, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    return images

def plot_loss(history_path: str, save_path: str = None, show: bool = False):
    """
    Load training history from a JSON file and plot train/validation loss
    with integer y-ticks, a red line at y=104 (no label), x-ticks 1–N, and
    annotated final train/validation values.

    Args:
        history_path (str): Path to the `history.json` file containing
                            {"train_loss": [...], "val_loss": [...]}.
        save_path (str, optional): If provided, save the figure here
                                   instead of (or in addition to) showing it.
    """
    if not os.path.isfile(history_path):
        raise FileNotFoundError(f"No history file found at {history_path!r}")

    with open(history_path, 'r') as fp:
        history = json.load(fp)

    train_loss = history.get('train_loss', [])
    val_loss   = history.get('val_loss', [])

    if not train_loss or not val_loss:
        raise ValueError("history.json must contain non-empty 'train_loss' and 'val_loss' lists")

    num_epochs = len(train_loss)
    epochs     = range(1, num_epochs + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, marker='o', label='Train Loss')
    ax.plot(epochs, val_loss,   marker='o', label='Validation Loss')

    # horizontal red line at y=104 (no label)
    ax.axhline(y=104, color='red', linestyle='--')

    # integer-only ticks
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks(range(1, num_epochs + 1))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # annotate final values in the top-right corner
    final_tr = train_loss[-1]
    final_va = val_loss[-1]
    text = f"Final Train: {final_tr:.4f}\nFinal Val:   {final_va:.4f}"
    ax.text(
        0.985, 0.8, text,
        transform=ax.transAxes,
        fontsize=10,
        ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6)
    )

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved loss plot to {save_path}")
    if show:
        plt.show()

import math
import os
from typing import Optional, Tuple

import torch
from torch import nn
import matplotlib.pyplot as plt


def generate_latent_traversals(
    model: nn.Module,
    n_factors: int = 20,
    n_steps: int = 5,
    eps: float = 2.0,
    device: Optional[torch.device] = None,
    data_sample: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Generate latent traversals for a VAE.

    If `data_sample` is provided, encodes it to get the base z; else samples z0 ~ N(0, I).

    Returns:
        Tensor of shape (n_factors, n_steps, 1, H, W) on CPU.
    """
    model.eval()
    device = device or next(model.parameters()).device
    model.to(device)

    # Prepare base latent vector
    if data_sample is not None:
        # flatten and encode
        x_flat = data_sample.to(device).view(1, -1)
        mu, logvar = model.encode(x_flat)
        z0 = model.reparameterize(mu, logvar)
    else:
        z0 = torch.randn(1, n_factors, device=device)

    traversals = []
    half_span = (n_steps - 1) / 2

    with torch.no_grad():
        for dim in range(n_factors):
            row = []
            for step in range(n_steps):
                z = z0.clone()
                z[0, dim] += (step - half_span) * eps
                recon = model.decode(z)         # (1, P)
                P = recon.size(1)
                side = int(math.sqrt(P))
                img = recon.view(1, 1, side, side)
                row.append(img)
            # stack steps → (n_steps, 1, H, W)
            traversals.append(torch.cat(row, dim=0))

    # stack dims → (n_factors, n_steps, 1, H, W)
    return torch.stack(traversals, dim=0).cpu()


def plot_traversals(
    traversals: torch.Tensor,
    eps: float,
    figsize: Tuple[float, float] = (4.8, 20),
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot a grid of latent traversals.

    Args:
        traversals: (n_factors, n_steps, 1, H, W) tensor.
        eps:        offset magnitude (for title).
        figsize:    figure size.
        save_path:  if given, saves the figure to this file.
        show:       if True, calls plt.show().
    """
    n_factors, n_steps, _, H, W = traversals.shape
    center = n_steps // 2
    mid_x = (W - 1) / 2
    mid_y = (H - 1) / 2

    fig, axes = plt.subplots(
        n_factors, n_steps,
        figsize=figsize,
        gridspec_kw={'wspace': 0},
        squeeze=False
    )

    fig.suptitle(f'Latent Traversals (ε = {eps})', fontsize=16, y=1.02)
    fig.supylabel('Latent Space', x=0.0, fontsize=12)

    for i in range(n_factors):
        for k in range(n_steps):
            ax = axes[i, k]
            ax.imshow(traversals[i, k, 0], cmap='gray', vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])

            # Top row: x-tick label = offset
            if i == 0:
                ax.xaxis.set_ticks([mid_x])
                ax.xaxis.set_ticklabels([f'{k - center:+d}ε'], fontsize=10)
                ax.xaxis.tick_top()
                ax.tick_params(axis='x', length=0)

            # First col: y-tick label = latent index
            if k == 0:
                ax.yaxis.set_ticks([mid_y])
                ax.yaxis.set_ticklabels([f'z[{i}]'], fontsize=10, rotation=90, va='center')
                ax.tick_params(axis='y', length=0)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')

    if show:
        plt.show()

    plt.close(fig)