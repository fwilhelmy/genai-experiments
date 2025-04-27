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
    figsize: Tuple[float, float] = (4.8, 10),
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot latent traversals in groups of 10 dimensions per figure.

    Args:
        traversals: (n_factors, n_steps, 1, H, W) tensor.
        eps:        offset magnitude (for title).
        figsize:    base figure size (height will scale with up to 10 rows).
        save_path:  if given, base filepath – '_partX' will be appended.
        show:       if True, calls plt.show().
    """
    n_factors, n_steps, _, H, W = traversals.shape
    center = n_steps // 2
    mid_x = (W - 1) / 2
    mid_y = (H - 1) / 2

    # iterate in blocks of 10 latent dims
    for block_start in range(0, n_factors, 10):
        block_end = min(block_start + 10, n_factors)
        block_size = block_end - block_start

        fig, axes = plt.subplots(
            block_size, n_steps,
            figsize=figsize,
            gridspec_kw={'wspace': 0},
            squeeze=False
        )

        for row_idx, i in enumerate(range(block_start, block_end)):
            for k in range(n_steps):
                ax = axes[row_idx, k]
                ax.imshow(traversals[i, k, 0], cmap='gray', vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])

                # Top row: x-tick label = offset
                if row_idx == 0:
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
            # append block index to filename before extension
            base, ext = os.path.splitext(save_path)
            part_path = f"{base}_part{block_start//10}{ext}"
            os.makedirs(os.path.dirname(part_path), exist_ok=True)
            fig.savefig(part_path, bbox_inches='tight')

        if show:
            plt.show()

        plt.close(fig)

import torch
import matplotlib.pyplot as plt
from typing import Optional, Tuple

def plot_interpolation_comparison(
    model: torch.nn.Module,
    endpoints: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    num_steps: int = 11,
    device: Optional[torch.device] = None,
    figsize: tuple = (12, 3),
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Compare latent-space vs. data-space interpolation between two images.

    Args:
        model      : trained VAE with .encode(x)->(mu,logvar) and .decode(z)->flattened image.
        endpoints  : tuple (x0, x1) of endpoint images [1,1,28,28]; if None, samples z0,z1~N(0,I) & decodes them.
        num_steps  : number of α values (default 11 for α=0,0.1,...,1).
        device     : torch device; inferred from model if None.
        figsize    : (width,height) for the overall figure.
        save_path  : where to save the final plot (optional).
        show       : whether to call plt.show().
    """
    model.eval()
    device = device or next(model.parameters()).device
    model.to(device)

    # latent dimensionality
    latent_dim = model.fc21.out_features

    # Unpack or sample endpoints
    if endpoints is None:
        # sample two random z's and decode
        z0 = torch.randn(1, latent_dim, device=device)
        z1 = torch.randn(1, latent_dim, device=device)
        with torch.no_grad():
            x0 = model.decode(z0).view(1,1,28,28).cpu()
            x1 = model.decode(z1).view(1,1,28,28).cpu()
    else:
        x0, x1 = endpoints
        # encode to get z0, z1
        x0 = x0.to(device); x1 = x1.to(device)
        flat0, flat1 = x0.view(1,-1), x1.view(1,-1)
        with torch.no_grad():
            mu0, logvar0 = model.encode(flat0)
            mu1, logvar1 = model.encode(flat1)
            z0 = model.reparameterize(mu0, logvar0)
            z1 = model.reparameterize(mu1, logvar1)
        # keep CPU copies for pixel interp
        x0, x1 = x0.cpu(), x1.cpu()

    # interpolation weights
    alphas = torch.linspace(0, 1, steps=num_steps, device=device)

    # latent-space decode
    lat_images = []
    for α in alphas:
        z = α*z0 + (1-α)*z1
        with torch.no_grad():
            img = model.decode(z).view(1,1,28,28).cpu()
        lat_images.append(img)
    lat_images = torch.cat(lat_images, dim=0)

    # pixel-space mix
    data_images = []
    for α in alphas:
        img = α*x0 + (1-α)*x1
        data_images.append(img)
    data_images = torch.cat(data_images, dim=0)

    # plot
    fig, axes = plt.subplots(2, num_steps, figsize=figsize, squeeze=False)
    for i in range(num_steps):
        axes[0,i].imshow(lat_images[i,0], cmap='gray', vmin=0, vmax=1)
        axes[0,i].axis('off')
        axes[1,i].imshow(data_images[i,0], cmap='gray', vmin=0, vmax=1)
        axes[1,i].axis('off')
        axes[0,i].set_title(f"α={alphas[i]:.1f}", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)