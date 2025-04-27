import numpy as np
import torch
import matplotlib.pyplot as plt
from q2_trainer_ddpm import experiment2

def plot_samples_across_epochs(
    epochs: list[int] = [5, 10, 15, 20],
    n_samples: int | None = None,
    n_steps: int | None = None,
    seed: int | None = None,
    figsize_per_sample: tuple = (2, 2),
    *,
    file_path: str | None = None,
    show: bool = False,
):
    """
    For each epoch in `epochs`, loads the DDPM checkpoint, draws `n_samples`
    with `n_steps` diffusion steps, and lays them out in a grid:
      • cols ← epochs
      • rows ← sample index

    Args:
        epochs (list[int]): which epochs to load.
        n_samples (int|None): images per epoch; defaults to trainer.args.n_samples.
        n_steps (int|None): diffusion steps per sample; defaults to trainer.args.n_steps.
        seed (int|None): seed for torch.manual_seed.
        figsize_per_sample (tuple): (width, height) in inches per cell.
        file_path (str|None): if set, path to save the figure (PNG, PDF, etc.).
        show (bool): if True, displays the plot with plt.show().
    """
    # optional reproducibility
    if seed is not None:
        torch.manual_seed(seed)

    # instantiate trainers & defaults
    trainers = [experiment2(train=False, checkpoint_epoch=ep) for ep in epochs]
    if n_samples is None:
        n_samples = trainers[0].args.n_samples
    if n_steps is None:
        n_steps = trainers[0].args.n_steps

    # collect samples
    all_samples = []
    for trainer in trainers:
        imgs = trainer.sample(
            n_steps=n_steps,
            n_samples=n_samples,
            set_seed=False,
            show=False,
            save=False
        ).cpu()  # [n_samples, C, H, W]
        all_samples.append(imgs)

    # build grid
    cols = len(epochs)
    rows = n_samples
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(figsize_per_sample[0] * cols, figsize_per_sample[1] * rows),
        squeeze=False
    )

    for col_idx, epoch in enumerate(epochs):
        batch = all_samples[col_idx]
        for row_idx in range(rows):
            ax = axes[row_idx, col_idx]
            img = batch[row_idx].squeeze().numpy()
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(f"Epoch {epoch}", fontsize=12)

    plt.tight_layout()

    # save if requested
    if file_path:
        fig.savefig(file_path, bbox_inches="tight")
        plt.close(fig)

    # show if requested
    if show:
        plt.show()

    plt.close(fig)  # prevent double-display if show=False