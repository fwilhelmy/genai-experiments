import numpy as np
import torch
import matplotlib.pyplot as plt
from ddpm_training import experiment2

def plot_samples_across_epochs(
    epochs: list[int] = [5, 10, 15, 20],
    n_steps: int | None = None,
    seed: int | None = None,
    figsize_per_sample: tuple = (2, 2),
    file_path: str | None = None,
    show: bool = False,
):
    """
    For each epoch in `epochs`, loads the DDPM checkpoint, draws 10 samples
    from the Gaussian prior using `n_steps` diffusion steps, and lays them out
    in two grids (part1: sample 0–4, part2: sample 5–9).

    Columns → epochs; Rows → sample indices within each part.

    Args:
        epochs             (list[int]): which checkpoint epochs to load.
        n_steps            (int|None): diffusion steps per sample; defaults to trainer.args.n_steps.
        seed               (int|None): torch.manual_seed for reproducibility.
        figsize_per_sample (tuple): (width, height) in inches per cell.
        file_path          (str|None): if set, base path to save the figures.
                                       Two files will be written: suffix `_part1` and `_part2`.
        show               (bool): if True, displays the plots with plt.show().
    """
    # 1) Reproducibility
    if seed is not None:
        torch.manual_seed(seed)

    # 2) Instantiate trainers & infer defaults
    trainers = [experiment2(train=False, checkpoint_epoch=ep) for ep in epochs]
    if n_steps is None:
        n_steps = trainers[0].args.n_steps

    # 3) Always sample exactly 10 images from N(0,I)
    n_samples = 10
    all_samples = []
    for trainer in trainers:
        imgs = trainer.sample(
            n_steps=n_steps,
            n_samples=n_samples,
            set_seed=False,
            show=False,
            save=False
        ).cpu()  # shape: [10, C, H, W]
        all_samples.append(imgs)

    # 4) Split into two parts: indices 0–4 and 5–9
    parts = {
        'part1': list(range(0, 5)),
        'part2': list(range(5, 10)),
    }

    # 5) Plot each part
    for part_name, idxs in parts.items():
        rows = len(idxs)
        cols = len(epochs)
        fig, axes = plt.subplots(
            rows, cols,
            figsize=(figsize_per_sample[0] * cols, figsize_per_sample[1] * rows),
            squeeze=False
        )

        for col_idx, epoch in enumerate(epochs):
            batch = all_samples[col_idx]
            for row_idx, sample_idx in enumerate(idxs):
                ax = axes[row_idx, col_idx]
                img = batch[sample_idx].squeeze().numpy()
                ax.imshow(img, cmap="gray")
                ax.axis("off")
                if row_idx == 0:
                    ax.set_title(f"Epoch {epoch}", fontsize=12)

        plt.tight_layout()

        # 6) Save if requested (appends _part1/_part2)
        if file_path:
            base, ext = file_path.rsplit('.', 1)
            fig.savefig(f"{base}_{part_name}.{ext}", bbox_inches="tight")

        # 7) Show if requested
        if show:
            plt.show()
        plt.close(fig)
