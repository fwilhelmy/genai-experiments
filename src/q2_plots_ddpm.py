import numpy as np
import torch
import matplotlib.pyplot as plt
from q2_trainer_ddpm import experiment2

def plot_samples_across_epochs(
    epochs=[5, 10, 15, 20],
    n_samples=None,
    n_steps=None,
    seed=None,
    figsize_per_sample=(2, 2)
):
    """
    For each epoch in `epochs`, loads the DDPM checkpoint, draws `n_samples`
    with `n_steps` diffusion steps, and displays them in a grid:
      • cols ← epochs
      • rows ← sample index

    Args:
        epochs (list of int): which epochs to load.
        n_samples (int, optional): how many images per epoch.
            Defaults to trainer.args.n_samples.
        n_steps (int, optional): how many diffusion steps when sampling.
            Defaults to trainer.args.n_steps.
        seed (int, optional): torch.manual_seed for reproducibility.
        figsize_per_sample (tuple): (width, height) in inches per image cell.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # 1) Instantiate trainers and infer defaults
    trainers = [experiment2(train=False, checkpoint_epoch=ep) for ep in epochs]
    if n_samples is None:
        n_samples = trainers[0].args.n_samples
    if n_steps is None:
        n_steps = trainers[0].args.n_steps

    # 2) Collect samples for each epoch
    all_samples = []
    for trainer in trainers:
        imgs = trainer.sample(
            n_steps=n_steps,
            n_samples=n_samples,
            set_seed=False,
            show=False,
            save=False
        )  # returns Tensor [n_samples, C, H, W]
        all_samples.append(imgs.cpu())

    # 3) Build the plot grid
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
    plt.show()
