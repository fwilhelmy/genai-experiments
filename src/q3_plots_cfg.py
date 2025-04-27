import torch
import matplotlib.pyplot as plt
from q3_trainer_cfg import experiment3

def plot_evolution_across_epochs(
    epochs: list[int] = [5, 10, 15, 20],
    n_steps: int | None = None,
    labels: torch.Tensor | None = None,
    seed: int = 42,
    *,
    file_path: str | None = None,
    show: bool = False,
):
    """
    Samples the CFG diffusion model at specific epochs and plots the results.

    Columns → epochs; Rows → fixed labels.

    Args:
        epochs      (list[int]): which checkpoint epochs to load & sample.
        n_steps     (int|None): number of reverse diffusion steps to use.
                               If None, uses trainer.args.n_steps.
        labels      (torch.Tensor|None): tensor of shape (n_samples,) with labels [0–9].
                               If None, will sample uniformly at random once,
                               using trainer.args.n_samples as length.
        seed        (int): RNG seed for reproducibility of labels and sampling.
        file_path   (str|None): path to save the final figure (PNG, PDF, etc.).  
                               If None, no file is written.
        show        (bool): whether to display the plot interactively.
    """
    # 1) Fix RNG
    torch.manual_seed(seed)

    # 2) Grab device & default sample count
    tmp_trainer = experiment3(train=False, checkpoint_epoch=epochs[0])
    device = tmp_trainer.args.device
    default_n = tmp_trainer.args.n_samples

    # 3) Prepare labels
    if labels is None:
        n = default_n
        labels = torch.randint(0, 10, (n,), device=device)
    else:
        labels = labels.to(device)
        n = labels.numel()
        assert labels.dim() == 1, "labels must be a 1-D tensor"
    
    all_samples: list[torch.Tensor] = []
    for ep in epochs:
        trainer = experiment3(train=False, checkpoint_epoch=ep)
        trainer.args.n_samples = n

        # sample
        z_t = trainer.sample(
            labels=labels,
            n_steps=n_steps,
            set_seed=True,
            show=False,
            save=False,
        )
        all_samples.append(z_t)

    # 4) Plot: rows = samples/labels, cols = epochs
    fig, axes = plt.subplots(
        n, len(epochs),
        figsize=(len(epochs)*3, n*3),
        squeeze=False
    )
    for col, z_t in enumerate(all_samples):
        for row in range(n):
            ax = axes[row][col]
            img = z_t[row].squeeze().cpu().numpy()
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            if row == 0:
                ax.set_title(f'Epoch {epochs[col]}')
            if col == 0:
                ax.set_ylabel(f'Label {labels[row].item()}', rotation=90, va='center')

    plt.tight_layout()

    # 5) Save if requested
    if file_path:
        fig.savefig(file_path, bbox_inches='tight')
        plt.close(fig)  # prevent double-display if show=False

    # 6) Show if requested
    if show:
        plt.show()

    plt.close(fig)  # prevent double-display if show=False
