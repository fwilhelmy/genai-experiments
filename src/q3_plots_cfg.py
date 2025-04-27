import torch
import matplotlib.pyplot as plt
from q3_trainer_cfg import experiment3

def plot_evolution_across_epochs(
    epochs: list[int] = [5, 10, 15, 20],
    n_steps: int | None = None,
    seed: int = 42,
    file_path: str | None = None,
    show: bool = False,
):
    """
    Samples the CFG diffusion model at specific epochs and plots the results in two parts:
      • part1: labels 0–4
      • part2: labels 5–9

    Columns → epochs; Rows → fixed labels.

    Args:
        epochs      (list[int]): which checkpoint epochs to load & sample.
        n_steps     (int|None): number of reverse diffusion steps to use.
                               If None, uses trainer.args.n_steps.
        seed        (int): RNG seed for reproducibility.
        file_path   (str|None): path to save the final figures (PNG, PDF, etc.).
                               Two files will be written: suffix _part1 and _part2.
        show        (bool): whether to display the plots interactively.
    """
    # 1) Fix RNG
    torch.manual_seed(seed)

    # 2) Prepare trainer and device
    tmp_trainer = experiment3(train=False, checkpoint_epoch=epochs[0])
    device = tmp_trainer.args.device

    # 3) Always use labels 0 through 9
    labels = torch.arange(0, 10, device=device)
    n = labels.numel()

    # 4) Sample for each epoch
    all_samples: list[torch.Tensor] = []
    for ep in epochs:
        trainer = experiment3(train=False, checkpoint_epoch=ep)
        trainer.args.n_samples = n
        z_t = trainer.sample(
            labels=labels,
            n_steps=n_steps,
            set_seed=True,
            show=False,
            save=False,
        )
        all_samples.append(z_t)

    # 5) Define the two parts
    parts = {
        'part1': list(range(0, 5)),
        'part2': list(range(5, 10)),
    }

    # 6) Plot each part separately
    for part_name, idxs in parts.items():
        fig, axes = plt.subplots(
            len(idxs), len(epochs),
            figsize=(len(epochs) * 3, len(idxs) * 3),
            squeeze=False
        )
        for col, z_t in enumerate(all_samples):
            for row_i, label_idx in enumerate(idxs):
                ax = axes[row_i][col]
                img = z_t[label_idx].squeeze().cpu().numpy()
                ax.imshow(img, cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if row_i == 0:
                    ax.set_title(f'Epoch {epochs[col]}')
                if col == 0:
                    ax.set_ylabel(f'Label {labels[label_idx].item()}', rotation=90, fontsize=12)
        plt.tight_layout()

        # 7) Save if requested
        if file_path:
            base, ext = file_path.rsplit('.', 1)
            fig.savefig(f"{base}_{part_name}.{ext}", bbox_inches='tight')

        # 8) Show if requested
        if show:
            plt.show()
        plt.close(fig)
