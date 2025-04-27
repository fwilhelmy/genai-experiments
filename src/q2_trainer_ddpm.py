import os
import json
import torch
from matplotlib import pyplot as plt 
from tqdm import tqdm
from torch.amp import GradScaler, autocast
import copy
import torch.nn.functional as F
from typing import List, Optional, Sequence
import numpy as np
import torch
import matplotlib.pyplot as plt

from ddpm_utils.args import args
from ddpm_utils.dataset import MNISTDataset
from ddpm_utils.unet import UNet, load_weights
from q2_ddpm import DenoiseDiffusion

# ensure all outputs go under results/experiment2
RESULTS_DIR = "results/experiment2"
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
IMAGE_DIR = os.path.join(RESULTS_DIR, "images")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

torch.manual_seed(42)

def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

class Trainer:
    def __init__(self, args, eps_model, diffusion_model):
        self.eps_model = eps_model.to(args.device)
        self.diffusion = diffusion_model
        self.optimizer = torch.optim.Adam(
            self.eps_model.parameters(), lr=args.learning_rate
        )
        self.args = args
        self.current_epoch = 0
        self.ema = EMA(0.995)
        self.ema_model = copy.deepcopy(self.eps_model).eval().requires_grad_(False)

    def train_epoch(self, dataloader, scaler):
        current_lr = round(self.optimizer.param_groups[0]['lr'], 5)
        i = 0
        running_loss = 0.
        with tqdm(range(len(dataloader)), desc=f'Epoch : - lr: - Loss :') as progress:
            for x0 in dataloader:
                i += 1
                x0 = x0.to(self.args.device)
                with autocast(device_type=args.device, enabled=self.args.fp16_precision):
                    loss = self.diffusion.loss(x0)
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.ema.step_ema(self.ema_model, self.eps_model)
                running_loss += loss.item()
                self.loss_per_iter.append(running_loss / i)
                progress.update()
                progress.set_description(
                    f'Epoch: {self.current_epoch}/{self.args.epochs} - '
                    f'lr: {current_lr} - Loss: {round(running_loss / i, 2)}'
                )
            progress.set_description(
                f'Epoch: {self.current_epoch}/{self.args.epochs} - '
                f'lr: {current_lr} - Loss: {round(running_loss / len(dataloader), 2)}'
            )
            self.scheduler.step()

    def train(self, dataloader):
        scaler = GradScaler(device=self.args.device, enabled=self.args.fp16_precision)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        start_epoch = self.current_epoch
        self.loss_per_iter = []
        for current_epoch in range(start_epoch, self.args.epochs):
            self.current_epoch = current_epoch
            self.train_epoch(dataloader, scaler)

            if current_epoch % self.args.show_every_n_epochs == 0:
                self.sample()

            if (current_epoch + 1) % self.args.save_every_n_epochs == 0 or current_epoch == self.args.epochs - 1:
                self.save_model()

    def sample(self, n_steps=None, n_samples=None, set_seed=False, show=False, save=True):
        if set_seed:
            torch.manual_seed(42)
        if n_steps is None:
            n_steps = self.args.n_steps
        if n_samples is None:
            n_samples = self.args.n_samples

        with torch.no_grad():
            x = torch.randn(
                [
                    n_samples,
                    self.args.image_channels,
                    self.args.image_size,
                    self.args.image_size,
                ],
                device=self.args.device,
            )
            
            for curr_t in tqdm(reversed(range(n_steps)), desc="Sampling"):
                t = torch.full((n_samples,), curr_t, device=self.args.device, dtype=torch.long)
                x = self.diffusion.p_sample(x, t)

        print(f"Showing/saving samples from epoch {self.current_epoch}")
        self.show_save(
            x, show=show, save=save,
            file_name=f"DDPM_epoch_{self.current_epoch}.png"
        )
        return x

    def save_model(self):
        ckpt = {
            'epoch': self.current_epoch,
            'model_state_dict': self.eps_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        filename = os.path.join(
            CHECKPOINT_DIR,
            f"ddpm_epoch_{self.current_epoch:03d}.pt"
        )
        torch.save(ckpt, filename)
        print(f"[Checkpoint] model saved to {filename}")

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=args.device)
        self.eps_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']

    def show_save(self, img_tensor, show=True, save=True, file_name="sample.png"):
        fig, axs = plt.subplots(3, 3, figsize=(10, 10))
        assert img_tensor.shape[0] >= 9, "Number of images should be at least 9"
        img_tensor = img_tensor[:9]
        for i, ax in enumerate(axs.flat):
            img = img_tensor[i].squeeze().cpu().numpy()
            ax.imshow(img, cmap="gray")
            ax.axis("off")
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(IMAGE_DIR, file_name))
        if show:
            plt.show()
        plt.close(fig)

    def generate_intermediate_samples(
        self,
        n_samples: int = 4,
        img_size: int = 32,
        steps_to_show: Sequence[int] = (0, 999),
        n_steps: Optional[int] = None,
        set_seed: bool = False,
        *,
        file_path: str | None = None,
        show: bool = False,
    ) -> list[np.ndarray]:
        """
        Run reverse diffusion, capture snapshots at given timesteps, and plot a grid.

        Rows = sample chains; Columns = timesteps in steps_to_show.

        Returns:
            List of arrays shaped (n_samples, 1, img_size, img_size), in order of sorted steps_to_show.

        Args:
            n_samples   (int): number of sample chains.
            img_size    (int): height/width of each square image.
            steps_to_show (Sequence[int]): timesteps at which to capture snapshots.
            n_steps     (int|None): total diffusion steps (defaults to self.args.n_steps).
            set_seed    (bool): whether to fix the RNG seed (42).
            file_path   (str|None): path to save the plotted grid (e.g. "grid.png").  
                                If None, no file is written.
            show        (bool): whether to display the plot with plt.show().
        """
        if set_seed:
            torch.manual_seed(42)

        total_steps = n_steps if n_steps is not None else self.args.n_steps
        valid_steps = sorted({t for t in steps_to_show if 0 <= t < total_steps}, reverse=True)
        if not valid_steps:
            raise ValueError(f"No valid timesteps in [0, {total_steps}): {steps_to_show!r}")

        # 1) generate noise and run reverse diffusion, capturing snapshots
        with torch.no_grad():
            x = torch.randn(n_samples, 1, img_size, img_size, device=self.args.device)
            snapshots: dict[int, torch.Tensor] = {}
            for t in reversed(range(total_steps)):
                timesteps = torch.full((n_samples,), t, device=self.args.device, dtype=torch.long)
                x = self.diffusion.p_sample(x, timesteps)
                if t in valid_steps:
                    snapshots[t] = x.cpu()

        # 2) assemble numpy arrays in sorted order
        images = [snapshots[t].numpy() for t in valid_steps]

        # 3) plot grid
        cols = len(images)
        fig, axes = plt.subplots(
            n_samples, cols,
            figsize=(cols * 2, n_samples * 2),
            squeeze=False
        )
        for i in range(n_samples):
            for j, t in enumerate(valid_steps):
                ax = axes[i][j]
                ax.imshow(images[j][i].squeeze(), cmap="gray")
                ax.axis("off")
                if i == 0:
                    ax.set_title(f"t={t}", fontsize=10)

        plt.tight_layout()

        # 4) save if requested
        if file_path:
            fig.savefig(file_path, bbox_inches="tight")

        # 5) show if requested
        if show:
            plt.show()

        plt.close(fig)

        return images



    def save_history(self):
        """Save full training loss history once, at the end."""
        json_path = os.path.join(CHECKPOINT_DIR, "loss_history.json")
        with open(json_path, 'w') as jf:
            json.dump(self.loss_per_iter, jf)
        print(f"[Training] loss history saved to {json_path}")

def experiment2(train: bool = True, checkpoint_epoch: int = None):
    # instantiate and (optionally) pre-load your base model
    eps_model = UNet(c_in=1, c_out=1)
    eps_model = load_weights(eps_model, args.MODEL_PATH)

    diffusion_model = DenoiseDiffusion(
        eps_model=eps_model,
        n_steps=args.n_steps,
        device=args.device,
    )

    trainer = Trainer(args, eps_model, diffusion_model)

    dataloader = torch.utils.data.DataLoader(
        MNISTDataset(),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )

    if not train:
        # default to last epoch if none specified
        epoch_to_load = checkpoint_epoch if checkpoint_epoch is not None else args.epochs - 1
        ckpt_path = os.path.join(
            CHECKPOINT_DIR,
            f"ddpm_epoch_{epoch_to_load:03d}.pt"
        )
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

        trainer.load_model(ckpt_path)
        print(f"[Checkpoint] Loaded model at epoch {trainer.current_epoch} from {ckpt_path}")
        return trainer

    # otherwise, run training as before
    trainer.train(dataloader)
    trainer.save_history()
    return trainer

if __name__ == "__main__":
    experiment2()
