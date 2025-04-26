import os
import json
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast

from cfg_utils.args import *
from cfg_utils.dataset import *
from cfg_utils.unet import *
from q3_cfg_diffusion import CFGDiffusion

import numpy as np 
import copy 

# ensure all outputs go under results/experiment3
RESULTS_DIR = "results/experiment3"
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
IMAGE_DIR = os.path.join(RESULTS_DIR, "images")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

torch.manual_seed(42)

def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))

class EMA:
    """Exponential Moving Average class to improve training"""
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
        current_lr = round(self.optimizer.param_groups[0]['lr'], 8)
        i = 0
        running_loss = 0.
        with tqdm(range(len(dataloader)), desc=f'Epoch : - lr: - Loss :') as progress:
            for x0, labels in dataloader:
                i += 1
                # Move data to device
                x0 = x0.to(self.args.device)
                # Use guidance
                labels = labels.to(self.args.device)
                if np.random.random() < 0.1:
                    labels = None

                # Calculate the loss
                with autocast(device_type=self.args.device, enabled=self.args.fp16_precision):
                    loss = self.diffusion.loss(x0, labels)
                    
                # Zero gradients
                self.optimizer.zero_grad()
                # Backward pass
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

            # Step the scheduler after each epoch
            self.scheduler.step()

    def train(self, dataloader):
        scaler = GradScaler(device=self.args.device, enabled=self.args.fp16_precision)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        start_epoch = self.current_epoch
        self.loss_per_iter = []
        for current_epoch in range(start_epoch, self.args.epochs):
            self.current_epoch = current_epoch
            self.train_epoch(dataloader, scaler)

            if current_epoch % self.args.show_every_n_epochs == 0:
                self.sample(cfg_scale=self.args.cfg_scale)

            if (current_epoch + 1) % self.args.save_every_n_epochs == 0:
                self.save_model()

        # at end of training, save loss history
        loss_json = os.path.join(CHECKPOINT_DIR, "loss_history.json")
        with open(loss_json, 'w') as jf:
            json.dump(self.loss_per_iter, jf)
        print(f"[Training Complete] loss history saved to {loss_json}")

    def sample(self, labels=None, cfg_scale=3., n_steps=None, set_seed=False, show=False, save=True):
        if set_seed:
            torch.manual_seed(42)
        if n_steps is None:
            n_steps = self.args.n_steps
            
        self.eps_model.eval()
        with torch.no_grad():
            z_t = torch.randn(
                [
                    self.args.n_samples,
                    self.args.image_channels,
                    self.args.image_size,
                    self.args.image_size,
                ],
                device=self.args.device
            )
            
            if labels is None:
                labels = torch.randint(0, 9, (self.args.n_samples,), device=self.args.device)
                
            if self.args.nb_save is not None:
                saving_steps = [self.args["n_steps"] - 1]
            
            for curr_t in tqdm(reversed(range(n_steps))):
                t = torch.full((self.args.n_samples,), curr_t, device=self.args.device, dtype=torch.long)
                lambda_t = self.diffusion.get_lambda(t)
                lambda_t_prim = self.diffusion.get_lambda(torch.clamp(t - 1, min=0))
                eps_uncond = self.eps_model(z_t, None)    
                eps_cond = self.eps_model(z_t, labels)    
                eps_guided = (1 + cfg_scale) * eps_cond - cfg_scale * eps_uncond
                alpha_t = self.diffusion.alpha_lambda(lambda_t)
                sigma_t = self.diffusion.sigma_lambda(lambda_t)
                x_t_hat = (z_t - sigma_t * eps_guided) / alpha_t
                z_t = self.diffusion.p_sample(z_t, lambda_t, lambda_t_prim, x_t_hat)

        print(f"Showing/saving samples from epoch {self.current_epoch} with labels: {labels.tolist()}")
        show_save(
            z_t,
            labels,
            show=show,
            save=save,
            file_name=f"DDPM_epoch_{self.current_epoch}.png",
        )
        return z_t

    def save_model(self):
        # save torch save model with epoch in filename
        ckpt = {
            'epoch': self.current_epoch,
            'model_state_dict': self.eps_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        filename = os.path.join(
            CHECKPOINT_DIR,
            f"cfg_epoch_{self.current_epoch:03d}.pt"
        )
        torch.save(ckpt, filename)
        print(f"[Checkpoint] model saved to {filename}")

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=args.device)
        self.eps_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
    
def show_save(img_tensor, labels=None, show=True, save=True, file_name="sample.png"):
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    assert img_tensor.shape[0] >= 9, "Number of images should be at least 9"
    img_tensor = img_tensor[:9]
    for i, ax in enumerate(axs.flat):
        img = img_tensor[i].squeeze().cpu().numpy()
        label = labels[i].item()
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Digit:{label}')
        ax.axis("off")
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(IMAGE_DIR, file_name))
    if show:
        plt.show()
    plt.close(fig)


def experiment3(train: bool = True, checkpoint_epoch: int = None):
    # -- prepare data, model, diffusion and trainer --
    dataloader = torch.utils.data.DataLoader(
        MNISTDataset(),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )

    eps_model = UNet_conditional(c_in=1, c_out=1, num_classes=10)
    diffusion_model = CFGDiffusion(
        eps_model=eps_model,
        n_steps=args.n_steps,
        device=args.device,
    )
    trainer = Trainer(args, eps_model, diffusion_model)

    # -- load-only mode --
    if not train:
        # default to last epoch if none specified
        epoch_to_load = checkpoint_epoch if checkpoint_epoch is not None else args.epochs - 1
        ckpt_path = os.path.join(
            CHECKPOINT_DIR,
            f"cfg_epoch_{epoch_to_load:03d}.pt"
        )
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

        trainer.load_model(ckpt_path)
        print(f"[Checkpoint] Loaded model at epoch {trainer.current_epoch} from {ckpt_path}")
        return trainer

    # -- training mode --
    trainer.train(dataloader)
    return trainer

if __name__ == "__main__":
    experiment3()