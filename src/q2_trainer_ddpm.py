import os
import json
import torch
from matplotlib import pyplot as plt 
from tqdm import tqdm
from torch.amp import GradScaler, autocast
import copy
import torch.nn.functional as F

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

            if (current_epoch + 1) % self.args.save_every_n_epochs == 0:
                self.save_model()

    def sample(self, n_steps=None, set_seed=False):
        if set_seed:
            torch.manual_seed(42)
        if n_steps is None:
            n_steps = self.args.n_steps

        with torch.no_grad():
            x = torch.randn(
                [
                    self.args.n_samples,
                    self.args.image_channels,
                    self.args.image_size,
                    self.args.image_size,
                ],
                device=self.args.device,
            )
            if self.args.nb_save is not None:
                saving_steps = [self.args["n_steps"] - 1]

            for curr_t in tqdm(reversed(range(n_steps)), desc="Sampling"):
                t = torch.full((self.args.n_samples,), curr_t, device=self.args.device, dtype=torch.long)
                x = self.diffusion.p_sample(x, t)
        if self.args.nb_save is not None and curr_t in saving_steps:
            print(f"Showing/saving samples from epoch {self.current_epoch}")
            self.show_save(
                x, show=False, save=True,
                file_name=f"DDPM_epoch_{self.current_epoch}.png"
            )
        return x

    def save_model(self):
        """Save model+optimizer with epoch stamp, plus loss history to JSON."""
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
        # save loss history
        json_path = os.path.join(
            CHECKPOINT_DIR,
            f"loss_history_epoch_{self.current_epoch:03d}.json"
        )
        with open(json_path, 'w') as jf:
            json.dump(self.loss_per_iter, jf)
        print(f"[Checkpoint] model saved to {filename}")
        print(f"[Checkpoint] loss history saved to {json_path}")

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

    def generate_intermediate_samples(self, n_samples=4, img_size=32, steps_to_show=[0,999], n_steps=None, set_seed=False):
        if set_seed:
            torch.manual_seed(42)
        if n_steps is None:
            n_steps = args.n_steps
        x = torch.randn(n_samples, 1, img_size, img_size, device=args.device, requires_grad=False)
        images = [x.detach().cpu().numpy()]
        for step in tqdm(range(1, n_steps-1), desc="Sampling"):
            t = torch.full((n_samples,), step, device=self.args.device, dtype=torch.long)
            x = self.diffusion.p_sample(x, t)
            if step in steps_to_show:
                images.append(x.detach().cpu().numpy())
        return images

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
    return trainer

if __name__ == "__main__":
    experiment2()
