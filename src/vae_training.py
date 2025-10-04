from __future__ import print_function
import os
import json
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from types import SimpleNamespace
from vae_objectives import log_likelihood_bernoulli, kl_gaussian_gaussian_analytic
import json
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# -------------------------------------------------------------------
# VAE definition (unchanged)
# -------------------------------------------------------------------
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1   = nn.Linear(784, 400)
        self.fc21  = nn.Linear(400, 20)
        self.fc22  = nn.Linear(400, 20)
        self.fc3   = nn.Linear(20, 400)
        self.fc4   = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z          = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# -------------------------------------------------------------------
# ELBO loss (unchanged)
# -------------------------------------------------------------------
def loss_function(recon_x, x, mu, logvar):
    rec = -log_likelihood_bernoulli(recon_x, x.flatten(1, 3)).sum()
    kl  = kl_gaussian_gaussian_analytic(
        mu_q=mu,
        logvar_q=logvar,
        mu_p=torch.zeros_like(mu),
        logvar_p=torch.zeros_like(logvar),
    ).sum()
    return rec + kl

# -------------------------------------------------------------------
# One epoch of training
# -------------------------------------------------------------------
def train_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for data, _ in loader:
        data = data.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        loss = loss_function(recon, data, mu, logvar)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader.dataset)

# -------------------------------------------------------------------
# One epoch of validation
# -------------------------------------------------------------------
def validate_epoch(model, loader, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            recon, mu, logvar = model(data)
            loss = loss_function(recon, data, mu, logvar)
            total += loss.item()
    return total / len(loader.dataset)

# -------------------------------------------------------------------
# Main experiment
# -------------------------------------------------------------------
def experiment1(train: bool = True):
    # Defaults + new save_interval
    args = SimpleNamespace(
        batch_size    = 128,
        epochs        = 20,
        seed          = 1,
        log_interval  = 10,
        no_cuda       = False,
        results_dir   = 'results/experiment1',
        save_interval = 5,   # save every 5 epochs by default
    )
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    os.makedirs(args.results_dir, exist_ok=True)

    # Data loaders
    loader_kwargs = {'num_workers':1, 'pin_memory':True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    # Model & optimizer
    model     = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # If not training: load the last checkpoint and return
    if not train:
        ckpt_path = os.path.join(args.results_dir, f'checkpoint_epoch_{args.epochs}.pt')
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return model, optimizer

    # --- training branch ---
    history = {'train_loss': [], 'val_loss': []}
    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, device)
        va = validate_epoch(model, val_loader, device)
        history['train_loss'].append(tr)
        history['val_loss'].append(va)

        print(f"Epoch {epoch}/{args.epochs}  train_loss={tr:.4f}  val_loss={va:.4f}")

        # save history
        with open(os.path.join(args.results_dir, 'history.json'), 'w') as fp:
            json.dump(history, fp, indent=4)

        # checkpoint at interval
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            path = os.path.join(args.results_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(ckpt, path)
            print(f"Saved checkpoint: {path}")

    return model, optimizer

if __name__ == "__main__":
    # To train:
    model, optimizer = experiment1(train=True)
    # To load instead of training:
    # model, optimizer = experiment1(train=False)
