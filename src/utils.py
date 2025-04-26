import os
import torch

def save_model(logdir, file_name, model, verbose=True) -> None:
    """ Save the model's state dictionary """
    os.makedirs(logdir, exist_ok=True)
    file_path = os.path.join(logdir, f"{file_name}.pth")
    torch.save(model, file_path)        
    if verbose: print(f"Model saved to {logdir}")

def load_model(logdir, model_name, map_location=None, verbose=True) -> nn.Module:
    """ Load the model's state dictionary """
    load_path = os.path.join(logdir, f"{model_name}.pth")
    # If map_location is None, will load onto the devices originally saved from. 
    model = torch.load(load_path, map_location=map_location, weights_only=False)
    if verbose: print(f"Model loaded from {logdir}")
    return model