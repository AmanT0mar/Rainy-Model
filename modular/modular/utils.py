import torch
from pathlib import Path
from torch import nn

def save_model(model: nn.Module, model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    model_name: A filename for the saved model.
    """

    model_path = Path("models")
    model_path.mkdir(parents=True, exist_ok=True)

    model_name = f"{model_name}_model.pth"
    model_save_path = model_path / model_name

    print(f"Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

    print(f"Saved model for city: {model_name}")