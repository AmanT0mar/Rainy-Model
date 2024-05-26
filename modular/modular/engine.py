import torch
from torch import nn
from torch.utils.data import DataLoader


def train_step(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device = 'cpu'):
    """Trains a Pytorch model for a single epoch.

    Turns a target PyTorch model to training mode and then runs through all of the required training steps.

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    """

    model.train()

    train_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    train_loss = train_loss / len(dataloader)

    return train_loss


def test_step(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
               device: torch.device = 'cpu'):
    model.eval()
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    """

    test_loss = 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            test_pred = model(X)

            loss = loss_fn(test_pred, y)
            test_loss += loss.item()

    test_loss = test_loss / len(dataloader)

    return test_loss


def train_model(model: nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader, loss_fn: nn.Module,
                optimizer: torch.optim.Optimizer, epochs: int = 1, device: torch.device = 'cpu') -> dict[str, list]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    """

    model.to(device)

    results = {"train_loss" : [],
               "test_loss" : []}

    for epoch in range(epochs):

        train_loss = train_step(model=model,
                                dataloader=train_dataloader,
                                loss_fn=loss_fn,
                                optimizer=optimizer,
                                device=device)

        test_loss = test_step(model=model,
                              dataloader=test_dataloader,
                              loss_fn=loss_fn,
                              optimizer=optimizer,
                              device=device)

        if epoch % 20 == 0:
            print(f"Epoch: {epoch} | Train loss: {train_loss:.6f} | Test loss: {test_loss:.6f}")

        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)

    return results