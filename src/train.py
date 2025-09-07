import torch
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def train_model(model, optimiser, loss_fcn, train_dloader, val_dloader, epoch, device, print_training = True):

    model.train()
    tot_loss = 0

    all_train_pred = []
    all_train_true = []
    for X_train, y_train in tqdm(train_dloader ,desc=f"Epoch {epoch+1}", leave=False):
        # Ensure the input retains its 4D shape for Conv2d
        X_train, y_train = X_train.to(device), y_train.to(device)
        y_pred = model(X_train)
        optimiser.zero_grad()
        loss = loss_fcn(y_pred, y_train)
        tot_loss += loss.item()
        loss.backward()
        optimiser.step()

        y_pred = y_pred.cpu()
        y_train = y_train.cpu()

        all_train_pred = np.append(all_train_pred, y_pred.detach().numpy(), axis=0) if len(all_train_pred) > 0 else y_pred.detach().numpy()
        all_train_true = np.append(all_train_true, y_train.detach().numpy(), axis=0) if len(all_train_true) > 0 else y_train.detach().numpy()

        # Calculate accuracy for batch

        all_train_pred = np.array(all_train_pred)
        train_pred_1 = torch.argmax(torch.tensor(all_train_pred), dim=1)
        train_acc = accuracy_score(all_train_true, train_pred_1)

    valid_loss = 0
    all_valid_pred = []
    all_valid_true = []


    for X_valid, y_valid in val_dloader:
        # Ensure the input retains its 4D shape for Conv2d
        X_valid, y_valid = X_valid.to(device), y_valid.to(device)
        y_valid_pred = model(X_valid)
        vloss = loss_fcn(y_valid_pred, y_valid)
        valid_loss += vloss.item()
        y_valid_pred = y_valid_pred.cpu()
        y_valid = y_valid.cpu()
        all_valid_pred = np.append(all_valid_pred, y_valid_pred.detach().numpy(), axis=0) if len(all_valid_pred) > 0 else y_valid_pred.detach().numpy()
        all_valid_true = np.append(all_valid_true, y_valid.detach().numpy(), axis=0) if len(all_valid_true) > 0 else y_valid.detach().numpy()

    all_valid_pred = np.array(all_valid_pred)
    y_valid_pred = torch.argmax(torch.tensor(all_valid_pred), dim=1)
    val_acc = accuracy_score(all_valid_true, y_valid_pred)

    if print_training is True:
        print(f"[Epoch {epoch+1:2d}] Training accuracy: {train_acc*100.0:05.2f}%, Validation accuracy: {val_acc*100.0:05.2f}%\n")

        return tot_loss/len(X_train), valid_loss/len(X_valid), val_acc

