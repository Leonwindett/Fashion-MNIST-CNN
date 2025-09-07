import json
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
import torch.nn as nn
import wandb
import optuna
import copy
import inspect


def train_model(model, optimiser, loss_fcn, train_dloader, val_dloader, epoch, device, print_training = True):

    model.train()
    model.to(device)
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

        
    return tot_loss/len(train_dloader.dataset), valid_loss/len(val_dloader.dataset), val_acc


def tune_model(model_class, train_dloader, val_dloader, model_name, device=torch.device("mps"),
               n_trials=20, project_name='my_project', hyperparam_space=None, epochs=5):

    best_model_wts = None
    best_score = -float('inf')

    # Get model constructor argument names to filter params
    model_init_args = inspect.signature(model_class.__init__).parameters

    def sample_trial(trial, space):
        params = {}
        for key, spec in space.items():
            if spec["type"] == "float":
                params[key] = trial.suggest_float(key, spec["low"], spec["high"], log=spec.get("log", False))
            elif spec["type"] == "int":
                params[key] = trial.suggest_int(key, spec["low"], spec["high"])
            elif spec["type"] == "categorical":
                params[key] = trial.suggest_categorical(key, spec["choices"])
        return params

    def objective(trial):
        nonlocal best_model_wts, best_score

        params = sample_trial(trial, hyperparam_space)
        model_params = {k: v for k, v in params.items() if k in model_init_args}
        
        model = model_class(**model_params).to(device)
        lr = params.get("lr", 1e-3)
        weight_decay = params.get("weight_decay", 0.0)
        optimizer_name = params.get("optimizer", "AdamW")
        momentum = params.get("momentum", 0.9)

        if optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:  # SGD
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

        loss_fcn = nn.CrossEntropyLoss()

        # Start wandb run
        wandb.init(project=project_name, config=params, reinit=True, name=f"{model_name}_trial_{trial.number}", dir=os.path.join("..", "wandb_logs"))

        # Training loop
        for epoch in range(epochs):
            train_loss, val_loss, val_acc = train_model(model, optimizer, loss_fcn, train_dloader, val_dloader, epoch, device, print_training=False)

            # Log metrics
            wandb.log({"val_acc": val_acc, "train_loss": train_loss, "val_loss": val_loss})

        # Save best model
        if val_acc > best_score:
            best_score = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

            # Save into parent directory, under saved_models/<model_name>
            run_folder = os.path.join("..", "saved_models", model_name)
            os.makedirs(run_folder, exist_ok=True)

            # Save best model (overwrites previous)
            best_model_path = os.path.join(run_folder, "best_model.pth")
            torch.save(best_model_wts, best_model_path)

            # Save best trial parameters (overwrites previous)
            params_path = os.path.join(run_folder, "study_params.json")
            with open(params_path, "w") as f:
                json.dump(params, f, indent=2)


        wandb.finish()
        return val_acc

    # Run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"Best trial params: {study.best_trial.params}")
    print(f"Best validation accuracy: {best_score:.4f}")
    
    return