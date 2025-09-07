import torch
import numpy as np
from sklearn.metrics import accuracy_score

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def evaluate_overall(model, test_dloader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_dloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    return accuracy_score(all_targets.numpy(), all_preds.numpy())

def evaluate_classwise(model, test_dloader, device, class_names):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_dloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    classwise_acc = {}
    for i, class_name in enumerate(class_names):
        idx = (all_targets == i)
        class_acc = accuracy_score(all_targets[idx].numpy(), all_preds[idx].numpy())
        classwise_acc[class_name] = class_acc

    return classwise_acc

def confusion_matrix_plot(model, test_dloader, device, class_names):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_dloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    cm = confusion_matrix(all_targets.numpy(), all_preds.numpy())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', xticks_rotation='vertical')
    return disp


def plot_roc_curves(model, test_dloader, device, class_names):

    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_dloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu())
            all_targets.append(targets.cpu())
    
    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()

    n_classes = len(class_names)
    all_targets_binarized = label_binarize(all_targets, classes=range(n_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(all_targets_binarized[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fig, axes = plt.subplots(5, 2, figsize=(16, 20), dpi=150)
    axes = axes.flatten()
    for i in range(n_classes):
        axes[i].plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:0.2f})')
        axes[i].plot([0, 1], [0, 1], 'k--')
        axes[i].set_xlim([0.0, 1.0])
        axes[i].set_ylim([0.0, 1.05])
        axes[i].set_xlabel('False Positive Rate', labelpad=15)
        axes[i].set_ylabel('True Positive Rate', labelpad=15)
        axes[i].legend(loc='lower right')
        # Remove title for more space
    # Hide unused subplots if n_classes < 10
    for j in range(n_classes, len(axes)):
        axes[j].axis('off')
    fig.tight_layout(pad=3.0)
    return fig
