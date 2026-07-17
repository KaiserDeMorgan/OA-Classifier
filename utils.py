import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import seaborn as sns
from collections import Counter
from torchmetrics.classification import F1Score, Precision, Recall, ConfusionMatrix

np.random.seed(50)
torch.manual_seed(50)

transform_original = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_augmented = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class TransformSubset(torch.utils.data.Dataset):
    """
    Wraps a Subset (from random_split) with a chosen transform.
    Needed because ImageFolder without a transform gives raw PIL images -
    this applies the actual transform pipeline (resize/augment/normalize)
    at __getitem__ time, so train/val/test can each get a different
    transform even though they're all slices of the same base_dataset.
    """
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label

def get_dataloaders(data_root, batch_size=64, data_augmentation=True, seed=50, use_oversampling=True, num_workers=1):
    base_dataset = torchvision.datasets.ImageFolder(root=data_root)
    n = len(base_dataset)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val

    train_subset, val_subset, test_subset = torch.utils.data.random_split(
        base_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )

    train_transform = transform_augmented if data_augmentation else transform_original
    train_dataset = TransformSubset(train_subset, train_transform)
    val_dataset = TransformSubset(val_subset, transform_original)
    test_dataset = TransformSubset(test_subset, transform_original)

    if use_oversampling:
        train_labels = [base_dataset.targets[i] for i in train_subset.indices]
        class_counts = Counter(train_labels)
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    valid_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    print(f"Classes: {base_dataset.classes}")
    print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}  Test: {len(test_dataset)}")

    return train_loader, valid_loader, test_loader

def getF1Score(pred, target):
    f1 = F1Score(task="multiclass", num_classes=5)
    score = f1(pred, target)
    return score


def getPrecision(pred, target):
    precision = Precision(task="multiclass", num_classes=5)
    score = precision(pred, target)
    return score


def getRecall(pred, target):
    recall = Recall(task="multiclass", num_classes=5)
    score = recall(pred, target)
    return score


def getConfusionMatrix(pred, target):
    cm = ConfusionMatrix(task="multiclass", num_classes=5)
    return cm(pred, target)


def get_all_preds_labels(model, loader, device):
    """
    One pass over a loader, returns concatenated logits and true labels as
    tensors (not lists) - torchmetrics functions need tensors, not Python lists.
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu())
            all_labels.append(labels)
    return torch.cat(all_preds), torch.cat(all_labels)


def train(model, train_loader, valid_loader, num_epochs=50, learning_rate=1e-4):
    """Training loop."""
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()  # BUG FIX: was missing () - referenced the class, not an instance
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "trainLoss": [], "validLoss": [],
        "trainF1Score": [], "validF1Score": [],
        "trainPrecision": [], "validPrecision": [],
        "trainRecall": [], "validRecall": [],
        "validConfusionMatrix": [],
    }

    for epoch in range(num_epochs):
        model.train()
        trainLoss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            trainLoss += loss.item() * inputs.size(0)
        trainLoss /= len(train_loader.dataset)

        model.eval()
        validLoss = 0.0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                validLoss += loss.item() * inputs.size(0)
        validLoss /= len(valid_loader.dataset)

        # single pass per split to get logits+labels, feeds all metrics below
        train_preds, train_labels_true = get_all_preds_labels(model, train_loader, device)
        valid_preds, valid_labels_true = get_all_preds_labels(model, valid_loader, device)

        trainF1Score = getF1Score(train_preds, train_labels_true)
        validF1Score = getF1Score(valid_preds, valid_labels_true)

        trainPrecision = getPrecision(train_preds, train_labels_true)
        validPrecision = getPrecision(valid_preds, valid_labels_true)

        trainRecall = getRecall(train_preds, train_labels_true)
        validRecall = getRecall(valid_preds, valid_labels_true)

        validCM = getConfusionMatrix(valid_preds, valid_labels_true)

        history["trainLoss"].append(trainLoss)
        history["validLoss"].append(validLoss)
        history["trainF1Score"].append(trainF1Score.item())
        history["validF1Score"].append(validF1Score.item())
        history["trainPrecision"].append(trainPrecision.item())
        history["validPrecision"].append(validPrecision.item())
        history["trainRecall"].append(trainRecall.item())
        history["validRecall"].append(validRecall.item())
        history["validConfusionMatrix"].append(validCM)

        print(f"Epoch [{epoch+1}/{num_epochs}]  "
              f"Train Loss: {trainLoss:.6f}  "
              f"Valid Loss: {validLoss:.6f}  "
              f"Train F1: {trainF1Score:.4f}  "
              f"Valid F1: {validF1Score:.4f}  "
              f"Train Precision: {trainPrecision:.4f}  "
              f"Valid Precision: {validPrecision:.4f}  "
              f"Train Recall: {trainRecall:.4f}  "
              f"Valid Recall: {validRecall:.4f}"
        )

    plt.figure(figsize=(6, 5))
    sns.heatmap(history["validConfusionMatrix"][-1].numpy(), annot=True, fmt="d",
                xticklabels=[0, 1, 2, 3, 4], yticklabels=[0, 1, 2, 3, 4], cmap="Blues")
    plt.xlabel("Predicted Grade")
    plt.ylabel("True Grade")
    plt.title(f"Validation Confusion Matrix - Epoch {num_epochs}")
    plt.show()

    return model, history
