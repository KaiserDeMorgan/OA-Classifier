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

# Load once, no transform yet, split first, THEN decide what transform each split gets
base_dataset = torchvision.datasets.ImageFolder(root="C:/Users/kaise/Downloads/cleaned_knee_xrays")

n = len(base_dataset)
n_train = int(0.7 * n)
n_val = int(0.15 * n)
n_test = n - n_train - n_val

train_subset, val_subset, test_subset = torch.utils.data.random_split(
    base_dataset, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(50)
)

# Wraps a Subset with a chosen transform (ImageFolder without transform gives raw PIL images)
class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label


def load_data(data_augmentation=False):
    train_transform = transform_augmented if data_augmentation else transform_original
    train_dataset = TransformSubset(train_subset, train_transform)
    val_dataset = TransformSubset(val_subset, transform_original)   # never augment val/test
    test_dataset = TransformSubset(test_subset, transform_original)
    return train_dataset, val_dataset, test_dataset


train_dataset, val_dataset, test_dataset = load_data(data_augmentation=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1)
valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, num_workers=1)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, num_workers=1)

print(f"Classes: {base_dataset.classes}")
print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}  Test: {len(test_dataset)}")

#Implementing oversampling as there's more 0 and 1 images compared to 3 and 4
from collections import Counter
from torch.utils.data import WeightedRandomSampler

train_labels = [base_dataset.targets[i] for i in train_subset.indices]

class_counts = Counter(train_labels)
print("Train class distribution before weighting:", class_counts)

class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
sample_weights = [class_weights[label] for label in train_labels]

sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, sampler=sampler, num_workers=1)

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
