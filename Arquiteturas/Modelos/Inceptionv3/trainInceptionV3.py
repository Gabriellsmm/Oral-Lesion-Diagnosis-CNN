import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import inception_v3, Inception_V3_Weights
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, ConfusionMatrixDisplay
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

# Training settings
SEEDS      = [42, 123, 456]
NUM_EPOCHS = 10
BATCH_SIZE = 8
LR         = 0.001

train1_dir = "../../../DatasetKagle/train1"
test_dir   = "../../../DatasetKagle/teste"
classes    = ["CANCER", "NON CANCER"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transforms for training (with augmentation) and testing
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def set_seed(seed):
    # Fix all random sources for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model():
    # Load pretrained Inception-V3 and replace the classifier head
    weights = Inception_V3_Weights.DEFAULT
    net = inception_v3(weights=weights)
    net.aux_logits = False
    net.fc = nn.Linear(net.fc.in_features, len(classes))
    return net.to(device)

# Store results from all runs
all_results = []

for run_idx, seed in enumerate(SEEDS):
    print(f"\n{'='*60}")
    print(f"  RUN {run_idx + 1}/3  |  seed = {seed}")
    print(f"{'='*60}")

    set_seed(seed)

    # DataLoaders recreated at each run
    train_dataset = ConcatDataset([
        torchvision.datasets.ImageFolder(root=train1_dir, transform=train_transform)
    ])
    test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=test_transform)

    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    testloader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Model, criterion and optimizer reinitialized at each run
    net       = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        net.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}] Batch [{i+1}] Loss: {running_loss/10:.3f}")
                running_loss = 0.0

    print(f"  Training Run {run_idx + 1} finished.")

    # Evaluation and ROC curve computation
    net.eval()
    y_true, y_pred = [], []
    all_labels, all_probs = [], []

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs  = net(images)
            probs    = F.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true     = np.array(y_true)
    y_pred     = np.array(y_pred)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    # Compute metrics for the CANCER class
    report = classification_report(y_true, y_pred, target_names=classes,
                                   digits=4, output_dict=True)
    acc  = np.mean(y_true == y_pred) * 100
    prec = report["CANCER"]["precision"] * 100
    sens = report["CANCER"]["recall"]    * 100
    f1   = report["CANCER"]["f1-score"]  * 100

    print(f"\n  Run {run_idx + 1} Results:")
    print(f"    Accuracy:    {acc:.2f}%")
    print(f"    Precision:   {prec:.2f}%")
    print(f"    Sensitivity: {sens:.2f}%")
    print(f"    F1-Score:    {f1:.2f}%")

    all_results.append({"seed": seed, "acc": acc, "prec": prec, "sens": sens, "f1": f1})

    # Compute and save ROC curve data
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc     = auc(fpr, tpr)
    print(f"    AUC: {roc_auc:.4f}")

    np.save(f"fpr_inceptionv3_run{run_idx + 1}.npy", fpr)
    np.save(f"tpr_inceptionv3_run{run_idx + 1}.npy", tpr)

    # Save model weights
    torch.save(net.state_dict(), f"inceptionv3_run{run_idx + 1}_seed{seed}.pth")
    print(f"  Model saved: inceptionv3_run{run_idx + 1}_seed{seed}.pth")

# Final summary across all runs
print(f"\n{'='*60}")
print("  FINAL SUMMARY — Mean ± Std Dev (3 runs)")
print(f"{'='*60}")

metrics      = ["acc",      "prec",      "sens",          "f1"]
labels_print = ["Accuracy", "Precision", "Sensitivity",   "F1-Score"]

summary = {}
for m, lbl in zip(metrics, labels_print):
    values       = [r[m] for r in all_results]
    mean, std    = np.mean(values), np.std(values)
    summary[m]   = (mean, std)
    print(f"  {lbl:<14}: {mean:.2f} ± {std:.2f}%")