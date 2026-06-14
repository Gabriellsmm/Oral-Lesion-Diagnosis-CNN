import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Evaluation settings
SEEDS      = [42, 123, 456]
BATCH_SIZE = 8

test_dir = "/content/drive/MyDrive/Colab Notebooks/DatasetKagle/teste"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same transform used during training
test_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=test_transform)
testloader   = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Detect classes from folder structure
classes = test_dataset.classes
print(f"Detected classes: {classes}")

# Store results from all runs
all_results = []

for run_idx, seed in enumerate(SEEDS):
    # Load saved model weights
    modelo_path = f"efficientnetb4_run{run_idx + 1}_seed{seed}.pth"
    print(f"\nLoading model: {modelo_path}")

    weights = EfficientNet_B4_Weights.DEFAULT
    net = efficientnet_b4(weights=weights)
    net.classifier[1] = nn.Linear(net.classifier[1].in_features, len(classes))
    net.load_state_dict(torch.load(modelo_path, map_location=device))
    net.to(device)
    net.eval()
    print(f"  Model Run {run_idx + 1} loaded!")

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute metrics for the CANCER class
    report = classification_report(y_true, y_pred, target_names=classes,
                                   digits=4, output_dict=True)
    acc  = np.mean(y_true == y_pred) * 100
    prec = report["CANCER"]["precision"] * 100
    sens = report["CANCER"]["recall"]    * 100
    f1   = report["CANCER"]["f1-score"]  * 100

    print(f"  Accuracy:    {acc:.2f}%")
    print(f"  Precision:   {prec:.2f}%")
    print(f"  Sensitivity: {sens:.2f}%")
    print(f"  F1-Score:    {f1:.2f}%")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))

    all_results.append({"seed": seed, "acc": acc, "prec": prec, "sens": sens, "f1": f1})

    # Plot and save confusion matrix
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="Greens")
    plt.title(f"Confusion Matrix – EfficientNet-B4 – Run {run_idx + 1} (seed={seed}) – Acc: {acc:.2f}%")
    plt.tight_layout()
    fname = f"matriz_efficientnetb4_run{run_idx + 1}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"  Confusion matrix saved: {fname}")

# Final summary across all runs
print(f"\n{'='*60}")
print("  FINAL SUMMARY — Mean ± Std Dev (3 runs)")
print(f"{'='*60}")

metrics      = ["acc",      "prec",      "sens",        "f1"]
labels_print = ["Accuracy", "Precision", "Sensitivity", "F1-Score"]

summary = {}
for m, lbl in zip(metrics, labels_print):
    values     = [r[m] for r in all_results]
    mean, std  = np.mean(values), np.std(values)
    summary[m] = (mean, std)
    print(f"  {lbl:<14}: {mean:.2f} ± {std:.2f}%")