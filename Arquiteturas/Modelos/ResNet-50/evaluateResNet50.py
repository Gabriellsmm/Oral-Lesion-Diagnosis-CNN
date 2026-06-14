import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Model path and class names
modelo_path = "ResNet-50.pth"
test_dir = "../../../DatasetKagle/teste"
classes = ["Cancer", "Non Cancer"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained ResNet-50 and replace the classifier head
weights = ResNet50_Weights.DEFAULT
net = resnet50(weights=weights)
net.fc = nn.Linear(net.fc.in_features, len(classes))
net.load_state_dict(torch.load(modelo_path, map_location=device))
net.to(device)
net.eval()
print("ResNet-50 model loaded for testing!")

# Same transform used during training
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=test_transform)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

accuracy = np.mean(np.array(y_true) == np.array(y_pred)) * 100
print(f"\nTotal accuracy: {accuracy:.2f}%")
print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=classes, digits=4))

# Plot and save confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap='Greens')
plt.title(f"Confusion Matrix – ResNet-50 – Accuracy: {accuracy:.2f}%")
plt.tight_layout()
plt.savefig("matriz_resnet50.png", dpi=300)
plt.show()