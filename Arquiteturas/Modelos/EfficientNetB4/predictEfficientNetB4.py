import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from PIL import Image

# Model path and class names
modelo_path = "efficientnetb4.pth"
classes = ["Cancer", "Non Cancer"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained EfficientNet-B4 and replace classifier head
weights = EfficientNet_B4_Weights.DEFAULT
net = efficientnet_b4(weights=weights)
net.classifier[1] = nn.Linear(net.classifier[1].in_features, len(classes))
net.load_state_dict(torch.load(modelo_path, map_location=device))
net.to(device)
net.eval()
print("EfficientNet-B4 loaded successfully! Ready for diagnosis.\n")

def diagnosticar_imagem(caminho_imagem):
    # Preprocess image to match training transform
    transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    imagem = Image.open(caminho_imagem).convert("RGB")
    imagem_tensor = transform(imagem).unsqueeze(0).to(device)

    # Run inference and get predicted class with confidence
    with torch.no_grad():
        saida = net(imagem_tensor)
        probabilidades = torch.nn.functional.softmax(saida, dim=1)
        confianca, indice_previsto = torch.max(probabilidades, 1)

    classe_prevista = classes[indice_previsto.item()]
    confianca_pct = confianca.item() * 100
    print(f"Diagnosis: {classe_prevista}")
    print(f"Confidence: {confianca_pct:.2f}%\n")
    return classe_prevista, confianca_pct

# Set image path and run diagnosis
imagem_teste = "ImagensTesteUnico/imagem.jpg"  # set image path here
diagnosticar_imagem(imagem_teste)