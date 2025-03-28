from flask import Flask, request, render_template
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import os

# --- Flask App Setup ---
app = Flask(__name__)

# --- Configuration ---
LABELS = sorted(os.listdir("PokemonData/"))
NUM_CLASSES = len(LABELS)
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Label Index Mapping ---
label_to_idx = {label: idx for idx, label in enumerate(LABELS)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

# --- Define Model ---
class PretrainedResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# --- Load Model ---
model = PretrainedResNet18(NUM_CLASSES)
model.load_state_dict(torch.load("best_model_ResNet18.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --- Image Transform ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image = Image.open(file).convert("RGB")
            img_tensor = transform(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(img_tensor)
                pred = torch.argmax(output, dim=1).item()
                prediction = idx_to_label[pred]
            image.save("static/uploaded_image.png")

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
