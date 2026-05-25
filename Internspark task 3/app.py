import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image

# 1. Initialize FastAPI Application
app = FastAPI(
    title="Airplane vs. Automobile Classifier API",
    description="Production API deploying a PyTorch transfer learning model.",
    version="1.0"
)

# 2. Setup Device (Forced to CPU inside container environments for efficiency)
device = torch.device("cpu")

# 3. Re-create the Model Architecture & Load Saved Weights
model = resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 output classes

try:
    # Load weights onto CPU (map_location='cpu' is critical here)
    model.load_state_dict(torch.load("binary_image_classifier.pth", map_location=device))
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model weights: {e}")

# 4. Define Image Preprocessing Transformation (Matches Task 2 Test Transform)
image_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CIFAR-10 image size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

# 5. Define Class Map
CLASS_NAMES = ["Airplane", "Automobile"]


@app.get("/")
def health_check():
    """Simple endpoint to verify API health."""
    return {
        "status": "healthy",
        "model_loaded": True,
        "classes": CLASS_NAMES
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Receives an image file and returns model prediction details."""
    # Ensure uploaded file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        # Read file contents into memory as PIL Image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess image
        tensor = image_transform(image).unsqueeze(0).to(device)

        # Run model inference
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        # Build clean JSON response
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        score = confidence.item()

        return {
            "prediction": predicted_class,
            "confidence": round(score, 4),
            "probabilities": {
                CLASS_NAMES[0]: round(probabilities[0][0].item(), 4),
                CLASS_NAMES[1]: round(probabilities[0][1].item(), 4)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")
