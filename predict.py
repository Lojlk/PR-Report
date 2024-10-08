import torch
from torchvision import transforms
from PIL import Image
import os
from modules import VisionTransformer

def load_model(model_path, device):
    model = VisionTransformer()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(model, image_path, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return "AD" if predicted.item() == 0 else "NC"

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "vit_model.pth"
    model = load_model(model_path, device)

    test_image_path = "/home/groups/comp3710/ADNI/AD_NC/test/AD/1003730_100.jpeg"  # Replace with an actual image path
    # 1003730_100.jpeg 
    result = predict(model, test_image_path, device)
    print(f"Prediction for the test image: {result}")

if __name__ == '__main__':
    main()
