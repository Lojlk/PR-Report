# predict.py

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from modules import VisionTransformer
import os

def load_model(model_path, device, img_size=224, patch_size=16, num_classes=2,
              embed_dim=768, depth=12, num_heads=12, mlp_dim=3072, dropout=0.1, cls_token=True):
    """
    Loads the trained Vision Transformer model.

    Args:
        model_path (str): Path to the saved model weights.
        device (torch.device): Device to load the model on.
        Other args: Model architecture parameters.

    Returns:
        nn.Module: Loaded model.
    """
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        cls_token=cls_token
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image_path, image_size=224):
    """
    Preprocesses the input image.

    Args:
        image_path (str): Path to the image.
        image_size (int): Size to resize the image.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                             std=[0.229, 0.224, 0.225])   # ImageNet stds
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def predict(image_path, model, device, class_names):
    """
    Predicts the class of the input image.

    Args:
        image_path (str): Path to the image.
        model (nn.Module): Trained model.
        device (torch.device): Device to perform inference on.
        class_names (list): List of class names.

    Returns:
        str: Predicted class name.
        torch.Tensor: Output probabilities.
    """
    image = preprocess_image(image_path)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        _, preds = torch.max(probabilities, 1)

    return class_names[preds.item()], probabilities.cpu().numpy()

def visualize_prediction(image_path, predicted_class, probabilities, class_names):
    """
    Visualizes the image with prediction probabilities.

    Args:
        image_path (str): Path to the image.
        predicted_class (str): Predicted class name.
        probabilities (numpy.ndarray): Probabilities for each class.
        class_names (list): List of class names.
    """
    image = Image.open(image_path).convert('RGB')

    plt.figure(figsize=(6,6))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Predicted: {predicted_class}\nProbabilities: {probabilities}")
    plt.show()

def main():
    # Configuration
    # maybe remove this 
    model_path = 'saved_models/best_vit_model.pth'  # Path to your trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 224
    patch_size = 16
    num_classes = 2
    embed_dim = 768
    depth = 12
    num_heads = 12
    mlp_dim = 3072
    dropout = 0.1
    cls_token = True
    class_names = ['AD', 'Normal']  # Adjust based on your dataset

    # Load the model
    model = load_model(
        model_path=model_path,
        device=device,
        img_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        cls_token=cls_token
    )
    print("Model loaded successfully.")

    # Path to the image you want to predict
    test_image_path = 'path_to_your_test_image.jpg'  # Replace with your image path

    if not os.path.exists(test_image_path):
        print(f"Image not found at {test_image_path}. Please provide a valid path.")
        return

    # Make prediction
    predicted_class, probabilities = predict(test_image_path, model, device, class_names)
    print(f"Predicted Class: {predicted_class}")
    print(f"Probabilities: {probabilities}")

    # Visualize prediction
    visualize_prediction(test_image_path, predicted_class, probabilities, class_names)

if __name__ == '__main__':
    main()
