# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from modules import VisionTransformer
from dataset import get_data_loaders
import time  # For tracking training time

def train(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    """
    Trains the Vision Transformer model and validates it.

    Args:
        model (nn.Module): The Vision Transformer model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        epochs (int): Number of training epochs.
        device (torch.device): Device to train on (CPU or GPU).

    Returns:
        tuple: Lists containing training and validation losses.
    """
    model.to(device)
    train_loss, val_loss = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_loss.append(avg_train_loss)
        
        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_loss.append(avg_val_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Plotting training and validation loss
    plt.figure(figsize=(10,5))
    plt.plot(range(1, epochs+1), train_loss, label='Train Loss')
    plt.plot(range(1, epochs+1), val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('saved_models/simple_loss_curve.png')  # Saving the plot
    plt.show()

    return train_loss, val_loss

def evaluate(model, test_loader, device):
    """
    Evaluates the model on the test set and calculates accuracy.

    Args:
        model (nn.Module): Trained Vision Transformer model.
        test_loader (DataLoader): DataLoader for test data.
        device (torch.device): Device to perform evaluation on.

    Returns:
        float: Test accuracy.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    return test_accuracy

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_dir = "/home/groups/comp3710/ADNI/AD_NC"  # Path to your dataset
    train_loader, val_loader, test_loader = get_data_loaders(data_dir)

    model = VisionTransformer()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    epochs = 100  # Set your desired number of epochs

    # Start time
    start_time = time.time()

    # Train the model
    train_losses, val_losses = train(model, train_loader, val_loader, criterion, optimizer, epochs, device)

    # End time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total training time: {elapsed_time//60:.0f}m {elapsed_time%60:.0f}s")

    # Save the trained model
    os.makedirs('saved_models', exist_ok=True)  # Ensure the directory exists
    torch.save(model.state_dict(), "saved_models/simple_vit_model.pth")
    print("Model saved to 'saved_models/simple_vit_model.pth'")

    # Evaluate on the test set
    test_accuracy = evaluate(model, test_loader, device)

    # Optionally, you can save the test accuracy to a file
    with open('saved_models/simple_test_accuracy.txt', 'w') as f:
        f.write(f"Test Accuracy: {test_accuracy*100:.2f}%\n")
    print("Test accuracy saved to 'saved_models/test_accuracy.txt'")

if __name__ == '__main__':
    main()
