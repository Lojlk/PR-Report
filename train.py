import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from modules import VisionTransformer
from dataset import get_data_loaders

def train(model, train_loader, val_loader, criterion, optimizer, epochs, device):
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

        train_loss.append(running_loss / len(train_loader))
        
        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

        val_loss.append(running_val_loss / len(val_loader))
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}")

    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig('loss_curve.png')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = "/home/groups/comp3710/ADNI/AD_NC"
    train_loader, val_loader, test_loader = get_data_loaders(data_dir)

    model = VisionTransformer()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    train(model, train_loader, val_loader, criterion, optimizer, epochs=1, device=device)
    torch.save(model.state_dict(), "vit_model.pth")

if __name__ == '__main__':
    main()
