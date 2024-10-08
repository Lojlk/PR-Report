# train.py

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_data_loaders
from modules import VisionTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

def train_epoch(model, dataloader, criterion, optimizer, device, clip_grad=False, max_norm=1.0):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        if clip_grad:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, save_dir):
    epochs = range(1, len(train_losses) + 1)

    # Plot Loss
    plt.figure(figsize=(10,5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Over Epochs')
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(10,5))
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Over Epochs')
    plt.savefig(os.path.join(save_dir, 'accuracy_curve.png'))
    plt.close()

def main():
    # Configuration
    data_dir = 'ADNI_Dataset'  # Replace with your dataset path
    image_size = 224
    batch_size = 16
    num_workers = 4
    num_classes = 2
    embed_dim = 768
    num_heads = 12
    mlp_dim = 3072
    depth = 12
    dropout = 0.1
    patch_size = 16
    cls_token = True
    num_epochs = 30
    patience = 5  # For Early Stopping
    learning_rate = 1e-4
    weight_decay = 1e-5
    save_dir = 'saved_models'

    # Create directory to save models and plots
    os.makedirs(save_dir, exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get data loaders, class names, and class weights
    data_loaders, class_names, class_weights = get_data_loaders(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # Initialize the Vision Transformer model
    model = VisionTransformer(
        img_size=image_size,
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

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

    # Training loop variables
    best_val_acc = 0.0
    counter = 0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Training
        train_loss, train_acc = train_epoch(
            model, data_loaders['train'], criterion, optimizer, device, clip_grad=True, max_norm=1.0
        )
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation
        val_loss, val_acc = eval_epoch(model, data_loaders['val'], criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%\n")

        # Scheduler step
        scheduler.step(val_acc)

        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_vit_model.pth'))
            print("Best model updated and saved.\n")
            counter = 0
        else:
            counter += 1
            print(f"No improvement for {counter} epochs.\n")
            if counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Plot training and validation metrics
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, save_dir)
    print(f"Training complete. Metrics plots saved to {save_dir}.")

    # Evaluate on test set
    test_loss, test_acc = eval_epoch(model, data_loaders['test'], criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

    # Detailed classification report and confusion matrix
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    print("Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    # Save the final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_vit_model.pth'))
    print(f"Final model saved to {save_dir}.")

if __name__ == '__main__':
    main()
