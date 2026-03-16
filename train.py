import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os

from dataset import PianoToneDataset, PreprocessedPianoDataset
from model import PianoNet

PREPROCESSED_V2_MANIFEST = os.path.join(os.path.dirname(__file__), "preprocessed_v2", "manifest.json")
PREPROCESSED_MANIFEST    = os.path.join(os.path.dirname(__file__), "preprocessed",    "manifest.json")

def train_model(data_dir: str, epochs: int = 30, batch_size: int = 32, lr: float = 0.001):
    print("Initializing PyTorch...", flush=True)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)

    # Load dataset — prefer v2 (larger windows, more chunks), then v1, then raw audio
    if os.path.exists(PREPROCESSED_V2_MANIFEST):
        print("Using pre-computed spectrograms v2 (fast path).", flush=True)
        full_dataset = PreprocessedPianoDataset(PREPROCESSED_V2_MANIFEST, augment=True)
        clean_dataset = PreprocessedPianoDataset(PREPROCESSED_V2_MANIFEST, augment=False)
        num_workers = 4
    elif os.path.exists(PREPROCESSED_MANIFEST):
        print("Using pre-computed spectrograms v1 (fast path).", flush=True)
        full_dataset = PreprocessedPianoDataset(PREPROCESSED_MANIFEST, augment=True)
        clean_dataset = PreprocessedPianoDataset(PREPROCESSED_MANIFEST, augment=False)
        num_workers = 4
    else:
        print(f"Loading raw audio from: {data_dir}", flush=True)
        verified_files = os.path.join(os.path.dirname(__file__), "verified_files.json")
        full_dataset = PianoToneDataset(data_dir, augment=True, verified_files=verified_files)
        clean_dataset = PianoToneDataset(data_dir, augment=False, verified_files=verified_files)
        num_workers = 0  # librosa is not safe with num_workers > 0 on Windows

    print(f"Dataset length: {len(full_dataset)}", flush=True)

    if len(full_dataset) == 0:
        print("No valid data found. Exiting.")
        return

    # Train/Val/Test split (80/10/10)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(total_size, generator=generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(clean_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(clean_dataset, test_indices)

    print(f"Training set: {train_size} | Validation: {val_size} | Test: {test_size} | num_workers: {num_workers}", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Initialize model, loss, optimizer, and scheduler
    model = PianoNet(num_classes=88).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Tracking metrics
    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_acc = 0.0
    epochs_no_improve = 0
    early_stop_patience = 8

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for specs, labels in train_loader:
            specs, labels = specs.to(device), labels.to(device)
            outputs = model(specs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for specs, labels in val_loader:
                specs, labels = specs.to(device), labels.to(device)
                outputs = model(specs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100 * correct / total

        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)

        scheduler.step(epoch_val_loss)

        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), "piano_net_best.pth")
            epochs_no_improve = 0
            best_marker = " *** best ***"
        else:
            epochs_no_improve += 1
            best_marker = ""

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%{best_marker}", flush=True)

        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs (no improvement for {early_stop_patience} epochs).")
            break

    print(f"\nTraining finished. Best val accuracy: {best_val_acc:.2f}%")
    
    # Final Test Set Evaluation
    print("\nRunning evaluation on Test Set...")
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for specs, labels in test_loader:
            specs, labels = specs.to(device), labels.to(device)
            outputs = model(specs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * specs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    test_accuracy = 100 * correct / max(total, 1)
    test_loss_avg = test_loss / max(total, 1)
    print(f"Test Loss: {test_loss_avg:.4f} | Test Accuracy: {test_accuracy:.2f}%")
    
    # Load best checkpoint for final evaluation and save as main model
    model.load_state_dict(torch.load("piano_net_best.pth"))
    torch.save(model.state_dict(), "piano_net.pth")
    print("Best model saved to piano_net.pth")
    
    # Plotting
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training plots saved to training_history.png")

if __name__ == '__main__':
    # Local path from user instructions
    data_directory = r"C:\Users\ToreGrüner\SkyDrive\Dokumenter\Musikk\Piano"
    
    # We use a small number of epochs for initial testing
    train_model(data_directory, epochs=30, batch_size=16, lr=0.001)
