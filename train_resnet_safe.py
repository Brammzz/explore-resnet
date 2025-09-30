import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import time
import json
from tqdm import tqdm
import os
from PIL import Image, ImageFile

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class SafeImageFolder(datasets.ImageFolder):
    """Custom ImageFolder that handles corrupt images gracefully."""
    
    def __getitem__(self, index):
        """Override to handle corrupt images."""
        for attempt in range(len(self.samples)):
            try:
                path, target = self.samples[index]
                sample = self.loader(path)
                if self.transform is not None:
                    sample = self.transform(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                return sample, target
            except (OSError, IOError, Image.UnidentifiedImageError) as e:
                print(f"Warning: Skipping corrupt image {path}: {e}")
                # Try next image
                index = (index + 1) % len(self.samples)
                continue
        
        # If all images fail, return a dummy image
        dummy_image = torch.zeros(3, 224, 224)
        return dummy_image, 0

class Trainer:
    """
    Complete training pipeline for ResNet-34 model.
    """
    
    def __init__(self, model, train_loader, val_loader, device, model_name="model"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_name = model_name
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epoch_time': []
        }
        
    def train_epoch(self, criterion, optimizer):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Training {self.model_name}')
        for inputs, labels in pbar:
            try:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
                
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
        
        epoch_loss = running_loss / total if total > 0 else 0
        epoch_acc = 100. * correct / total if total > 0 else 0
        
        return epoch_loss, epoch_acc
    
    def validate(self, criterion):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Validating {self.model_name}')
            for inputs, labels in pbar:
                try:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100.*correct/total:.2f}%'
                    })
                    
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        epoch_loss = running_loss / total if total > 0 else 0
        epoch_acc = 100. * correct / total if total > 0 else 0
        
        return epoch_loss, epoch_acc
    
    def train(self, num_epochs, learning_rate=0.001, optimizer_name='Adam'):
        """Complete training loop."""
        # Define loss function
        criterion = nn.CrossEntropyLoss()
        
        # Define optimizer
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, 
                                momentum=0.9, weight_decay=1e-4)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        
        print(f"\n{'='*70}")
        print(f"Training {self.model_name}")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Optimizer: {optimizer_name}")
        print(f"Initial Learning Rate: {learning_rate}")
        print(f"Number of Epochs: {num_epochs}")
        print(f"Batch Size: {self.train_loader.batch_size}")
        print(f"{'='*70}\n")
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            print("-" * 70)
            
            # Train
            train_loss, train_acc = self.train_epoch(criterion, optimizer)
            
            # Validate
            val_loss, val_acc = self.validate(criterion)
            
            # Update scheduler
            scheduler.step()
            
            # Calculate epoch time
            epoch_time = time.time() - start_time
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['epoch_time'].append(epoch_time)
            
            # Print epoch summary
            print(f"\nEpoch Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  Time: {epoch_time:.2f}s | LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(f'best_{self.model_name}.pth', epoch, best_val_acc)
                print(f"  ✓ New best validation accuracy: {best_val_acc:.2f}%")
        
        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"{'='*70}\n")
        
        return self.history
    
    def save_checkpoint(self, filename, epoch, best_acc):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'best_acc': best_acc,
            'history': self.history
        }
        torch.save(checkpoint, filename)
    
    def plot_history(self, save_path=None):
        """Plot training history."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Plot loss
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title(f'{self.model_name} - Loss Curve', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        axes[1].plot(epochs, self.history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        axes[1].plot(epochs, self.history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title(f'{self.model_name} - Accuracy Curve', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        
    def save_history(self, filename):
        """Save training history to JSON."""
        with open(filename, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"History saved to {filename}")

def prepare_data_safe(data_dir, batch_size=32, img_size=224):
    """Prepare data loaders with safe image handling."""
    
    # Data augmentation and normalization
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets with safe image handling
    train_dataset = SafeImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=train_transform
    )
    
    val_dataset = SafeImageFolder(
        root=os.path.join(data_dir, 'val'),
        transform=val_transform
    )
    
    # Create data loaders with num_workers=0 for Windows compatibility
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Changed to 0 for Windows stability
        pin_memory=False  # Changed to False for CPU training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Changed to 0 for Windows stability
        pin_memory=False  # Changed to False for CPU training
    )
    
    print(f"Dataset loaded successfully!")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Classes: {train_dataset.classes}")
    print(f"  Number of classes: {len(train_dataset.classes)}")
    
    return train_loader, val_loader, train_dataset.classes

def main():
    """Train only ResNet-34 model."""
    
    # Configuration
    DATA_DIR = './data_organized'
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    IMG_SIZE = 224
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    train_loader, val_loader, classes = prepare_data_safe(
        DATA_DIR, 
        batch_size=BATCH_SIZE, 
        img_size=IMG_SIZE
    )
    
    # Import ResNet model
    from resnet34 import create_resnet34
    
    # Train ResNet-34
    print("\n" + "="*70)
    print("TRAINING RESNET-34")
    print("="*70)
    resnet_model = create_resnet34(num_classes=len(classes))
    resnet_trainer = Trainer(resnet_model, train_loader, val_loader, device, "ResNet-34")
    resnet_history = resnet_trainer.train(NUM_EPOCHS, LEARNING_RATE)
    resnet_trainer.plot_history('resnet34_history.png')
    resnet_trainer.save_history('resnet34_history.json')
    
    print("\n" + "="*70)
    print("RESNET-34 TRAINING COMPLETE!")
    print("="*70)
    print(f"Final Train Acc: {resnet_history['train_acc'][-1]:.2f}%")
    print(f"Final Val Acc:   {resnet_history['val_acc'][-1]:.2f}%")
    print(f"Best Val Acc:    {max(resnet_history['val_acc']):.2f}%")
    print("="*70)

if __name__ == "__main__":
    main()
