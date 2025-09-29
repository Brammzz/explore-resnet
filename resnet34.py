import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class ResidualBlock(nn.Module):
    """
    Residual Block WITH residual connection (skip connection).
    This is the key difference from PlainBlock - we add the identity mapping.
    
    MODIFICATION FROM PLAIN-34:
    - Added skip connection: out += identity before final ReLU
    - This allows gradient to flow directly through the network
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Downsample layer for dimension matching (if needed)
        self.downsample = downsample
        
    def forward(self, x):
        # Store input for residual connection
        identity = x
        
        # First conv + bn + relu
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # Second conv + bn
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply downsample to identity if needed (for dimension matching)
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        # *** KEY MODIFICATION: ADD RESIDUAL CONNECTION ***
        # This is the main difference from PlainBlock!
        # We add the input (identity) to the output
        out += identity  # <-- RESIDUAL CONNECTION ADDED HERE
        
        # Apply ReLU after addition
        out = F.relu(out)
        
        return out

class ResNet34(nn.Module):
    """
    ResNet-34 Network: Complete implementation with residual connections.
    
    Architecture:
    - Initial conv layer (7x7, stride=2)
    - MaxPool (3x3, stride=2)
    - 4 stages of Residual blocks:
      - Stage 1: 3 blocks, 64 channels
      - Stage 2: 4 blocks, 128 channels, stride=2 for first block
      - Stage 3: 6 blocks, 256 channels, stride=2 for first block
      - Stage 4: 3 blocks, 512 channels, stride=2 for first block
    - Global Average Pool
    - Fully Connected layer
    
    MODIFICATION FROM PLAIN-34:
    - Changed PlainBlock to ResidualBlock (with skip connections)
    - This allows deeper networks to train effectively
    """
    
    def __init__(self, num_classes=5):
        super(ResNet34, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual block stages (MODIFIED: using ResidualBlock instead of PlainBlock)
        self.stage1 = self._make_stage(64, 64, 3, stride=1)    # 3 blocks, 64 channels
        self.stage2 = self._make_stage(64, 128, 4, stride=2)   # 4 blocks, 128 channels
        self.stage3 = self._make_stage(128, 256, 6, stride=2)  # 6 blocks, 256 channels
        self.stage4 = self._make_stage(256, 512, 3, stride=2)  # 3 blocks, 512 channels
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        """
        Create a stage consisting of multiple ResidualBlocks.
        
        Args:
            in_channels: Input channels for the first block
            out_channels: Output channels for all blocks in this stage
            num_blocks: Number of blocks in this stage
            stride: Stride for the first block (usually 1 or 2)
        """
        downsample = None
        
        # If we need to change dimensions or stride, create downsample layer
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        
        # First block (may have stride=2 and different input/output channels)
        # MODIFIED: using ResidualBlock instead of PlainBlock
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        
        # Remaining blocks (stride=1, same input/output channels)
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial conv + bn + relu + maxpool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        # Residual block stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        # Final classification layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def create_resnet34(num_classes=5):
    """
    Factory function to create ResNet-34 model.
    
    Args:
        num_classes: Number of output classes (default: 5 for Indonesian food dataset)
    
    Returns:
        ResNet34 model instance
    """
    return ResNet34(num_classes=num_classes)

def test_model():
    """
    Test function to verify the model works correctly.
    This function creates a model and prints its architecture summary.
    """
    print("Creating ResNet-34 model...")
    model = create_resnet34(num_classes=5)
    
    # Print model summary
    print("\n" + "="*50)
    print("RESNET-34 MODEL ARCHITECTURE SUMMARY")
    print("="*50)
    
    # Test with typical input size for image classification (224x224)
    try:
        summary(model, input_size=(1, 3, 224, 224), verbose=1)
    except Exception as e:
        print(f"Error in torchinfo summary: {e}")
        print("Trying manual forward pass...")
        
        # Manual test
        model.eval()
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224)
            output = model(test_input)
            print(f"Input shape: {test_input.shape}")
            print(f"Output shape: {output.shape}")
            print(f"Expected output shape: (1, 5)")
            print(f"Model works correctly: {output.shape == (1, 5)}")
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model

if __name__ == "__main__":
    # Test the model when running this file directly
    model = test_model()
    
    print("\n" + "="*50)
    print("MODEL READY FOR TRAINING!")
    print("="*50)
    print("Next steps:")
    print("1. Load your Indonesian food dataset")
    print("2. Set up data loaders")
    print("3. Define loss function and optimizer")
    print("4. Train the model")
    print("5. Compare with Plain-34 (without residual connections)")
    print("\n" + "="*50)
    print("KEY DIFFERENCES FROM PLAIN-34:")
    print("="*50)
    print("1. ResidualBlock has skip connection: out += identity")
    print("2. Allows gradients to flow directly through network")
    print("3. Solves degradation problem in deep networks")
    print("4. Same architecture depth, better performance expected")