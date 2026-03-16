import torch
import torch.nn as nn
import torch.nn.functional as F

class PianoNet(nn.Module):
    def __init__(self, num_classes=88):
        super(PianoNet, self).__init__()
        
        # Input shape: [batch, 1, 128, time_steps]
        # time_steps will be ~32 for 8192 samples (8192 / hop_length 256)
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        # Calculate feature size after 3 poolings
        # Height: 128 -> 64 -> 32 -> 16
        # Width (time): ~32 -> 16 -> 8 -> 4
        # Flat size will be 128 (channels) * 16 * 4  = 8192
        # But time steps might vary slightly depending on hop length padding,
        # so we will use AdaptiveAvgPool2d to ensure a fixed size before fully connected layers.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 4)) # Results in 128 * 8 * 4 = 4096 flat size
        
        self.fc1 = nn.Linear(128 * 8 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Pool to fixed size regardless of exact time steps
        x = self.adaptive_pool(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Classifier
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)
        
        return out

if __name__ == '__main__':
    # Test model shape
    model = PianoNet()
    # Mock Mel Spectrogram
    mock_input = torch.randn(2, 1, 128, 32)
    output = model(mock_input)
    print(f"Model output shape: {output.shape} (Expected: [2, 88])")
