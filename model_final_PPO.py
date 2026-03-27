import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CarRacingCNN(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network (Nature CNN architecture) designed for the CarRacing-v3 environment.
    This 'Final' version is optimized to read 4 stacked frames and uses a more aggressive approach for down sampling data.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        # Inherits functions from Stable-Baselines3 BaseFeaturesExtractor
        super().__init__(observation_space, features_dim)

        # Convolutional Layers for feature extraction from images
        # in_channels=12 because we stack 4 RGB frames (4 frames * 3 color channels = 12), providing temporal memory.
        # Layer 1: Detects basic edges, lines, and boundaries (track vs grass).
        # Layer 2: Detects more complex shapes like the car and curves.
        # Layer 3: Consolidates the spatial and temporal information (speed, drift angle).
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU()
        )

        # Fully Connected (Linear) Layer
        # The 96x96 image is downsampled through the conv layers to a 3x3 grid with 64 channels.
        # We flatten this (64 * 3 * 3 = 576) and map it to 256 neurons for an optimal data compression ratio (approx. 2:1).
        self.fc = nn.Sequential(
            nn.Linear(64 * 3 * 3, features_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # Forward pass through the convolutional layers
        x = self.conv(x)
        # Flatten the 3D tensor into a 1D vector for the linear layers
        x = x.reshape(x.size(0), -1)
        # Returns the final 256-dimensional feature vector to the PPO policy.
        return self.fc(x)