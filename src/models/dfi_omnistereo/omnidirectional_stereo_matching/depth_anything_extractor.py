import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../depth_foundation_model')))
from depth_anything import DepthAnythingCore


class SubModule(nn.Module):
    """
    Base class for submodules, inheriting from PyTorch's nn.Module.
    """
    def __init__(self):
        super(SubModule, self).__init__()


class DepthAnythingExtractor(SubModule):
    """
    Extractor module for Depth Anything features.

    This module uses the DepthAnythingCore to extract relative depth and features.
    """
    def __init__(self, cfg):
        """
        Initialize the DepthAnythingExtractor.

        Parameters:
        - cfg: Configuration object containing model parameters.
        """
        super(DepthAnythingExtractor, self).__init__()
        self.cfg = cfg

        # Prepare arguments for DepthAnythingCore
        kwargs = {
            "img_size": list(self.cfg.img_size),
            "depth_anything_version": self.cfg.depth_anything_version
        }
        resize = False

        # Build the DepthAnythingCore
        self.core = DepthAnythingCore.build(
            train_midas=self.cfg.train_depth_anything,
            train_encoder=self.cfg.train_encoder,
            fetch_features=True,
            freeze_bn=self.cfg.freeze_depth_anything_bn,
            resize=resize,
            **kwargs
        )

    def forward(self, x):
        """
        Forward pass through the DepthAnythingExtractor.

        Parameters:
        - x: Input tensor.

        Returns:
        - rel_depth: Relative depth predictions.
        - out: Extracted features.
        """
        rel_depth, out = self.core(x, return_rel_depth=True)
        return rel_depth, out


class DepthAnythingFeature(SubModule):
    """
    Feature extraction module for Depth Anything.

    This module processes multi-scale features extracted by DepthAnythingCore.
    """
    def __init__(self, cfg):
        """
        Initialize the DepthAnythingFeature module.

        Parameters:
        - cfg: Configuration object containing model parameters.
        """
        super(DepthAnythingFeature, self).__init__()
        self.cfg = cfg

        # Determine the number of input channels based on the version
        if self.cfg.depth_anything_version == 'bv2':
            depth_anything_channels = 128
        elif self.cfg.depth_anything_version == 'lv2':
            depth_anything_channels = 256
        elif self.cfg.depth_anything_version == 'sv2':
            depth_anything_channels = 64
        else:
            raise ValueError(f"Unsupported depth_anything_version: {self.cfg.depth_anything_version}")

        # Define 1x1 convolution layers for multi-scale feature extraction
        self.conv1x1_4 = Conv1x1(in_channels=depth_anything_channels, out_channels=48)
        self.conv1x1_8 = Conv1x1(in_channels=depth_anything_channels, out_channels=64)
        self.conv1x1_16 = Conv1x1(in_channels=depth_anything_channels, out_channels=192)
        self.conv1x1_32 = Conv1x1(in_channels=depth_anything_channels, out_channels=160)

    def forward(self, out, h, w):
        """
        Forward pass through the DepthAnythingFeature module.

        Parameters:
        - out: Multi-scale feature maps from DepthAnythingCore.
        - h: Height of the original input image.
        - w: Width of the original input image.

        Returns:
        - List of processed feature maps at different scales.
        """
        # Extract multi-scale features
        out32 = out[1]
        out16 = out[2]
        out8 = out[3]
        out4 = out[4]

        # Resize feature maps to match the target resolution
        out4 = F.interpolate(out4, size=(h // 4, w // 4), mode='bilinear', align_corners=True)
        out8 = F.interpolate(out8, size=(h // 8, w // 8), mode='bilinear', align_corners=True)
        out16 = F.interpolate(out16, size=(h // 16, w // 16), mode='bilinear', align_corners=True)
        out32 = F.interpolate(out32, size=(h // 32, w // 32), mode='bilinear', align_corners=True)

        # Apply 1x1 convolutions to the resized feature maps
        x32 = self.conv1x1_32(out32)
        x16 = self.conv1x1_16(out16)
        x8 = self.conv1x1_8(out8)
        x4 = self.conv1x1_4(out4)

        return [x4, x8, x16, x32]


class Conv1x1(nn.Module):
    """
    1x1 Convolutional Layer with Instance Normalization and Leaky ReLU activation.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initialize the Conv1x1 module.

        Parameters:
        - in_channels: Number of input channels.
        - out_channels: Number of output channels.
        """
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.IN = nn.InstanceNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU()

        # Initialize weights to identity for the first min(out_channels, in_channels) channels
        with torch.no_grad():
            for i in range(min(out_channels, in_channels)):
                self.conv.weight[i, i, 0, 0] = 1.0

    def forward(self, x):
        """
        Forward pass through the Conv1x1 module.

        Parameters:
        - x: Input tensor.

        Returns:
        - Processed tensor after convolution, normalization, and activation.
        """
        x = self.conv(x)
        x = self.IN(x)
        x = self.leaky_relu(x)
        return x