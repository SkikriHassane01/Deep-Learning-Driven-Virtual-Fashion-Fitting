"""Human Parsing model with DeepLab-style architecture and self-correction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Any


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module"""
    
    def __init__(self, in_channels: int, out_channels: int, rates: Tuple[int, ...] = (6, 12, 18)):
        super().__init__()
        
        # 1x1 convolution
        self.conv1x1 = self._make_branch(in_channels, out_channels, kernel_size=1)
        
        # Atrous convolutions
        self.atrous_branches = nn.ModuleList([
            self._make_branch(in_channels, out_channels, kernel_size=3, dilation=rate)
            for rate in rates
        ])
        
        # Global average pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Projection layer
        num_branches = len(rates) + 2  # Atrous + 1x1 + global
        self.projection = nn.Sequential(
            nn.Conv2d(out_channels * num_branches, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def _make_branch(self, in_channels: int, out_channels: int, 
                     kernel_size: int, dilation: int = 1) -> nn.Sequential:
        """Create a convolutional branch"""
        padding = 0 if kernel_size == 1 else dilation
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 
                     padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Collect features from all branches
        features = [self.conv1x1(x)]
        features.extend([branch(x) for branch in self.atrous_branches])
        
        # Global pooling branch
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=x.shape[-2:], 
                                   mode="bilinear", align_corners=False)
        features.append(global_feat)
        
        # Concatenate and project
        concatenated = torch.cat(features, dim=1)
        return self.projection(concatenated)


class SelfCorrectionModule(nn.Module):
    """Self-correction module with edge awareness"""
    
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        
        # Edge detection branch
        self.edge_branch = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1)  # Output edge logits
        )
        
        # Refinement branch
        self.refinement_branch = nn.Sequential(
            nn.Conv2d(in_channels + 1, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Predict edges
        edge_logits = self.edge_branch(features)
        
        # Concatenate features with edge information
        enhanced_features = torch.cat([features, edge_logits], dim=1)
        
        # Refine predictions
        refined_logits = self.refinement_branch(enhanced_features)
        
        return refined_logits, edge_logits


class HumanParsingNet(nn.Module):
    """Main Human Parsing Network with Self-Correction"""
    
    def __init__(self, num_classes: int = 18):
        super().__init__()
        
        # Load pretrained ResNet101 backbone
        backbone = models.resnet101(pretrained=True)
        
        # Extract backbone layers
        self.initial_layers = nn.Sequential(*list(backbone.children())[:5])  # Conv1 -> Layer1
        self.layer2 = nn.Sequential(*list(backbone.children())[5])
        self.layer3 = nn.Sequential(*list(backbone.children())[6])
        self.layer4 = nn.Sequential(*list(backbone.children())[7])
        
        # Low-level feature processing
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # ASPP module
        self.aspp = ASPP(2048, 256)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Output heads
        self.coarse_head = nn.Conv2d(256, num_classes, 1)
        self.self_correction = SelfCorrectionModule(256, num_classes)
    
    def forward(self, x: torch.Tensor) -> Any:
        input_shape = x.shape[-2:]
        
        # Backbone forward pass
        low_level = self.initial_layers(x)  # 256 channels
        x = self.layer2(low_level)          # 512 channels
        x = self.layer3(x)                  # 1024 channels
        x = self.layer4(x)                  # 2048 channels
        
        # Process low-level features
        low_level_features = self.low_level_conv(low_level)
        
        # ASPP
        aspp_features = self.aspp(x)
        
        # Upsample and concatenate
        aspp_features = F.interpolate(aspp_features, size=low_level_features.shape[-2:],
                                      mode="bilinear", align_corners=False)
        decoder_input = torch.cat([aspp_features, low_level_features], dim=1)
        
        # Decode
        decoder_features = self.decoder(decoder_input)
        
        # Generate outputs
        coarse_logits = self.coarse_head(decoder_features)
        refined_logits, edge_logits = self.self_correction(decoder_features)
        
        # Upsample to input resolution
        coarse_logits = F.interpolate(coarse_logits, size=input_shape,
                                      mode="bilinear", align_corners=False)
        refined_logits = F.interpolate(refined_logits, size=input_shape,
                                       mode="bilinear", align_corners=False)
        edge_logits = F.interpolate(edge_logits, size=input_shape,
                                    mode="bilinear", align_corners=False)
        
        if self.training:
            return coarse_logits, refined_logits, edge_logits
        else:
            return refined_logits