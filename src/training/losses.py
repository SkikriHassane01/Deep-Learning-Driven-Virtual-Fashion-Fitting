"""Custom loss functions for human parsing model training."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple


class EdgeAwareLoss(nn.Module):
    """Combined loss with edge awareness for human parsing"""
    
    def __init__(self, edge_weight: float = 0.4, ignore_index: int = 255):
        """
        Initialize edge-aware loss.
        
        Args:
            edge_weight: Weight for edge loss component
            ignore_index: Index to ignore in loss calculation
        """
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.bce_logits_loss = nn.BCEWithLogitsLoss()
        self.edge_weight = edge_weight
        self.ignore_index = ignore_index
    
    @torch.no_grad()
    def compute_edge_targets(self, masks: torch.Tensor) -> torch.Tensor:
        """Compute edge targets from segmentation masks"""
        batch_size = masks.shape[0]
        edges = []
        
        for i in range(batch_size):
            mask = masks[i].cpu().numpy().astype(np.int32)
            mask[mask == self.ignore_index] = -1
            
            # Compute gradients
            grad_y, grad_x = np.gradient(mask)
            edge = ((np.abs(grad_x) > 0) | (np.abs(grad_y) > 0)).astype(np.float32)
            edges.append(edge)
        
        return torch.from_numpy(np.stack(edges, axis=0)).to(masks.device)
    
    def forward(self, coarse_logits: torch.Tensor, refined_logits: torch.Tensor,
                edge_logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass of edge-aware loss.
        
        Args:
            coarse_logits: Coarse segmentation predictions
            refined_logits: Refined segmentation predictions 
            edge_logits: Edge predictions
            targets: Ground truth segmentation masks
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Segmentation losses
        coarse_loss = self.ce_loss(coarse_logits, targets)
        refined_loss = self.ce_loss(refined_logits, targets)
        
        # Edge loss
        edge_targets = self.compute_edge_targets(targets)
        valid_mask = (targets != self.ignore_index).float()
        edge_loss = self.bce_logits_loss(
            edge_logits.squeeze(1) * valid_mask,
            edge_targets * valid_mask
        )
        
        # Combined loss
        total_loss = coarse_loss + refined_loss + self.edge_weight * edge_loss
        
        # Loss components for logging
        loss_dict = {
            "coarse": coarse_loss.item(),
            "refined": refined_loss.item(),
            "edge": edge_loss.item(),
            "total": total_loss.item()
        }
        
        return total_loss, loss_dict


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, ignore_index: int = 255):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            ignore_index: Index to ignore in loss calculation
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass of focal loss"""
        ce_loss = self.ce_loss(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Only compute mean over valid pixels
        valid_mask = targets != self.ignore_index
        if valid_mask.sum() > 0:
            return focal_loss[valid_mask].mean()
        else:
            return focal_loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks"""
    
    def __init__(self, smooth: float = 1.0, ignore_index: int = 255):
        """
        Initialize Dice Loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
            ignore_index: Index to ignore in loss calculation
        """
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass of dice loss"""
        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=1)
        
        # Create one-hot encoding for targets
        num_classes = logits.shape[1]
        targets_one_hot = torch.zeros_like(probs)
        
        # Handle ignore index
        valid_mask = targets != self.ignore_index
        valid_targets = targets[valid_mask]
        
        if valid_targets.numel() > 0:
            targets_one_hot = targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
            targets_one_hot[:, self.ignore_index] = 0  # Zero out ignore class if exists
            
            # Compute dice coefficient
            intersection = (probs * targets_one_hot).sum(dim=(2, 3))
            union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss = 1.0 - dice.mean()
        else:
            dice_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        return dice_loss


class CombinedLoss(nn.Module):
    """Combined loss combining CrossEntropy and Dice losses"""
    
    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0, 
                 ignore_index: int = 255):
        """
        Initialize combined loss.
        
        Args:
            ce_weight: Weight for CrossEntropy loss
            dice_weight: Weight for Dice loss
            ignore_index: Index to ignore in loss calculation
        """
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Forward pass of combined loss"""
        ce_loss = self.ce_loss(logits, targets)
        dice_loss = self.dice_loss(logits, targets)
        
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        
        loss_dict = {
            "ce": ce_loss.item(),
            "dice": dice_loss.item(),
            "total": total_loss.item()
        }
        
        return total_loss, loss_dict