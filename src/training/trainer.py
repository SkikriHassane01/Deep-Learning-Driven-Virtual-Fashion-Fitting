"""Training pipeline and logic for Human Parsing model."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple, Optional

from .losses import EdgeAwareLoss
from .metrics import Metrics
from ..utils.config import Config


class CheckpointManager:
    """Manage model checkpoints and resume training"""
    
    @staticmethod
    def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                       scaler: GradScaler, epoch: int, best_miou: float,
                       loss: float, filepath: str):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_miou': best_miou,
            'loss': loss,
            'config': {
                'num_classes': Config.NUM_CLASSES,
                'input_size': Config.INPUT_SIZE,
                'learning_rates': {
                    'backbone': Config.LEARNING_RATE_BACKBONE,
                    'head': Config.LEARNING_RATE_HEAD
                }
            }
        }
        torch.save(checkpoint, filepath)
    
    @staticmethod
    def load_checkpoint(filepath: str, model: nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scaler: Optional[GradScaler] = None) -> Tuple[int, float]:
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=Config.DEVICE)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_miou = checkpoint['best_miou']
        
        print(f"Resumed from epoch {start_epoch}, best mIoU: {best_miou:.4f}")
        return start_epoch, best_miou


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_score: Validation score (higher is better)
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        
        return self.early_stop


class Trainer:
    """Main trainer class for Human Parsing model"""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, config: Config = None):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration object
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or Config()
        
        # Loss function
        self.criterion = EdgeAwareLoss(
            edge_weight=self.config.EDGE_WEIGHT,
            ignore_index=self.config.IGNORE_INDEX
        )
        
        # Optimizer with different learning rates
        self.optimizer = self._create_optimizer()
        
        # Mixed precision training
        self.scaler = GradScaler(enabled=self.config.DEVICE.type == "cuda")
        
        # Training state
        self.start_epoch = 0
        self.best_miou = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_mious = []
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=self.config.EARLY_STOPPING_PATIENCE)
        
        # Resume from checkpoint if specified
        if self.config.RESUME_FROM and os.path.exists(self.config.RESUME_FROM):
            self.start_epoch, self.best_miou = CheckpointManager.load_checkpoint(
                self.config.RESUME_FROM, self.model, self.optimizer, self.scaler
            )
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with different learning rates for backbone and head"""
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if "layer" in name or "initial" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": self.config.LEARNING_RATE_BACKBONE},
            {"params": head_params, "lr": self.config.LEARNING_RATE_HEAD}
        ], weight_decay=self.config.WEIGHT_DECAY)
        
        return optimizer
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.EPOCHS} [Train]")
        
        for images, masks in pbar:
            images = images.to(self.config.DEVICE)
            masks = masks.to(self.config.DEVICE)
            
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.config.DEVICE.type == "cuda"):
                coarse, refined, edges = self.model(images)
                loss, loss_dict = self.criterion(coarse, refined, edges, masks)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'coarse': f"{loss_dict['coarse']:.3f}",
                'refined': f"{loss_dict['refined']:.3f}",
                'edge': f"{loss_dict['edge']:.3f}"
            })
        
        avg_loss = total_loss / max(1, num_batches)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, epoch: int) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        all_mious = []
        
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.config.EPOCHS} [Val]"):
                images = images.to(self.config.DEVICE)
                masks = masks.to(self.config.DEVICE)
                
                with autocast(enabled=self.config.DEVICE.type == "cuda"):
                    output = self.model(images)
                    if isinstance(output, tuple):
                        predictions = output[1]  # Use refined predictions for validation
                    else:
                        predictions = output
                    loss = F.cross_entropy(predictions, masks, ignore_index=self.config.IGNORE_INDEX)
                
                total_loss += loss.item()
                
                # Compute mIoU
                pred_masks = predictions.argmax(dim=1)
                for i in range(images.size(0)):
                    miou, _ = Metrics.compute_miou(
                        pred_masks[i].cpu().numpy(),
                        masks[i].cpu().numpy(),
                        self.config.NUM_CLASSES,
                        self.config.IGNORE_INDEX
                    )
                    all_mious.append(miou)
        
        avg_loss = total_loss / max(1, len(self.val_loader))
        avg_miou = float(np.mean(all_mious))
        
        self.val_losses.append(avg_loss)
        self.val_mious.append(avg_miou)
        
        return avg_loss, avg_miou
    
    def save_best_model(self, epoch: int, miou: float):
        """Save the best model"""
        torch.save({
            "model": self.model.state_dict(),
            "best_miou": miou,
            "epoch": epoch,
            "config": {
                "num_classes": self.config.NUM_CLASSES,
                "input_size": self.config.INPUT_SIZE
            }
        }, self.config.MODEL_PATH)
        print(f"Saved best model with mIoU: {miou:.4f}")
    
    def train(self) -> float:
        """Main training loop"""
        print(f"Starting training on {self.config.DEVICE}")
        print(f"Training from epoch {self.start_epoch} to {self.config.EPOCHS}")
        
        for epoch in range(self.start_epoch, self.config.EPOCHS):
            # Training phase
            train_loss = self.train_epoch(epoch)
            
            # Validation phase
            val_loss, val_miou = self.validate(epoch)
            
            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_mIoU={val_miou:.4f}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(
                self.config.CHECKPOINT_DIR,
                f"checkpoint_epoch_{epoch+1}.pth"
            )
            CheckpointManager.save_checkpoint(
                self.model, self.optimizer, self.scaler,
                epoch, self.best_miou, train_loss, checkpoint_path
            )
            
            # Save best model
            if val_miou > self.best_miou:
                self.best_miou = val_miou
                self.save_best_model(epoch, val_miou)
            
            # Early stopping check
            if self.early_stopping(val_miou):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        print(f"Training completed. Best mIoU: {self.best_miou:.4f}")
        return self.best_miou
    
    def get_training_history(self) -> Dict:
        """Get training history for plotting"""
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_mious': self.val_mious,
            'best_miou': self.best_miou
        }