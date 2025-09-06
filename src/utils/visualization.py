"""Visualization and plotting utilities."""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torch.utils.data import DataLoader
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .config import CLASS_NAMES, CLASS_COLORS
from .image_processing import ImageProcessor


class Visualizer:
    """Visualization utilities for model predictions and results"""
    
    @staticmethod
    def mask_to_color(mask: np.ndarray) -> np.ndarray:
        """Convert segmentation mask to colored image"""
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in enumerate(CLASS_COLORS):
            colored[mask == class_id] = color
        
        return colored
    
    @staticmethod
    def visualize_predictions(model: torch.nn.Module, 
                            dataloader: DataLoader,
                            device: torch.device, 
                            num_samples: int = 4,
                            save_path: str = "predictions.png") -> None:
        """Visualize model predictions on validation data"""
        model.eval()
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        samples_shown = 0
        
        with torch.no_grad():
            for images, masks in dataloader:
                if samples_shown >= num_samples:
                    break
                
                images = images.to(device)
                masks = masks.to(device)
                
                predictions = model(images)
                pred_masks = predictions.argmax(dim=1)
                
                batch_size = min(images.size(0), num_samples - samples_shown)
                
                for i in range(batch_size):
                    # Process image
                    img_np = ImageProcessor.denormalize_image(images[i].cpu())
                    img_np = img_np.permute(1, 2, 0).numpy()
                    
                    # Process masks
                    mask_np = masks[i].cpu().numpy()
                    pred_np = pred_masks[i].cpu().numpy()
                    
                    # Convert to colors
                    gt_colored = Visualizer.mask_to_color(mask_np)
                    pred_colored = Visualizer.mask_to_color(pred_np)
                    
                    # Calculate IoU (avoiding circular import)
                    def compute_simple_iou(pred, gt):
                        intersection = np.logical_and(pred, gt).sum()
                        union = np.logical_or(pred, gt).sum()
                        return intersection / (union + 1e-6) if union > 0 else 0.0
                    
                    miou = compute_simple_iou(pred_np, mask_np)
                    
                    # Create overlay
                    overlay = cv2.addWeighted(
                        (img_np * 255).astype(np.uint8), 0.6,
                        pred_colored, 0.4, 0
                    )
                    
                    # Plot
                    row = samples_shown
                    axes[row, 0].imshow(img_np)
                    axes[row, 0].set_title('Input Image')
                    axes[row, 0].axis('off')
                    
                    axes[row, 1].imshow(gt_colored)
                    axes[row, 1].set_title('Ground Truth')
                    axes[row, 1].axis('off')
                    
                    axes[row, 2].imshow(pred_colored)
                    axes[row, 2].set_title(f'Prediction (mIoU: {miou:.3f})')
                    axes[row, 2].axis('off')
                    
                    axes[row, 3].imshow(overlay)
                    axes[row, 3].set_title('Overlay')
                    axes[row, 3].axis('off')
                    
                    samples_shown += 1
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization to {save_path}")
    
    @staticmethod
    def visualize_tryon_result(result: Dict, save_path: str = None) -> None:
        """Visualize virtual try-on results"""
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original Image
        axes[0].imshow(result['original'])
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Body Segmentation
        axes[1].imshow(result['segmentation'], cmap='tab20')
        axes[1].set_title('Body Segmentation', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Target Area
        axes[2].imshow(result['original'])
        axes[2].imshow(result['mask'], alpha=0.5, cmap='Reds')
        axes[2].set_title('Target Area', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        # Final Result
        axes[3].imshow(result['result'])
        axes[3].set_title('Virtual Try-On Result', fontsize=12, fontweight='bold')
        axes[3].axis('off')
        
        # Set main title as the prompt
        plt.suptitle(f'"{result["prompt"]}"', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()
    
    @staticmethod
    def visualize_batch_results(results: List[Dict], save_dir: str = "results") -> None:
        """Visualize multiple try-on results"""
        os.makedirs(save_dir, exist_ok=True)
        
        for i, result in enumerate(results):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_prompt = "".join(c for c in result['prompt'][:30] 
                                if c.isalnum() or c in (' ', '-', '_')).rstrip()
            save_path = os.path.join(save_dir, f"result_{i+1}_{safe_prompt}_{timestamp}.png")
            
            Visualizer.visualize_tryon_result(result, save_path)
    
    @staticmethod
    def plot_training_metrics(train_losses: List[float], 
                            val_losses: List[float],
                            val_mious: List[float],
                            save_path: str = "training_metrics.png") -> None:
        """Plot training metrics over epochs"""
        epochs = range(1, len(train_losses) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # mIoU plot
        ax2.plot(epochs, val_mious, 'g-', label='Validation mIoU')
        ax2.set_title('Validation mIoU')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mIoU')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved training metrics to {save_path}")
    
    @staticmethod
    def save_generated_images(images: List[Image.Image], 
                            prompts: List[str],
                            save_dir: str = "generated") -> None:
        """Save generated garment images"""
        os.makedirs(save_dir, exist_ok=True)
        
        for i, (image, prompt) in enumerate(zip(images, prompts)):
            safe_prompt = "".join(c for c in prompt[:30] 
                                if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"garment_{i+1}_{safe_prompt}.png"
            save_path = os.path.join(save_dir, filename)
            
            image.save(save_path)
            print(f"Saved generated image: {save_path}")
    
    @staticmethod
    def display_comparison(original: Image.Image, 
                         generated: Image.Image,
                         title: str = "Comparison") -> None:
        """Display side-by-side comparison of original and generated images"""
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(original)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(generated)
        axes[1].set_title('Generated')
        axes[1].axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

# Add these functions outside the Visualizer class
def visualize_predictions(image, mask, class_names, colors):
    """Standalone function to visualize predictions"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth/Prediction mask
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(colors):
        colored_mask[mask == class_id] = color
    
    axes[1].imshow(colored_mask)
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')
    
    # Overlay
    overlay = cv2.addWeighted(np.array(image), 0.6, colored_mask, 0.4, 0)
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig

def save_visualization(fig, save_path):
    """Save visualization figure"""
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)