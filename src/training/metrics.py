"""Evaluation metrics for segmentation tasks."""

import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics import confusion_matrix


class Metrics:
    """Metrics computation for segmentation tasks"""
    
    @staticmethod
    def compute_miou(predictions: np.ndarray, targets: np.ndarray,
                     num_classes: int, ignore_index: int = 255) -> Tuple[float, List[float]]:
        """
        Compute mean Intersection over Union.
        
        Args:
            predictions: Predicted segmentation masks
            targets: Ground truth segmentation masks
            num_classes: Number of classes
            ignore_index: Index to ignore in computation
            
        Returns:
            Tuple of (mean_iou, per_class_ious)
        """
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Filter out ignored pixels
        valid_mask = targets != ignore_index
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]
        
        ious = []
        for class_id in range(num_classes):
            pred_mask = predictions == class_id
            target_mask = targets == class_id
            
            intersection = np.logical_and(pred_mask, target_mask).sum()
            union = np.logical_or(pred_mask, target_mask).sum()
            
            if target_mask.sum() == 0:  # Class not present
                ious.append(1.0)
            else:
                ious.append(intersection / (union + 1e-6))
        
        return float(np.mean(ious)), ious
    
    @staticmethod
    def compute_pixel_accuracy(predictions: np.ndarray, targets: np.ndarray,
                              ignore_index: int = 255) -> float:
        """
        Compute pixel-wise accuracy.
        
        Args:
            predictions: Predicted segmentation masks
            targets: Ground truth segmentation masks
            ignore_index: Index to ignore in computation
            
        Returns:
            Pixel accuracy
        """
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Filter out ignored pixels
        valid_mask = targets != ignore_index
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]
        
        if len(predictions) == 0:
            return 0.0
        
        correct = (predictions == targets).sum()
        total = len(predictions)
        
        return float(correct / total)
    
    @staticmethod
    def compute_class_accuracy(predictions: np.ndarray, targets: np.ndarray,
                              num_classes: int, ignore_index: int = 255) -> List[float]:
        """
        Compute per-class accuracy.
        
        Args:
            predictions: Predicted segmentation masks
            targets: Ground truth segmentation masks
            num_classes: Number of classes
            ignore_index: Index to ignore in computation
            
        Returns:
            List of per-class accuracies
        """
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Filter out ignored pixels
        valid_mask = targets != ignore_index
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]
        
        accuracies = []
        for class_id in range(num_classes):
            class_mask = targets == class_id
            if class_mask.sum() == 0:  # Class not present
                accuracies.append(1.0)
            else:
                class_correct = (predictions[class_mask] == class_id).sum()
                class_total = class_mask.sum()
                accuracies.append(float(class_correct / class_total))
        
        return accuracies
    
    @staticmethod
    def compute_confusion_matrix(predictions: np.ndarray, targets: np.ndarray,
                               num_classes: int, ignore_index: int = 255) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Args:
            predictions: Predicted segmentation masks
            targets: Ground truth segmentation masks
            num_classes: Number of classes
            ignore_index: Index to ignore in computation
            
        Returns:
            Confusion matrix
        """
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Filter out ignored pixels
        valid_mask = targets != ignore_index
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]
        
        return confusion_matrix(targets, predictions, labels=list(range(num_classes)))
    
    @staticmethod
    def compute_dice_coefficient(predictions: np.ndarray, targets: np.ndarray,
                               num_classes: int, ignore_index: int = 255) -> Tuple[float, List[float]]:
        """
        Compute Dice coefficient.
        
        Args:
            predictions: Predicted segmentation masks
            targets: Ground truth segmentation masks
            num_classes: Number of classes
            ignore_index: Index to ignore in computation
            
        Returns:
            Tuple of (mean_dice, per_class_dice)
        """
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Filter out ignored pixels
        valid_mask = targets != ignore_index
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]
        
        dice_scores = []
        for class_id in range(num_classes):
            pred_mask = predictions == class_id
            target_mask = targets == class_id
            
            intersection = np.logical_and(pred_mask, target_mask).sum()
            total = pred_mask.sum() + target_mask.sum()
            
            if total == 0:  # Class not present in both
                dice_scores.append(1.0)
            else:
                dice_scores.append(2.0 * intersection / total)
        
        return float(np.mean(dice_scores)), dice_scores
    
    @staticmethod
    def compute_comprehensive_metrics(predictions: np.ndarray, targets: np.ndarray,
                                    num_classes: int, ignore_index: int = 255,
                                    class_names: List[str] = None) -> Dict:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            predictions: Predicted segmentation masks
            targets: Ground truth segmentation masks
            num_classes: Number of classes
            ignore_index: Index to ignore in computation
            class_names: Names of classes for reporting
            
        Returns:
            Dictionary containing all metrics
        """
        # Compute all metrics
        miou, per_class_ious = Metrics.compute_miou(predictions, targets, num_classes, ignore_index)
        pixel_acc = Metrics.compute_pixel_accuracy(predictions, targets, ignore_index)
        class_accs = Metrics.compute_class_accuracy(predictions, targets, num_classes, ignore_index)
        dice_coeff, per_class_dice = Metrics.compute_dice_coefficient(predictions, targets, num_classes, ignore_index)
        
        # Organize results
        results = {
            'overall': {
                'miou': miou,
                'pixel_accuracy': pixel_acc,
                'mean_dice': dice_coeff
            },
            'per_class': {
                'iou': per_class_ious,
                'accuracy': class_accs,
                'dice': per_class_dice
            }
        }
        
        # Add class names if provided
        if class_names:
            results['class_names'] = class_names
            results['per_class_named'] = {}
            for i, name in enumerate(class_names):
                if i < len(per_class_ious):
                    results['per_class_named'][name] = {
                        'iou': per_class_ious[i],
                        'accuracy': class_accs[i],
                        'dice': per_class_dice[i]
                    }
        
        return results
    
    @staticmethod
    def print_metrics_summary(metrics: Dict, top_n: int = 3):
        """
        Print a formatted summary of metrics.
        
        Args:
            metrics: Metrics dictionary from compute_comprehensive_metrics
            top_n: Number of best/worst classes to show
        """
        print("=" * 60)
        print("EVALUATION METRICS SUMMARY")
        print("=" * 60)
        
        # Overall metrics
        print(f"Overall mIoU: {metrics['overall']['miou']:.4f}")
        print(f"Pixel Accuracy: {metrics['overall']['pixel_accuracy']:.4f}")
        print(f"Mean Dice: {metrics['overall']['mean_dice']:.4f}")
        
        if 'per_class_named' in metrics:
            # Best performing classes
            class_metrics = metrics['per_class_named']
            sorted_by_iou = sorted(class_metrics.items(), key=lambda x: x[1]['iou'], reverse=True)
            
            print(f"\nTop {top_n} performing classes (by IoU):")
            print("-" * 40)
            for i, (name, metric) in enumerate(sorted_by_iou[:top_n]):
                print(f"{i+1}. {name}: {metric['iou']:.3f}")
            
            print(f"\nWorst {top_n} performing classes (by IoU):")
            print("-" * 40)
            for i, (name, metric) in enumerate(sorted_by_iou[-top_n:][::-1]):
                print(f"{i+1}. {name}: {metric['iou']:.3f}")
        
        print("=" * 60)