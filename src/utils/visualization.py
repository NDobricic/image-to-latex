"""Visualization utilities for training curves and results."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt


def plot_training_curves(
    train_losses: list[float],
    val_losses: Optional[list[float]] = None,
    token_accuracies: Optional[list[float]] = None,
    sequence_accuracies: Optional[list[float]] = None,
    bleu_scores: Optional[list[float]] = None,
    learning_rates: Optional[list[float]] = None,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot training curves.
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        token_accuracies: Token accuracies per epoch
        sequence_accuracies: Sequence accuracies per epoch
        bleu_scores: BLEU scores per epoch
        learning_rates: Learning rates per epoch
        save_path: Path to save the figure
        show: Whether to display the figure
        
    Returns:
        Matplotlib figure
    """
    # Determine number of subplots needed
    n_plots = 1  # Always have loss plot
    if token_accuracies or sequence_accuracies:
        n_plots += 1
    if bleu_scores:
        n_plots += 1
    if learning_rates:
        n_plots += 1
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Loss plot
    ax = axes[plot_idx]
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Train Loss', color='blue')
    if val_losses:
        val_epochs = range(1, len(val_losses) + 1)
        # Scale val_epochs if validation is less frequent
        if len(val_losses) < len(train_losses):
            scale = len(train_losses) // len(val_losses)
            val_epochs = [i * scale for i in range(1, len(val_losses) + 1)]
        ax.plot(val_epochs, val_losses, label='Val Loss', color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_idx += 1
    
    # Accuracy plot
    if token_accuracies or sequence_accuracies:
        ax = axes[plot_idx]
        if token_accuracies:
            ax.plot(range(1, len(token_accuracies) + 1), token_accuracies, 
                   label='Token Accuracy', color='green')
        if sequence_accuracies:
            ax.plot(range(1, len(sequence_accuracies) + 1), sequence_accuracies, 
                   label='Sequence Accuracy', color='red')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # BLEU plot
    if bleu_scores:
        ax = axes[plot_idx]
        ax.plot(range(1, len(bleu_scores) + 1), bleu_scores, 
               label='BLEU', color='purple')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('BLEU Score')
        ax.set_title('BLEU Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Learning rate plot
    if learning_rates:
        ax = axes[plot_idx]
        ax.plot(range(1, len(learning_rates) + 1), learning_rates, 
               label='Learning Rate', color='brown')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def save_plots(stats: dict, output_dir: Path):
    """
    Save all training plots.
    
    Args:
        stats: Training statistics dictionary
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Main training curves
    plot_training_curves(
        train_losses=stats.get('train_losses', []),
        val_losses=stats.get('val_losses', []),
        token_accuracies=stats.get('token_accuracies', []),
        sequence_accuracies=stats.get('sequence_accuracies', []),
        bleu_scores=stats.get('bleu_scores', []),
        learning_rates=stats.get('learning_rates', []),
        save_path=output_dir / 'training_curves.png',
    )
    plt.close()
    
    # Individual plots
    if stats.get('train_losses'):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(stats['train_losses'])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'train_loss.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    if stats.get('val_losses'):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(stats['val_losses'])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Validation Loss')
        ax.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'val_loss.png', dpi=150, bbox_inches='tight')
        plt.close()

