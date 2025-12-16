"""Extract training metrics from TensorBoard logs and generate plots."""

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path


def load_metrics(log_dir: str) -> dict:
    """Load all scalar metrics from a TensorBoard log directory."""
    ea = EventAccumulator(log_dir)
    ea.Reload()
    
    metrics = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        metrics[tag] = {
            'steps': [e.step for e in events],
            'values': [e.value for e in events],
        }
    return metrics


def plot_training_curves(pretrain_metrics: dict, finetune_metrics: dict, output_dir: Path):
    """Generate training curve plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})
    
    # 1. Pretraining loss curves
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Train loss (subsample for clarity)
    train_steps = pretrain_metrics['train/loss']['steps'][::10]
    train_values = pretrain_metrics['train/loss']['values'][::10]
    ax.plot(train_steps, train_values, 'b-', alpha=0.7, label='Train Loss', linewidth=1)
    
    # Val loss - scale steps to match epoch
    val_epochs = list(range(len(pretrain_metrics['val/loss']['values'])))
    val_values = pretrain_metrics['val/loss']['values']
    ax2 = ax.twiny()
    ax2.plot(val_epochs, val_values, 'r-o', label='Validation Loss', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch')
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Pretraining Loss (Printed Dataset)')
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    ax2.legend(loc='upper center')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pretrain_loss.png', bbox_inches='tight')
    plt.close()
    
    # 2. Pretraining metrics
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    epochs = list(range(len(pretrain_metrics['val/token_accuracy']['values'])))
    
    # Token accuracy
    ax = axes[0]
    ax.plot(epochs, [v * 100 for v in pretrain_metrics['val/token_accuracy']['values']], 
            'g-o', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Token Accuracy (%)')
    ax.set_title('Token Accuracy')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    # Sequence accuracy
    ax = axes[1]
    ax.plot(epochs, [v * 100 for v in pretrain_metrics['val/sequence_accuracy']['values']], 
            'orange', marker='o', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Sequence Accuracy (%)')
    ax.set_title('Sequence Accuracy (Exact Match)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    # BLEU
    ax = axes[2]
    ax.plot(epochs, pretrain_metrics['val/bleu']['values'], 
            'm-o', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('BLEU Score')
    ax.set_title('BLEU Score')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    plt.suptitle('Pretraining Validation Metrics', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'pretrain_metrics.png', bbox_inches='tight')
    plt.close()
    
    # 3. Fine-tuning loss
    fig, ax = plt.subplots(figsize=(8, 5))
    
    train_steps = finetune_metrics['train/loss']['steps'][::20]
    train_values = finetune_metrics['train/loss']['values'][::20]
    ax.plot(train_steps, train_values, 'b-', alpha=0.7, label='Train Loss', linewidth=1)
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Fine-tuning Loss (Handwritten Datasets)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'finetune_loss.png', bbox_inches='tight')
    plt.close()
    
    # 4. Fine-tuning metrics comparison
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    ft_epochs = list(range(len(finetune_metrics['val/token_accuracy']['values'])))
    
    # Token accuracy
    ax = axes[0]
    ax.plot(ft_epochs, [v * 100 for v in finetune_metrics['val/token_accuracy']['values']], 
            'g-o', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Token Accuracy (%)')
    ax.set_title('Token Accuracy')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([60, 100])
    
    # Sequence accuracy
    ax = axes[1]
    ax.plot(ft_epochs, [v * 100 for v in finetune_metrics['val/sequence_accuracy']['values']], 
            'orange', marker='o', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Sequence Accuracy (%)')
    ax.set_title('Sequence Accuracy (Exact Match)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([40, 80])
    
    # BLEU
    ax = axes[2]
    ax.plot(ft_epochs, finetune_metrics['val/bleu']['values'], 
            'm-o', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('BLEU Score')
    ax.set_title('BLEU Score')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([70, 100])
    
    plt.suptitle('Fine-tuning Validation Metrics (Handwritten Data)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'finetune_metrics.png', bbox_inches='tight')
    plt.close()
    
    # 5. Combined comparison: Before and After Fine-tuning
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics_names = ['Token Acc.', 'Seq. Acc.', 'BLEU']
    pretrain_final = [
        pretrain_metrics['val/token_accuracy']['values'][-1] * 100,
        pretrain_metrics['val/sequence_accuracy']['values'][-1] * 100,
        pretrain_metrics['val/bleu']['values'][-1],
    ]
    
    # Best fine-tune results: use best value for each metric
    finetune_best = [
        max(finetune_metrics['val/token_accuracy']['values']) * 100,
        max(finetune_metrics['val/sequence_accuracy']['values']) * 100,
        max(finetune_metrics['val/bleu']['values']),
    ]
    
    x = range(len(metrics_names))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], pretrain_final, width, label='After Pretraining', color='steelblue')
    bars2 = ax.bar([i + width/2 for i in x], finetune_best, width, label='After Fine-tuning', color='coral')
    
    ax.set_ylabel('Score (%)')
    ax.set_title('Performance Comparison: Pretraining vs Fine-tuning')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_pretrain_finetune.png', bbox_inches='tight')
    plt.close()
    
    # 6. Learning rate schedule
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    ax = axes[0]
    lr_steps = pretrain_metrics['train/lr']['steps']
    lr_values = pretrain_metrics['train/lr']['values']
    ax.plot(lr_steps, lr_values, 'b-', linewidth=1.5)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Pretraining LR Schedule (OneCycleLR)')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    ax = axes[1]
    lr_steps = finetune_metrics['train/lr']['steps']
    lr_values = finetune_metrics['train/lr']['values']
    ax.plot(lr_steps, lr_values, 'r-', linewidth=1.5)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Fine-tuning LR Schedule (OneCycleLR)')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_rate_schedules.png', bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {output_dir}")
    
    # Print summary statistics
    print("\n=== TRAINING SUMMARY ===")
    print("\nPretraining (Printed Dataset):")
    print(f"  Epochs: {len(pretrain_metrics['val/loss']['values'])}")
    print(f"  Final Val Loss: {pretrain_metrics['val/loss']['values'][-1]:.4f}")
    print(f"  Final Token Accuracy: {pretrain_metrics['val/token_accuracy']['values'][-1]*100:.2f}%")
    print(f"  Final Sequence Accuracy: {pretrain_metrics['val/sequence_accuracy']['values'][-1]*100:.2f}%")
    print(f"  Final BLEU: {pretrain_metrics['val/bleu']['values'][-1]:.2f}")
    
    print("\nFine-tuning (Handwritten Datasets):")
    print(f"  Epochs: {len(finetune_metrics['val/loss']['values'])}")
    print(f"  Best Val Loss: {min(finetune_metrics['val/loss']['values']):.4f}")
    print(f"  Best Token Accuracy: {max(finetune_metrics['val/token_accuracy']['values'])*100:.2f}%")
    print(f"  Best Sequence Accuracy: {max(finetune_metrics['val/sequence_accuracy']['values'])*100:.2f}%")
    print(f"  Best BLEU: {max(finetune_metrics['val/bleu']['values']):.2f}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract TensorBoard metrics and generate plots')
    parser.add_argument('--pretrain-log', default='logs/pretrain', help='Pretrain log directory')
    parser.add_argument('--finetune-log', default='logs/finetune', help='Finetune log directory')
    parser.add_argument('--output-dir', default='report/Plots', help='Output directory for plots')
    
    args = parser.parse_args()
    
    pretrain_metrics = load_metrics(args.pretrain_log)
    finetune_metrics = load_metrics(args.finetune_log)
    
    plot_training_curves(pretrain_metrics, finetune_metrics, Path(args.output_dir))
