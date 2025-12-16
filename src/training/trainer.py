"""Training loop with logging, checkpointing, and metrics tracking."""

import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .metrics import compute_metrics


@dataclass
class TrainingStats:
    """Container for training statistics."""
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    token_accuracies: list[float] = field(default_factory=list)
    sequence_accuracies: list[float] = field(default_factory=list)
    bleu_scores: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    epoch_times: list[float] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'token_accuracies': self.token_accuracies,
            'sequence_accuracies': self.sequence_accuracies,
            'bleu_scores': self.bleu_scores,
            'learning_rates': self.learning_rates,
            'epoch_times': self.epoch_times,
        }


class Trainer:
    """
    Training class for image-to-LaTeX models.
    
    Handles:
    - Training loop with gradient accumulation
    - Validation and metric computation
    - Checkpointing and early stopping
    - Learning rate scheduling
    - TensorBoard logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0,
        accumulation_steps: int = 1,
        checkpoint_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None,
        pad_id: int = 0,
        eos_id: int = 2,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            learning_rate: Base learning rate
            weight_decay: Weight decay for AdamW
            warmup_steps: Number of warmup steps
            max_grad_norm: Maximum gradient norm for clipping
            accumulation_steps: Gradient accumulation steps
            checkpoint_dir: Directory for checkpoints
            log_dir: Directory for TensorBoard logs
            pad_id: Padding token ID
            eos_id: End-of-sequence token ID
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.accumulation_steps = accumulation_steps
        self.pad_id = pad_id
        self.eos_id = eos_id
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # LR scheduler (will be set in train method)
        self.scheduler = None
        self.warmup_steps = warmup_steps
        
        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.log_dir = Path(log_dir) if log_dir else None
        self.writer = None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(self.log_dir))
        
        # Stats tracking
        self.stats = TrainingStats()
        self.best_val_loss = float('inf')
        self.global_step = 0
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            leave=False,
        )
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress):
            # Move batch to device
            images = batch['images'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            # Forward pass
            loss = self.model.get_loss(images, input_ids, target_ids)
            loss = loss / self.accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )
                
                # Optimizer step
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Log to TensorBoard
                if self.writer and self.global_step % 100 == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar('train/loss', loss.item() * self.accumulation_steps, self.global_step)
                    self.writer.add_scalar('train/lr', lr, self.global_step)
            
            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1
            
            progress.set_postfix({
                'loss': f"{loss.item() * self.accumulation_steps:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """Run validation and compute metrics."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        # Log validation progress
        num_val_batches = len(self.val_loader)
        val_progress = tqdm(self.val_loader, desc="Validating", leave=False)
        
        for batch_idx, batch in enumerate(val_progress):
            images = batch['images'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            # Compute loss
            loss = self.model.get_loss(images, input_ids, target_ids)
            current_loss = loss.item()
            total_loss += current_loss
            
            # Update progress bar
            val_progress.set_postfix({'val_loss': f"{current_loss:.4f}"})
            
            # Log intermediate validation loss to TensorBoard
            if self.writer and batch_idx % 50 == 0:
                step = self.global_step + batch_idx  # Use pseudo-step for visualization
                self.writer.add_scalar('val/batch_loss', current_loss, step)
            
            # Generate predictions (greedy)
            generated = self.model.generate(images, temperature=0)
            
            # Collect for metrics
            # Strip SOS token from predictions (first token)
            generated = generated[:, 1:]
            
            all_predictions.extend(generated.cpu().tolist())
            all_targets.extend(target_ids.cpu().tolist())
        
        avg_loss = total_loss / num_val_batches
        
        # Compute metrics
        metrics = compute_metrics(
            all_predictions,
            all_targets,
            pad_id=self.pad_id,
            eos_id=self.eos_id,
        )
        metrics['loss'] = avg_loss
        
        # Log to TensorBoard
        if self.writer:
            self.writer.add_scalar('val/loss', avg_loss, epoch)
            self.writer.add_scalar('val/token_accuracy', metrics['token_accuracy'], epoch)
            self.writer.add_scalar('val/sequence_accuracy', metrics['sequence_accuracy'], epoch)
            self.writer.add_scalar('val/bleu', metrics['bleu'], epoch)
        
        return metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'stats': self.stats.to_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
        
        # Save periodic checkpoint
        if epoch % 5 == 0:
            epoch_path = self.checkpoint_dir / f'epoch_{epoch}.pt'
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            # Update total_steps in the loaded state dict to match the current configuration
            # This prevents "Tried to step X times. The specified number of total steps is Y" errors
            # when resuming with OneCycleLR
            if 'total_steps' in checkpoint['scheduler_state_dict']:
                checkpoint['scheduler_state_dict']['total_steps'] = self.scheduler.total_steps
            
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # Restore stats
        stats_dict = checkpoint.get('stats', {})
        for key, value in stats_dict.items():
            setattr(self.stats, key, value)
        
        return checkpoint.get('epoch', 0)
    
    def train(
        self,
        num_epochs: int,
        eval_every: int = 1,
        patience: int = 10,
        resume_from: Optional[Path] = None,
    ) -> TrainingStats:
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs to train
            eval_every: Evaluate every N epochs
            patience: Early stopping patience
            resume_from: Checkpoint path to resume from
            
        Returns:
            Training statistics
        """
        start_epoch = 0
        
        # Initialize scheduler
        # Add a small buffer to total_steps to prevent OneCycleLR from crashing
        # if the dataloader has one extra batch or rounding issues occur
        steps_per_epoch = len(self.train_loader) // self.accumulation_steps
        total_steps = steps_per_epoch * num_epochs + 100
        
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_steps=total_steps,
            pct_start=min(0.1, self.warmup_steps / total_steps),
            anneal_strategy='cos',
        )

        # Resume from checkpoint
        if resume_from and resume_from.exists():
            print(f"Resuming from {resume_from}")
            start_epoch = self.load_checkpoint(resume_from) + 1
        
        # Training loop
        epochs_without_improvement = 0
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            self.stats.train_losses.append(train_loss)
            self.stats.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
            # Validate
            if self.val_loader and (epoch + 1) % eval_every == 0:
                metrics = self.validate(epoch)
                val_loss = metrics.get('loss', float('inf'))
                
                self.stats.val_losses.append(val_loss)
                self.stats.token_accuracies.append(metrics.get('token_accuracy', 0))
                self.stats.sequence_accuracies.append(metrics.get('sequence_accuracy', 0))
                self.stats.bleu_scores.append(metrics.get('bleu', 0))
                
                # Check for improvement
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                
                print(
                    f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, "
                    f"token_acc={metrics.get('token_accuracy', 0):.4f}, "
                    f"seq_acc={metrics.get('sequence_accuracy', 0):.4f}, "
                    f"bleu={metrics.get('bleu', 0):.2f}"
                )
                
                # Save checkpoint
                self.save_checkpoint(epoch, is_best)
                
                # Early stopping
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            else:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}")
                self.save_checkpoint(epoch)
            
            epoch_time = time.time() - epoch_start
            self.stats.epoch_times.append(epoch_time)
        
        # Close TensorBoard writer
        if self.writer:
            self.writer.close()
        
        return self.stats

