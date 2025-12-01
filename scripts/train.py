#!/usr/bin/env python3
"""Main training script for image-to-LaTeX models."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    LaTeXTokenizer,
    create_dataset,
    create_combined_dataset,
    create_dataloader,
)
from src.models import Img2LaTeX
from src.training import Trainer
from src.utils import load_config, save_plots


def main():
    parser = argparse.ArgumentParser(description="Train image-to-LaTeX model")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config file"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on"
    )
    parser.add_argument(
        "--data-fraction",
        type=float,
        default=None,
        help="Fraction of data to use (0.0-1.0)"
    )
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    data_config = config.get('data', {})
    model_config = config.get('model', {})
    train_config = config.get('training', {})
    log_config = config.get('logging', {})
    
    # Set up paths
    data_dir = Path(data_config.get('data_dir', '/Data/img-to-latex'))
    checkpoint_dir = Path(log_config.get('checkpoint_dir', 'checkpoints'))
    log_dir = Path(log_config.get('log_dir', 'logs'))
    
    # Load or build tokenizer
    vocab_path = Path(data_config.get('vocab_path', data_dir / 'vocab.json'))
    if vocab_path.exists():
        print(f"Loading tokenizer from {vocab_path}")
        tokenizer = LaTeXTokenizer.load(vocab_path)
    else:
        # Build from PRINTED_TEX_230k vocab
        base_vocab = data_dir / "PRINTED_TEX_230k" / "230k.json"
        if base_vocab.exists():
            print(f"Loading tokenizer from {base_vocab}")
            tokenizer = LaTeXTokenizer.from_printed_tex(base_vocab)
        else:
            print("Creating empty tokenizer - consider running build_vocab.py first")
            tokenizer = LaTeXTokenizer()
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create datasets
    datasets = data_config.get('datasets', ['printed_tex'])
    img_size = data_config.get('img_size', 224)
    max_seq_len = data_config.get('max_seq_len', 256)
    limit_samples = data_config.get('limit_samples', None)
    limit_fraction = args.data_fraction  # Command line arg overrides config
    
    print(f"\nLoading datasets: {datasets}")
    
    if len(datasets) == 1:
        train_dataset = create_dataset(
            datasets[0], data_dir, tokenizer,
            split='train',
            img_size=img_size,
            max_seq_len=max_seq_len,
            limit_samples=limit_samples,
            limit_fraction=limit_fraction,
        )
        val_dataset = create_dataset(
            datasets[0], data_dir, tokenizer,
            split='valid',
            img_size=img_size,
            max_seq_len=max_seq_len,
            limit_samples=limit_samples,
            limit_fraction=limit_fraction,
        )
    else:
        train_dataset = create_combined_dataset(
            datasets, data_dir, tokenizer,
            split='train',
            img_size=img_size,
            max_seq_len=max_seq_len,
            limit_samples=limit_samples,
            limit_fraction=limit_fraction,
        )
        val_dataset = create_combined_dataset(
            datasets, data_dir, tokenizer,
            split='valid',
            img_size=img_size,
            max_seq_len=max_seq_len,
            limit_samples=limit_samples,
            limit_fraction=limit_fraction,
        )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create data loaders
    batch_size = train_config.get('batch_size', 32)
    num_workers = data_config.get('num_workers', 4)
    
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    # Create model
    print("\nCreating model...")
    # Update model config with tokenizer's special tokens
    model_config.update({
        'pad_id': tokenizer.pad_id,
        'sos_id': tokenizer.sos_id,
        'eos_id': tokenizer.eos_id,
    })
    model = Img2LaTeX.from_config(model_config, vocab_size=tokenizer.vocab_size)
    
    # Load pretrained weights if specified
    checkpoint_config = config.get('checkpoint', {})
    pretrained_path = checkpoint_config.get('pretrained_path')
    if pretrained_path:
        pretrained_path = Path(pretrained_path)
        if pretrained_path.exists():
            print(f"Loading pretrained weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=args.device)
            model.load_state_dict(checkpoint['model_state_dict'])
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=train_config.get('learning_rate', 1e-4),
        weight_decay=train_config.get('weight_decay', 0.01),
        warmup_steps=train_config.get('warmup_steps', 1000),
        max_grad_norm=train_config.get('max_grad_norm', 1.0),
        accumulation_steps=train_config.get('accumulation_steps', 1),
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        pad_id=tokenizer.pad_id,
        eos_id=tokenizer.eos_id,
    )
    
    # Train
    print("\nStarting training...")
    stats = trainer.train(
        num_epochs=train_config.get('num_epochs', 50),
        eval_every=train_config.get('eval_every', 1),
        patience=train_config.get('patience', 10),
        resume_from=args.resume,
    )
    
    # Save plots
    print("\nSaving training plots...")
    save_plots(stats.to_dict(), checkpoint_dir / 'plots')
    
    print("\nTraining complete!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()

