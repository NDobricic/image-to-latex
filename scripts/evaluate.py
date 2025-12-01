#!/usr/bin/env python3
"""Evaluation script for image-to-LaTeX models."""

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    LaTeXTokenizer,
    create_dataset,
    create_combined_dataset,
    create_dataloader,
)
from src.models import Img2LaTeX
from src.training.metrics import compute_metrics
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Evaluate image-to-LaTeX model")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Data split to evaluate on"
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=None,
        help="Beam size for decoding (None for greedy)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save predictions"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to evaluate on"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (None for all)"
    )
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    data_config = config.get('data', {})
    model_config = config.get('model', {})
    
    # Set up paths
    data_dir = Path(data_config.get('data_dir', '/Data/img-to-latex'))
    
    # Load tokenizer
    vocab_path = Path(data_config.get('vocab_path', data_dir / 'vocab.json'))
    if vocab_path.exists():
        tokenizer = LaTeXTokenizer.load(vocab_path)
    else:
        base_vocab = data_dir / "PRINTED_TEX_230k" / "230k.json"
        tokenizer = LaTeXTokenizer.from_printed_tex(base_vocab)
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create dataset
    datasets = data_config.get('datasets', ['printed_tex'])
    img_size = data_config.get('img_size', 224)
    max_seq_len = data_config.get('max_seq_len', 256)
    
    print(f"\nLoading {args.split} split from: {datasets}")
    
    if len(datasets) == 1:
        test_dataset = create_dataset(
            datasets[0], data_dir, tokenizer,
            split=args.split,
            img_size=img_size,
            max_seq_len=max_seq_len,
        )
    else:
        test_dataset = create_combined_dataset(
            datasets, data_dir, tokenizer,
            split=args.split,
            img_size=img_size,
            max_seq_len=max_seq_len,
        )
    
    # Limit samples if specified
    if args.num_samples and args.num_samples < len(test_dataset):
        from torch.utils.data import Subset
        indices = list(range(args.num_samples))
        test_dataset = Subset(test_dataset, indices)
    
    print(f"Evaluation samples: {len(test_dataset)}")
    
    # Create data loader
    test_loader = create_dataloader(
        test_dataset,
        batch_size=1 if args.beam_size else 16,  # Beam search requires batch size 1
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
    )
    
    # Create and load model
    print(f"\nLoading model from {args.checkpoint}")
    # Update model config with tokenizer's special tokens
    model_config.update({
        'pad_id': tokenizer.pad_id,
        'sos_id': tokenizer.sos_id,
        'eos_id': tokenizer.eos_id,
    })
    model = Img2LaTeX.from_config(model_config, vocab_size=tokenizer.vocab_size)
    
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()
    
    # Evaluate
    print("\nEvaluating...")
    all_predictions = []
    all_targets = []
    all_latex_pred = []
    all_latex_gt = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['images'].to(args.device)
            target_ids = batch['target_ids']
            latex_gt = batch['latex']
            
            # Generate predictions
            if args.beam_size and args.beam_size > 1:
                generated = model.generate(
                    images,
                    beam_size=args.beam_size,
                )
            else:
                generated = model.generate(images, temperature=0)
            
            # DEBUG: Print first prediction raw tokens
            if len(all_predictions) == 0:
                print(f"\nDEBUG: Raw generated[0]: {generated[0].tolist()}")
                print(f"DEBUG: Decoded: {tokenizer.decode(generated[0].tolist())!r}")
            
            # Strip SOS token
            generated = generated[:, 1:]
            
            # Collect results
            all_predictions.extend(generated.cpu().tolist())
            all_targets.extend(target_ids.tolist())
            
            # Decode to LaTeX
            for pred, gt in zip(generated.cpu().tolist(), latex_gt):
                pred_latex = tokenizer.decode(pred)
                all_latex_pred.append(pred_latex)
                all_latex_gt.append(gt)
    
    # Compute metrics
    metrics = compute_metrics(
        all_predictions,
        all_targets,
        pad_id=tokenizer.pad_id,
        eos_id=tokenizer.eos_id,
    )
    
    print("\nResults:")
    print(f"  Token Accuracy:    {metrics['token_accuracy']:.4f}")
    print(f"  Sequence Accuracy: {metrics['sequence_accuracy']:.4f}")
    print(f"  BLEU Score:        {metrics['bleu']:.2f}")
    
    # Save predictions if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write("prediction\tground_truth\n")
            for pred, gt in zip(all_latex_pred, all_latex_gt):
                f.write(f"{pred}\t{gt}\n")
        print(f"\nPredictions saved to {args.output}")
    
    # Show some examples
    print("\nExamples (prediction | ground truth):")
    for i in range(min(5, len(all_latex_pred))):
        print(f"Pred: {all_latex_pred[i][:60]}")
        print(f"True: {all_latex_gt[i][:60]}")
        print("-" * 40)


if __name__ == "__main__":
    main()

