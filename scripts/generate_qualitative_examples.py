#!/usr/bin/env python3
"""Generate qualitative examples showing model predictions vs ground truth."""

import argparse
import random
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    LaTeXTokenizer,
    create_combined_dataset,
    create_dataset,
)
from src.models import Img2LaTeX
from src.utils import load_config


def wrap_latex(text: str, max_len: int = 45) -> str:
    """Wrap long LaTeX strings for display."""
    if len(text) <= max_len:
        return text
    # Try to break at spaces or operators
    lines = []
    current = ""
    for char in text:
        current += char
        if len(current) >= max_len and char in " +=-{}":
            lines.append(current)
            current = ""
    if current:
        lines.append(current)
    return "\n".join(lines)


def generate_examples(
    model,
    dataset,
    tokenizer,
    device,
    num_examples: int = 6,
    seed: int = 42,
):
    """Generate prediction examples."""
    random.seed(seed)
    
    # Sample indices - try to get mix of correct and incorrect
    indices = random.sample(range(len(dataset)), min(num_examples * 3, len(dataset)))
    
    examples = []
    model.eval()
    
    with torch.no_grad():
        for idx in indices:
            if len(examples) >= num_examples:
                break
                
            sample = dataset[idx]
            image = sample['image'].unsqueeze(0).to(device)
            target_ids = sample['target_ids']
            gt_latex = sample['latex']
            
            # Generate prediction
            generated = model.generate(image, temperature=0)
            pred_ids = generated[0, 1:].cpu().tolist()  # Strip SOS
            pred_latex = tokenizer.decode(pred_ids)
            
            # Check if correct
            pred_norm = "".join(pred_latex.split())
            gt_norm = "".join(gt_latex.split())
            is_correct = pred_norm == gt_norm
            
            # Get original image (denormalize)
            img_np = sample['image'].squeeze().cpu().numpy()
            # Denormalize: img = img * std + mean
            img_np = img_np * 0.5 + 0.5  # Assuming normalize with mean=0.5, std=0.5
            img_np = np.clip(img_np, 0, 1)
            
            examples.append({
                'image': img_np,
                'ground_truth': gt_latex,
                'prediction': pred_latex,
                'is_correct': is_correct,
            })
    
    return examples


def create_figure(examples, output_path: Path):
    """Create a figure showing qualitative examples."""
    n = len(examples)
    
    fig = plt.figure(figsize=(14, 3 * n))
    gs = gridspec.GridSpec(n, 1, hspace=0.4)
    
    for i, ex in enumerate(examples):
        ax = fig.add_subplot(gs[i])
        
        # Show image
        ax.imshow(ex['image'], cmap='gray', aspect='auto')
        ax.axis('off')
        
        # Add text annotations
        status = "✓" if ex['is_correct'] else "✗"
        color = "green" if ex['is_correct'] else "red"
        
        gt_text = wrap_latex(ex['ground_truth'], 55)
        pred_text = wrap_latex(ex['prediction'], 55)
        
        title = f"{status} Ground Truth: {gt_text}\n   Prediction: {pred_text}"
        ax.set_title(title, fontsize=9, loc='left', color=color if not ex['is_correct'] else 'black',
                    fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Figure saved to {output_path}")


def create_compact_figure(examples, output_path: Path):
    """Create a compact figure for the report with preserved aspect ratios."""
    # Separate correct and incorrect
    correct = [e for e in examples if e['is_correct']]
    incorrect = [e for e in examples if not e['is_correct']]
    
    # Take 3 correct and 3 incorrect (or as many as available)
    selected = correct[:3] + incorrect[:3]
    if len(selected) < 6:
        selected = examples[:6]
    
    n = len(selected)
    
    # Calculate height ratios based on image aspect ratios
    fig_width = 12
    row_heights = []
    for ex in selected:
        h, w = ex['image'].shape
        img_display_height = 1.2 * (h / w) * fig_width * 0.6
        row_heights.append(max(img_display_height, 1.5))
    
    total_height = sum(row_heights) + 0.5 * n
    
    fig = plt.figure(figsize=(fig_width, total_height))
    gs = gridspec.GridSpec(n, 1, height_ratios=row_heights, hspace=0.4)
    
    for i, ex in enumerate(selected):
        ax = fig.add_subplot(gs[i])
        ax.imshow(ex['image'], cmap='gray', aspect='equal')
        ax.axis('off')
        
        gt = ex['ground_truth'][:70] + ('...' if len(ex['ground_truth']) > 70 else '')
        pred = ex['prediction'][:70] + ('...' if len(ex['prediction']) > 70 else '')
        
        status = "✓" if ex['is_correct'] else "✗"
        color = 'darkgreen' if ex['is_correct'] else 'darkred'
        ax.set_title(f"{status}  GT: {gt}\n     Pred: {pred}", 
                    fontsize=8, loc='left', fontfamily='monospace',
                    color=color)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Compact figure saved to {output_path}")


def save_individual_images(examples, output_dir: Path):
    """Save each example as an individual image and generate LaTeX code."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Separate correct and incorrect
    correct = [e for e in examples if e['is_correct']]
    incorrect = [e for e in examples if not e['is_correct']]
    
    # Take 3 correct and 3 incorrect
    selected = correct[:3] + incorrect[:3]
    if len(selected) < 6:
        selected = examples[:6]
    
    latex_lines = []
    latex_lines.append(r"\begin{figure}[h]")
    latex_lines.append(r"    \centering")
    latex_lines.append(r"    \small")
    latex_lines.append(r"    \setlength{\tabcolsep}{4pt}")
    latex_lines.append(r"    \begin{tabular}{c|c|c|c}")
    latex_lines.append(r"    \textbf{Image} & \textbf{Ground Truth} & \textbf{Prediction} & \textbf{Match} \\")
    latex_lines.append(r"    \hline")
    
    for i, ex in enumerate(selected):
        # Save image
        img_path = output_dir / f"example_{i+1}.png"
        fig, ax = plt.subplots(figsize=(6, 6 * ex['image'].shape[0] / ex['image'].shape[1]))
        ax.imshow(ex['image'], cmap='gray', aspect='equal')
        ax.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(img_path, dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none', pad_inches=0.02)
        plt.close()
        print(f"Saved {img_path}")
        
        # Clean up GT and pred for rendering
        gt = ex['ground_truth'].strip()
        pred = ex['prediction'].strip()
        
        # Truncate if too long
        if len(gt) > 50:
            gt = gt[:47] + "..."
        if len(pred) > 50:
            pred = pred[:47] + "..."
        
        # Status symbol
        if ex['is_correct']:
            status = r"\textcolor{green!60!black}{\ding{51}}"
        else:
            status = r"\textcolor{red}{\ding{55}}"
        
        # Generate table row
        latex_lines.append(f"    \\includegraphics[height=1.2cm]{{example_{i+1}.png}} & ${gt}$ & ${pred}$ & {status} \\\\")
        if i < len(selected) - 1:
            latex_lines.append(r"    \hline")
    
    latex_lines.append(r"    \end{tabular}")
    latex_lines.append(r"    \caption{\textbf{Qualitative Examples.} Sample predictions from the handwritten validation set showing both successful matches (\textcolor{green!60!black}{\ding{51}}) and failure cases (\textcolor{red}{\ding{55}}).}")
    latex_lines.append(r"    \label{fig:qualitative}")
    latex_lines.append(r"\end{figure}")
    
    # Save LaTeX snippet
    latex_path = output_dir / "qualitative_latex.tex"
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    print(f"\nLaTeX snippet saved to {latex_path}")
    
    # Also print for easy copy-paste
    print("\n" + "="*60)
    print("LaTeX code to add to report:")
    print("="*60)
    print('\n'.join(latex_lines))
    
    return selected


def main():
    parser = argparse.ArgumentParser(description="Generate qualitative examples")
    parser.add_argument("--config", type=Path, default=Path("configs/finetune.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/finetune/latest.pt"))
    parser.add_argument("--output", type=Path, default=Path("report/Plots/qualitative_examples.png"))
    parser.add_argument("--num-examples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    data_config = config.get('data', {})
    model_config = config.get('model', {})
    
    # Paths
    data_dir = Path(data_config.get('data_dir', '/Data/img-to-latex'))
    vocab_path = Path(data_config.get('vocab_path', data_dir / 'vocab.json'))
    
    # Load tokenizer
    tokenizer = LaTeXTokenizer.load(vocab_path)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Load dataset (validation split for examples)
    datasets = data_config.get('datasets', ['crohme', 'mathwriting', 'hme100k'])
    img_size = data_config.get('img_size', 224)
    max_seq_len = data_config.get('max_seq_len', 256)
    
    print(f"Loading validation data from: {datasets}")
    dataset = create_combined_dataset(
        datasets, data_dir, tokenizer,
        split='valid',
        img_size=img_size,
        max_seq_len=max_seq_len,
    )
    print(f"Dataset size: {len(dataset)}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
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
    
    # Generate examples
    print(f"\nGenerating {args.num_examples} examples...")
    examples = generate_examples(
        model, dataset, tokenizer, args.device,
        num_examples=args.num_examples,
        seed=args.seed,
    )
    
    # Count correct/incorrect
    n_correct = sum(1 for e in examples if e['is_correct'])
    print(f"Examples: {n_correct}/{len(examples)} correct")
    
    # Save individual images and generate LaTeX
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_individual_images(examples, args.output.parent)
    
    # Also print examples to console
    print("\n" + "="*60)
    for i, ex in enumerate(examples):
        status = "✓ CORRECT" if ex['is_correct'] else "✗ WRONG"
        print(f"\nExample {i+1} [{status}]")
        print(f"  GT:   {ex['ground_truth'][:80]}")
        print(f"  Pred: {ex['prediction'][:80]}")


if __name__ == "__main__":
    main()
