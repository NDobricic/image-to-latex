#!/usr/bin/env python3
"""Find correctly decoded samples from the dataset and save them to samples folder."""

import random
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import LaTeXTokenizer, create_combined_dataset
from src.models import Img2LaTeX
from src.utils import load_config


def main():
    config_path = Path("configs/finetune.yaml")
    checkpoint_path = Path("checkpoints/finetune/latest.pt")
    output_dir = Path("samples")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_samples = 10
    seed = 456
    
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    print(f"Loading config from {config_path}")
    config = load_config(config_path)
    data_config = config.get('data', {})
    model_config = config.get('model', {})
    
    # Paths
    data_dir = Path(data_config.get('data_dir', '/Data/img-to-latex'))
    vocab_path = Path(data_config.get('vocab_path', data_dir / 'vocab.json'))
    
    # Load tokenizer
    tokenizer = LaTeXTokenizer.load(vocab_path)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Load dataset
    datasets = data_config.get('datasets', ['crohme', 'mathwriting', 'hme100k'])
    img_size = data_config.get('img_size', 224)
    max_seq_len = data_config.get('max_seq_len', 256)
    
    print(f"Loading validation data...")
    dataset = create_combined_dataset(
        datasets, data_dir, tokenizer,
        split='valid',
        img_size=img_size,
        max_seq_len=max_seq_len,
    )
    print(f"Dataset size: {len(dataset)}")
    
    # Load model
    print(f"Loading model from {checkpoint_path}")
    model_config.update({
        'pad_id': tokenizer.pad_id,
        'sos_id': tokenizer.sos_id,
        'eos_id': tokenizer.eos_id,
    })
    model = Img2LaTeX.from_config(model_config, vocab_size=tokenizer.vocab_size)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Find correct samples
    print(f"\nSearching for {num_samples} correctly decoded samples...")
    indices = random.sample(range(len(dataset)), min(500, len(dataset)))
    
    correct_samples = []
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            if len(correct_samples) >= num_samples:
                break
            
            sample = dataset[idx]
            image = sample['image'].unsqueeze(0).to(device)
            gt_latex = sample['latex']
            orig_path = sample['image_path']
            
            # Generate prediction
            generated = model.generate(image, temperature=0)
            pred_ids = generated[0, 1:].cpu().tolist()
            pred_latex = tokenizer.decode(pred_ids)
            
            # Check if correct (normalize whitespace)
            pred_norm = "".join(pred_latex.split())
            gt_norm = "".join(gt_latex.split())
            
            if pred_norm == gt_norm:
                correct_samples.append({
                    'idx': idx,
                    'orig_path': orig_path,
                    'latex': gt_latex,
                })
                print(f"  Found {len(correct_samples)}/{num_samples}: {gt_latex[:60]}...")
            
            if (i + 1) % 50 == 0:
                print(f"  Checked {i+1}/{len(indices)} samples...")
    
    # Copy original images
    print(f"\nCopying {len(correct_samples)} original images to {output_dir}/")
    
    info_lines = []
    for i, sample in enumerate(correct_samples):
        orig_path = Path(sample['orig_path'])
        ext = orig_path.suffix
        
        # Create filename from LaTeX (sanitized)
        latex_clean = sample['latex'].replace(' ', '_').replace('\\', '')[:30]
        latex_clean = ''.join(c for c in latex_clean if c.isalnum() or c in '_-+')
        filename = f"sample_{i+1}_{latex_clean}{ext}"
        
        dest_path = output_dir / filename
        shutil.copy2(orig_path, dest_path)
        info_lines.append(f"{filename}: {sample['latex']}")
        print(f"  Copied: {filename}")
    
    # Save info file
    info_path = output_dir / "samples_info.txt"
    with open(info_path, 'w') as f:
        f.write('\n'.join(info_lines))
    print(f"\nInfo saved to {info_path}")


if __name__ == "__main__":
    main()
