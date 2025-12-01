#!/usr/bin/env python3
"""Debug script to inspect model generation outputs."""

import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import LaTeXTokenizer, create_dataset, create_dataloader
from src.models import Img2LaTeX
from src.utils import load_config

def debug_generation():
    # Config
    config_path = Path("configs/pretrain.yaml")
    checkpoint_path = Path("checkpoints/pretrain/latest.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading config from {config_path}")
    config = load_config(config_path)
    
    # Load tokenizer
    vocab_path = Path(config['data']['vocab_path'])
    tokenizer = LaTeXTokenizer.load(vocab_path)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Special tokens: PAD={tokenizer.pad_id}, SOS={tokenizer.sos_id}, EOS={tokenizer.eos_id}")
    
    # Create dataset (small subset)
    print("Loading validation dataset...")
    dataset = create_dataset(
        'printed_tex',
        Path(config['data']['data_dir']),
        tokenizer,
        split='valid',
        limit_samples=10
    )
    
    # Load model
    print("Loading model...")
    model = Img2LaTeX.from_config(config['model'], vocab_size=tokenizer.vocab_size)
    
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Warning: No checkpoint found! Using random weights.")
    
    model = model.to(device)
    model.eval()
    
    # Get one sample
    loader = create_dataloader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(loader))
    
    images = batch['images'].to(device)
    latex_gt = batch['latex'][0]
    
    print("\n--- Debugging Generation ---")
    print(f"Ground Truth: {latex_gt}")
    
    # 1. Check Encoder Output
    with torch.no_grad():
        memory, _ = model.encoder(images)
        print(f"Encoder memory shape: {memory.shape}")
        print(f"Encoder memory stats: min={memory.min():.4f}, max={memory.max():.4f}, mean={memory.mean():.4f}")
    
    # 2. Run Generation Step-by-Step (Simulating Greedy Search)
    print("\n--- Step-by-Step Generation ---")
    curr_seq = torch.tensor([[tokenizer.sos_id]], device=device, dtype=torch.long)
    
    for i in range(20):
        with torch.no_grad():
            logits = model.decoder(curr_seq, memory)
            next_token_logits = logits[:, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            token_id = next_token.item()
            token_str = tokenizer.id2token.get(token_id, "???")
            prob = probs[0, token_id].item()
            
            print(f"Step {i}: Predicted ID={token_id} ({token_str!r}) Prob={prob:.4f}")
            
            curr_seq = torch.cat([curr_seq, next_token.unsqueeze(0)], dim=1)
            
            if token_id == tokenizer.eos_id:
                print("EOS reached.")
                break
    
    final_pred = tokenizer.decode(curr_seq[0].tolist())
    print(f"\nFinal Prediction string: {final_pred!r}")

if __name__ == "__main__":
    debug_generation()

