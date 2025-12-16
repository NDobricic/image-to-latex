#!/usr/bin/env python3
"""
Interactive demo script for the Img2LaTeX model.
Loads the model first, then prompts the user to input image paths for LaTeX conversion.
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import LaTeXTokenizer
from src.models import Img2LaTeX
from src.utils import load_config


def load_model(config_path: Path, checkpoint_path: Path, device: str):
    """Load the model and tokenizer."""
    print("="*60)
    print("  Loading Img2LaTeX Model...")
    print("="*60)
    
    print(f"\n[1/4] Loading config from {config_path}")
    config = load_config(config_path)
    
    data_config = config.get('data', {})
    model_config = config.get('model', {})
    
    # Load tokenizer
    print("[2/4] Loading tokenizer...")
    data_dir = Path(data_config.get('data_dir', '/Data/img-to-latex'))
    vocab_path = Path(data_config.get('vocab_path', data_dir / 'vocab.json'))
    tokenizer = LaTeXTokenizer.load(vocab_path)
    print(f"       Vocabulary size: {tokenizer.vocab_size}")
    
    # Create model
    print("[3/4] Building model architecture...")
    model_config.update({
        'pad_id': tokenizer.pad_id,
        'sos_id': tokenizer.sos_id,
        'eos_id': tokenizer.eos_id,
    })
    model = Img2LaTeX.from_config(model_config, vocab_size=tokenizer.vocab_size)
    
    # Load checkpoint
    print(f"[4/4] Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"\n✓ Model loaded successfully on {device.upper()}")
    return model, tokenizer, data_config


def preprocess_image(image_path: str, img_size: int = 224) -> torch.Tensor:
    """Load and preprocess an image for the model."""
    # Load image
    image = Image.open(image_path)
    
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    
    # Apply transforms
    image_tensor = transform(image)  # Shape: (1, H, W)
    
    # Ensure correct shape (B, C, H, W)
    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)  # Add channel dim
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dim
    
    return image_tensor


def predict(model, tokenizer, image_tensor, device, beam_size=5, max_len=256):
    """Run inference on an image."""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        try:
            if beam_size > 1:
                # Beam search (returns (1, L) tensor)
                generated = model.generate(image_tensor, beam_size=beam_size, max_len=max_len)
            else:
                # Greedy decoding
                generated = model.generate(image_tensor, temperature=0, max_len=max_len)
        except Exception as e:
            # Fall back to greedy if beam search fails
            print(f"  Warning: Beam search failed ({e}), using greedy decoding...")
            generated = model.generate(image_tensor, temperature=0, max_len=max_len)
    
    # Handle different output shapes
    if generated.dim() == 1:
        tokens = generated[1:].cpu().tolist()  # Skip SOS token
    else:
        tokens = generated[0, 1:].cpu().tolist()  # Skip SOS token
    
    # Remove padding and EOS tokens
    clean_tokens = []
    for t in tokens:
        if t == tokenizer.eos_id or t == tokenizer.pad_id:
            break
        clean_tokens.append(t)
    
    latex = tokenizer.decode(clean_tokens)
    
    return latex


def interactive_mode(model, tokenizer, data_config, device, beam_size=5):
    """Run interactive demo loop."""
    img_size = data_config.get('img_size', 224)
    max_seq_len = data_config.get('max_seq_len', 256)
    
    print("\n" + "="*60)
    print("  ✓ MODEL READY - Img2LaTeX Interactive Demo")
    print("="*60)
    print("\nThe model is loaded and ready to process images!")
    print("\nCommands:")
    print("  - Enter an image path to convert to LaTeX")
    print("  - Type 'quit' or 'exit' to stop")
    decoding_mode = "greedy" if beam_size <= 1 else f"beam search (k={beam_size})"
    print(f"  - Type 'beam N' to change decoding (current: {decoding_mode})")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input(">>> Enter image path: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            # Check for beam size command
            if user_input.lower().startswith('beam '):
                try:
                    beam_size = int(user_input.split()[1])
                    print(f"Beam size set to {beam_size}")
                except (ValueError, IndexError):
                    print("Usage: beam N (e.g., 'beam 5')")
                continue
            
            # Skip empty input
            if not user_input:
                continue
            
            # Check if file exists
            image_path = Path(user_input)
            if not image_path.exists():
                print(f"Error: File not found: {image_path}")
                continue
            
            # Process image
            print(f"Processing: {image_path}")
            try:
                image_tensor = preprocess_image(str(image_path), img_size)
                latex = predict(model, tokenizer, image_tensor, device, 
                               beam_size=beam_size, max_len=max_seq_len)
                
                print("\n" + "-"*40)
                print("LaTeX Output:")
                print(f"  {latex}")
                print("-"*40)
                print(f"Rendered: ${latex}$")
                print("-"*40 + "\n")
                
            except Exception as e:
                print(f"Error processing image: {e}")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def single_image_mode(model, tokenizer, data_config, device, image_path: str, beam_size=5):
    """Process a single image and print the result."""
    img_size = data_config.get('img_size', 224)
    max_seq_len = data_config.get('max_seq_len', 256)
    
    image_tensor = preprocess_image(image_path, img_size)
    latex = predict(model, tokenizer, image_tensor, device,
                   beam_size=beam_size, max_len=max_seq_len)
    
    print(latex)
    return latex


def main():
    parser = argparse.ArgumentParser(
        description="Img2LaTeX Demo - Convert handwritten math to LaTeX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default)
  python demo.py
  
  # Single image (non-interactive)
  python demo.py --image path/to/image.png
  
  # With custom checkpoint
  python demo.py --checkpoint checkpoints/finetune/best.pt
        """
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/finetune.yaml"),
        help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/finetune/latest.pt"),
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single image (skips interactive mode)"
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for decoding (default: 5)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    args = parser.parse_args()
    
    # ========================================
    # STEP 1: Load model first
    # ========================================
    model, tokenizer, data_config = load_model(
        args.config, args.checkpoint, args.device
    )
    
    # ========================================
    # STEP 2: Now prompt user for images
    # ========================================
    if args.image:
        # Single image mode (non-interactive)
        single_image_mode(model, tokenizer, data_config, args.device, 
                         args.image, args.beam_size)
    else:
        # Interactive mode - prompt user for images
        interactive_mode(model, tokenizer, data_config, args.device, 
                        args.beam_size)


if __name__ == "__main__":
    main()
