#!/usr/bin/env python3
"""
Generate a detailed Excel report with images, predictions, and metrics.
"""

import argparse
import sys
import io
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    LaTeXTokenizer,
    create_dataset,
    create_combined_dataset,
    create_dataloader,
)
from src.models import Img2LaTeX
from src.training.metrics import compute_token_accuracy, compute_sequence_accuracy, compute_bleu
from src.utils import load_config


def tensor_to_image(tensor):
    """Convert a normalized tensor to a PIL Image."""
    # Denormalize assuming mean=0.5, std=0.5 (standard for this repo?)
    # Or just simple rescaling if it was 0-1.
    # Checking base_dataset.py would confirm transforms, but standard 
    # ToTensor gives 0-1. If no Normalize was used, just * 255.
    # Assuming 0-1 range for now as per common practice if no explicit norm mentioned.
    
    img_np = tensor.cpu().numpy().squeeze()  # (H, W) for grayscale
    img_np = (img_np * 255).astype(np.uint8)
    return Image.fromarray(img_np, mode='L')


def calculate_single_metrics(pred_ids, gt_ids, tokenizer):
    """Calculate metrics for a single sample."""
    # Wrap in list because metric functions expect list of lists
    token_acc = compute_token_accuracy([pred_ids], [gt_ids], tokenizer.pad_id)
    seq_acc = compute_sequence_accuracy([pred_ids], [gt_ids], tokenizer.pad_id, tokenizer.eos_id)
    bleu = compute_bleu([pred_ids], [gt_ids], pad_id=tokenizer.pad_id, eos_id=tokenizer.eos_id)
    
    # Semantic accuracy
    pred_str = tokenizer.decode(pred_ids)
    gt_str = tokenizer.decode(gt_ids)
    
    pred_norm = "".join(pred_str.split())
    gt_norm = "".join(gt_str.split())
    semantic_acc = 1.0 if pred_norm == gt_norm else 0.0
    
    return {
        "Token Acc": token_acc,
        "Seq Acc": seq_acc,
        "Semantic Acc": semantic_acc,
        "BLEU": bleu
    }


def main():
    parser = argparse.ArgumentParser(description="Generate Excel report with images and predictions")
    parser.add_argument("--config", type=Path, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Output .xlsx file path")
    parser.add_argument("--split", type=str, default="test", help="Data split to use")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to include")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--beam-size", type=int, default=None, help="Beam size for decoding")
    args = parser.parse_args()

    # Load config
    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    data_config = config.get('data', {})
    model_config = config.get('model', {})

    # Setup paths
    data_dir = Path(data_config.get('data_dir', '/Data/img-to-latex'))
    vocab_path = Path(data_config.get('vocab_path', data_dir / 'vocab.json'))

    # Load tokenizer
    if vocab_path.exists():
        tokenizer = LaTeXTokenizer.load(vocab_path)
    else:
        print("Error: Vocab file not found.")
        return

    # Create dataset
    datasets = data_config.get('datasets', ['printed_tex'])
    img_size = data_config.get('img_size', 224)
    max_seq_len = data_config.get('max_seq_len', 256)
    
    print(f"Loading {args.split} data from {datasets}...")
    if len(datasets) == 1:
        dataset = create_dataset(datasets[0], data_dir, tokenizer, split=args.split, img_size=img_size, max_seq_len=max_seq_len)
    else:
        dataset = create_combined_dataset(datasets, data_dir, tokenizer, split=args.split, img_size=img_size, max_seq_len=max_seq_len)

    # Subset
    if args.num_samples < len(dataset):
        indices = np.random.choice(len(dataset), args.num_samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)
    
    loader = create_dataloader(dataset, batch_size=1, shuffle=False) # Batch size 1 to handle images easily

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

    # Lists to store data
    records = []
    images_input = [] # Transformed tensor images
    images_original = [] # Original file images

    print("Running inference...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            image_tensor = batch['images'].to(args.device)
            target_ids = batch['target_ids'].squeeze(0).tolist()
            gt_latex = batch['latex'][0]
            img_path = batch['image_paths'][0]
            
            # Generate
            if args.beam_size and args.beam_size > 1:
                generated = model.generate(image_tensor, beam_size=args.beam_size)
            else:
                generated = model.generate(image_tensor, temperature=0)
            
            pred_ids = generated.squeeze(0).tolist()[1:] # Strip SOS
            pred_latex = tokenizer.decode(pred_ids)

            # Calculate metrics
            metrics = calculate_single_metrics(pred_ids, target_ids, tokenizer)
            
            # 1. Get transformed image
            pil_img_input = tensor_to_image(batch['images'][0])
            images_input.append(pil_img_input)
            
            # 2. Get original image
            try:
                pil_img_orig = Image.open(img_path).convert('RGB') # RGB for display
                images_original.append(pil_img_orig)
            except Exception:
                images_original.append(pil_img_input) # Fallback

            # Store record
            record = {
                "Image ID": i,
                "Ground Truth": gt_latex,
                "Prediction": pred_latex,
                **metrics
            }
            records.append(record)

    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Reorder columns
    cols = ["Image ID", "Ground Truth", "Prediction", "Semantic Acc", "BLEU", "Token Acc", "Seq Acc"]
    df = df[cols]

    # Escape strings starting with = to prevent Excel from treating them as formulas
    # Also handle + or - which can sometimes trigger formula mode in some excel versions if strictly interpreted
    def escape_excel_formula(val):
        if isinstance(val, str) and (val.startswith('=') or val.startswith('+') or val.startswith('-')):
            return "'" + val
        return val

    df['Ground Truth'] = df['Ground Truth'].apply(escape_excel_formula)
    df['Prediction'] = df['Prediction'].apply(escape_excel_formula)

    print(f"Saving report to {args.output}...")
    
    # Save to Excel with images
    # We use XlsxWriter engine to insert images
    with pd.ExcelWriter(args.output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Predictions', index=False)
        
        workbook = writer.book
        worksheet = writer.sheets['Predictions']
        
        # Format text wrap for long LaTeX strings
        wrap_format = workbook.add_format({'text_wrap': True, 'valign': 'top'})
        worksheet.set_column('B:C', 50, wrap_format) # GT and Pred columns width 50
        
        # Insert images columns
        worksheet.set_column('H:H', 30) # Original Image column width
        worksheet.set_column('I:I', 30) # Input Tensor Image column width
        worksheet.write('H1', 'Original Image') # Header
        worksheet.write('I1', 'Model Input') # Header
        
        print("Inserting images into Excel...")
        for i, (img_orig, img_input) in enumerate(zip(images_original, images_input)):
            row_idx = i + 1
            worksheet.set_row(row_idx, 100)
            
            # 1. Original Image (Column H)
            try:
                img_orig_rgb = img_orig.convert('RGB')
                img_orig_rgb.thumbnail((200, 200))
                byte_arr_orig = io.BytesIO()
                img_orig_rgb.save(byte_arr_orig, format='PNG')
                worksheet.insert_image(row_idx, 7, 'orig.png', {
                    'image_data': byte_arr_orig,
                    'x_scale': 0.5, 'y_scale': 0.5, 'object_position': 1
                })
            except Exception as e:
                print(f"Warning: Failed to save original image for row {row_idx}: {e}")
            
            # 2. Model Input (Column I)
            try:
                img_input_rgb = img_input.convert('RGB')
                img_input_rgb.thumbnail((200, 200))
                byte_arr_input = io.BytesIO()
                img_input_rgb.save(byte_arr_input, format='PNG')
                worksheet.insert_image(row_idx, 8, 'input.png', {
                    'image_data': byte_arr_input,
                    'x_scale': 0.5, 'y_scale': 0.5, 'object_position': 1
                })
            except Exception as e:
                print(f"Warning: Failed to save input image for row {row_idx}: {e}")


    print("Done!")


if __name__ == "__main__":
    main()

