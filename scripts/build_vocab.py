#!/usr/bin/env python3
"""Build unified vocabulary from all datasets."""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.tokenizer import LaTeXTokenizer


def load_printed_tex_labels(data_dir: Path) -> list[str]:
    """Load labels from PRINTED_TEX_230k dataset."""
    labels_path = data_dir / "PRINTED_TEX_230k" / "final_png_formulas.txt"
    if not labels_path.exists():
        print(f"  [SKIP] {labels_path} not found")
        return []
    
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f if line.strip()]
    
    print(f"  [OK] PRINTED_TEX_230k: {len(labels)} labels")
    return labels


def load_hme100k_labels(data_dir: Path) -> list[str]:
    """Load labels from HME100K dataset."""
    labels = []
    for split in ["train", "test"]:
        labels_path = data_dir / "HME100K" / f"{split}_labels.txt"
        if not labels_path.exists():
            continue
        
        with open(labels_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '\t' in line:
                    _, label = line.split('\t', 1)
                    labels.append(label)
    
    print(f"  [OK] HME100K: {len(labels)} labels")
    return labels


def load_mathwriting_labels(data_dir: Path) -> list[str]:
    """Load labels from MathWriting dataset (from converted labels.txt files)."""
    labels = []
    mw_converted = data_dir / "converted" / "mathwriting"
    
    if not mw_converted.exists():
        # Try reading directly from InkML if not converted yet
        import xml.etree.ElementTree as ET
        mw_dir = data_dir / "mathwriting-2024"
        for split in ["train", "valid", "test"]:
            split_dir = mw_dir / split
            if not split_dir.exists():
                continue
            for inkml_path in split_dir.glob("*.inkml"):
                try:
                    tree = ET.parse(inkml_path)
                    root = tree.getroot()
                    ns = {'inkml': 'http://www.w3.org/2003/InkML'}
                    for ann in root.findall('.//inkml:annotation', ns):
                        if ann.get('type') == 'normalizedLabel' and ann.text:
                            labels.append(ann.text.strip())
                            break
                except:
                    continue
        print(f"  [OK] MathWriting (from InkML): {len(labels)} labels")
    else:
        for split_dir in mw_converted.iterdir():
            if split_dir.is_dir():
                labels_file = split_dir / "labels.txt"
                if labels_file.exists():
                    with open(labels_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if '\t' in line:
                                _, label = line.strip().split('\t', 1)
                                labels.append(label)
        print(f"  [OK] MathWriting (converted): {len(labels)} labels")
    
    return labels


def load_crohme_labels(data_dir: Path) -> list[str]:
    """Load labels from CROHME dataset."""
    labels = []
    crohme_converted = data_dir / "converted" / "crohme"
    
    if crohme_converted.exists():
        for split_dir in crohme_converted.iterdir():
            if split_dir.is_dir():
                labels_file = split_dir / "labels.txt"
                if labels_file.exists():
                    with open(labels_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if '\t' in line:
                                _, label = line.strip().split('\t', 1)
                                labels.append(label)
    
    print(f"  [OK] CROHME: {len(labels)} labels")
    return labels


def main():
    parser = argparse.ArgumentParser(description="Build unified vocabulary")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/Data/img-to-latex"),
        help="Base data directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output vocabulary file"
    )
    parser.add_argument(
        "--base-vocab",
        type=Path,
        default=None,
        help="Base vocabulary to extend (e.g., 230k.json)"
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=2,
        help="Minimum token frequency to include"
    )
    args = parser.parse_args()
    
    output_path = args.output or (args.data_dir / "vocab.json")
    
    # Load base vocabulary if provided
    if args.base_vocab:
        print(f"Loading base vocabulary from {args.base_vocab}")
        tokenizer = LaTeXTokenizer.from_printed_tex(args.base_vocab)
        print(f"  Base vocab size: {tokenizer.vocab_size}")
    else:
        tokenizer = LaTeXTokenizer()
    
    # Collect all labels
    print("\nLoading labels from datasets...")
    all_labels = []
    all_labels.extend(load_printed_tex_labels(args.data_dir))
    all_labels.extend(load_hme100k_labels(args.data_dir))
    all_labels.extend(load_mathwriting_labels(args.data_dir))
    all_labels.extend(load_crohme_labels(args.data_dir))
    
    print(f"\nTotal labels: {len(all_labels)}")
    
    # Count token frequencies
    token_counts = Counter()
    for label in all_labels:
        tokens = tokenizer.tokenize(label)
        token_counts.update(tokens)
    
    print(f"Unique tokens found: {len(token_counts)}")
    
    # Add tokens meeting frequency threshold
    added = 0
    for token, count in token_counts.items():
        if count >= args.min_freq and token not in tokenizer.token2id:
            tokenizer.add_token(token)
            added += 1
    
    print(f"Added {added} new tokens (min_freq={args.min_freq})")
    print(f"Final vocab size: {tokenizer.vocab_size}")
    
    # Show some statistics
    print("\nMost common tokens:")
    for token, count in token_counts.most_common(20):
        print(f"  {token!r}: {count}")
    
    # Save vocabulary
    tokenizer.save(output_path)
    print(f"\nVocabulary saved to {output_path}")


if __name__ == "__main__":
    main()

