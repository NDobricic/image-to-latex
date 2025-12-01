#!/usr/bin/env python3
"""Convert InkML files (online handwriting) to PNG images (offline)."""

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
from multiprocessing import Pool, cpu_count

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


def parse_inkml(inkml_path: Path) -> tuple[list[list[tuple[float, float]]], Optional[str]]:
    """
    Parse an InkML file and extract strokes and label.
    
    Returns:
        strokes: List of strokes, each stroke is a list of (x, y) points
        label: The LaTeX label (normalizedLabel or truth annotation)
    """
    tree = ET.parse(inkml_path)
    root = tree.getroot()
    
    ns = {'inkml': 'http://www.w3.org/2003/InkML'}
    
    # Extract label - try normalizedLabel first (mathwriting), then truth (CROHME)
    label = None
    for annotation in root.findall('.//inkml:annotation', ns):
        ann_type = annotation.get('type', '')
        if ann_type == 'normalizedLabel':
            label = annotation.text
            break
        elif ann_type == 'truth' and label is None:
            label = annotation.text
            if label:
                label = label.strip().strip('$').strip()
    
    # Fallback: try without namespace
    if label is None:
        for annotation in root.findall('.//annotation'):
            ann_type = annotation.get('type', '')
            if ann_type == 'normalizedLabel':
                label = annotation.text
                break
            elif ann_type == 'truth' and label is None:
                label = annotation.text
                if label:
                    label = label.strip().strip('$').strip()
    
    # Extract traces (strokes)
    strokes = []
    
    # Try with namespace
    traces = root.findall('.//inkml:trace', ns)
    if not traces:
        traces = root.findall('.//trace')
    
    for trace in traces:
        if trace.text is None:
            continue
        
        points = []
        # Parse trace data - format varies:
        # "x y, x y, ..." or "x y t, x y t, ..."
        trace_text = trace.text.strip()
        point_strs = trace_text.split(',')
        
        for point_str in point_strs:
            coords = point_str.strip().split()
            if len(coords) >= 2:
                try:
                    x, y = float(coords[0]), float(coords[1])
                    points.append((x, y))
                except ValueError:
                    continue
        
        if points:
            strokes.append(points)
    
    return strokes, label


def render_strokes_to_image(
    strokes: list[list[tuple[float, float]]],
    img_size: int = 256,
    line_width: int = 2,
    padding: int = 10,
    bg_color: int = 255,
    stroke_color: int = 0
) -> Image.Image:
    """Render strokes to a PIL Image."""
    if not strokes or all(len(s) == 0 for s in strokes):
        # Return blank image if no strokes
        return Image.new('L', (img_size, img_size), bg_color)
    
    # Collect all points to find bounding box
    all_x = []
    all_y = []
    for stroke in strokes:
        for x, y in stroke:
            all_x.append(x)
            all_y.append(y)
    
    if not all_x:
        return Image.new('L', (img_size, img_size), bg_color)
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    # Handle edge case of single point or zero range
    width = max_x - min_x
    height = max_y - min_y
    if width < 1e-6:
        width = 1
    if height < 1e-6:
        height = 1
    
    # Calculate scale to fit in image with padding
    available_size = img_size - 2 * padding
    scale = min(available_size / width, available_size / height)
    
    # Create image
    img = Image.new('L', (img_size, img_size), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Transform and draw strokes
    for stroke in strokes:
        if len(stroke) < 2:
            if len(stroke) == 1:
                # Draw a single point as a small circle
                x, y = stroke[0]
                x_t = (x - min_x) * scale + padding
                y_t = (y - min_y) * scale + padding
                draw.ellipse([x_t - 1, y_t - 1, x_t + 1, y_t + 1], fill=stroke_color)
            continue
        
        # Transform points
        transformed = []
        for x, y in stroke:
            x_t = (x - min_x) * scale + padding
            y_t = (y - min_y) * scale + padding
            transformed.append((x_t, y_t))
        
        # Draw lines connecting consecutive points
        draw.line(transformed, fill=stroke_color, width=line_width)
    
    return img


def process_inkml_file(args: tuple) -> Optional[tuple[str, str]]:
    """Process a single InkML file. Returns (image_id, label) or None on failure."""
    inkml_path, output_dir, img_size, line_width = args
    
    try:
        strokes, label = parse_inkml(inkml_path)
        
        if not strokes:
            return None
        
        img = render_strokes_to_image(strokes, img_size=img_size, line_width=line_width)
        
        # Save image
        img_id = inkml_path.stem
        output_path = output_dir / f"{img_id}.png"
        img.save(output_path)
        
        return (img_id, label if label else "")
    
    except Exception as e:
        return None


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    img_size: int = 256,
    line_width: int = 2,
    num_workers: int = None
) -> dict[str, str]:
    """
    Process all InkML files in a directory.
    
    Returns:
        Dictionary mapping image IDs to labels
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    inkml_files = list(input_dir.rglob("*.inkml"))
    if not inkml_files:
        print(f"No InkML files found in {input_dir}")
        return {}
    
    num_workers = num_workers or max(1, cpu_count() - 1)
    
    args_list = [(f, output_dir, img_size, line_width) for f in inkml_files]
    
    labels = {}
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_inkml_file, args_list),
            total=len(inkml_files),
            desc=f"Converting {input_dir.name}"
        ))
    
    for result in results:
        if result is not None:
            img_id, label = result
            labels[img_id] = label
    
    return labels


def save_labels(labels: dict[str, str], output_path: Path):
    """Save labels to a TSV file (image_id TAB label)."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for img_id, label in sorted(labels.items()):
            f.write(f"{img_id}.png\t{label}\n")


def main():
    parser = argparse.ArgumentParser(description="Convert InkML files to PNG images")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mathwriting", "crohme", "all"],
        default="all",
        help="Which dataset to convert"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/Data/img-to-latex"),
        help="Base data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for images (default: <data-dir>/converted)"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=256,
        help="Output image size (square)"
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Stroke line width in pixels"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers"
    )
    args = parser.parse_args()
    
    output_base = args.output_dir or (args.data_dir / "converted")
    
    datasets_to_process = []
    
    if args.dataset in ["mathwriting", "all"]:
        mw_dir = args.data_dir / "mathwriting-2024"
        for split in ["train", "valid", "test", "synthetic"]:
            split_dir = mw_dir / split
            if split_dir.exists():
                datasets_to_process.append((
                    split_dir,
                    output_base / "mathwriting" / split,
                    f"mathwriting_{split}"
                ))
    
    if args.dataset in ["crohme", "all"]:
        tc11_dir = args.data_dir / "TC11_package"
        
        # CROHME 2016
        crohme_dirs = [
            (tc11_dir / "CROHME2016_data" / "TEST2016_INKML_GT", "crohme2016_test"),
            (tc11_dir / "CROHME2013_data" / "TrainINKML", "crohme2013_train"),
            (tc11_dir / "CROHME2013_data" / "TestINKMLGT", "crohme2013_test"),
            (tc11_dir / "CROHME2014_data" / "TestEM2014GT", "crohme2014_test"),
            (tc11_dir / "CROHME2012_data" / "trainData", "crohme2012_train"),
            (tc11_dir / "CROHME2012_data" / "testDataGT", "crohme2012_test"),
        ]
        
        for src_dir, name in crohme_dirs:
            if src_dir.exists():
                datasets_to_process.append((
                    src_dir,
                    output_base / "crohme" / name,
                    name
                ))
    
    print(f"Found {len(datasets_to_process)} dataset splits to process")
    
    for src_dir, dst_dir, name in datasets_to_process:
        print(f"\nProcessing {name}...")
        labels = process_dataset(
            src_dir, dst_dir,
            img_size=args.img_size,
            line_width=args.line_width,
            num_workers=args.num_workers
        )
        
        if labels:
            labels_path = dst_dir / "labels.txt"
            save_labels(labels, labels_path)
            print(f"  Saved {len(labels)} labels to {labels_path}")


if __name__ == "__main__":
    main()

