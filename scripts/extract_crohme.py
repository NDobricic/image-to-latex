#!/usr/bin/env python3
"""Extract CROHME zip files to a consistent directory structure."""

import argparse
import zipfile
from pathlib import Path


CROHME_ZIPS = [
    "CROHME2016_data/Task-1-Formula.zip",
    "CROHME2013_data/TrainINKML.zip",
    "CROHME2013_data/TestINKMLGT.zip",
    "CROHME2013_data/TestINKML.zip",
    "CROHME2014_data/TestEM2014GT.zip",
    "CROHME2012_data/trainData/trainData.zip",
    "CROHME2012_data/testData/testData.zip",
]


def extract_zip(zip_path: Path, output_dir: Path) -> int:
    """Extract a single zip file and return the number of files extracted."""
    if not zip_path.exists():
        print(f"  [SKIP] {zip_path.name} not found")
        return 0
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = zf.namelist()
        zf.extractall(output_dir)
        print(f"  [OK] {zip_path.name} -> {len(members)} files")
        return len(members)


def main():
    parser = argparse.ArgumentParser(description="Extract CROHME dataset zip files")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/Data/img-to-latex/TC11_package"),
        help="Path to TC11_package directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: extract in place)"
    )
    args = parser.parse_args()
    
    total_files = 0
    
    print("Extracting CROHME zip files...")
    for zip_rel_path in CROHME_ZIPS:
        zip_path = args.data_dir / zip_rel_path
        output_dir = args.output_dir if args.output_dir else zip_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        total_files += extract_zip(zip_path, output_dir)
    
    print(f"\nDone! Extracted {total_files} total files.")


if __name__ == "__main__":
    main()

