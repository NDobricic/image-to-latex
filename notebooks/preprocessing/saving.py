import sys
from pathlib import Path
import cv2
from tqdm import tqdm
from img_preprocessing import *

def process_and_save_best_images(input_folder, output_folder, W=400, H=96):
    """
    Processes all images in input_folder, selects the best binarized version
    (soft vs hard), and saves them as .jpg in output_folder.
    Skips images already existing in the output folder.
    """
    # Make paths absolute
    input_folder = Path(input_folder).expanduser().resolve()
    output_folder = Path(output_folder).expanduser().resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    img_files = [f for f in input_folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]

    if not img_files:
        print(f"No images found in {input_folder}")
        return

    for file in tqdm(img_files, desc="Processing images"):
        save_path = output_folder / file.with_suffix('.jpg').name

        if save_path.exists():
            continue  # skip already processed images

        img = cv2.imread(str(file))
        if img is None:
            print(f"Warning: could not read {file}, skipping.")
            continue

        best_img, _ = get_best_soft_hard_image(img, W=W, H=H)

        # save as JPG
        cv2.imwrite(str(save_path), best_img)

    print(f"Processed {len(img_files)} images (skipped existing) and saved to {output_folder}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python saving.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    process_and_save_best_images(input_folder, output_folder)
