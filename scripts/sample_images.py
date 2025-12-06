import os
import random
import shutil

# Fix random seed for reproducibility
random.seed(42)

def sample_images_recursive(src_root, dst_root, n=5, exts=('.png', '.jpg', '.jpeg')):
    """
    Copy n random images per subfolder from src_root to dst_root
    while preserving subfolder structure. Works from any working directory.
    """
    # Convert to absolute paths
    src_root = os.path.abspath(src_root)
    dst_root = os.path.abspath(dst_root)

    for root, dirs, files in os.walk(src_root):
        # Filter image files
        image_files = [f for f in files if f.lower().endswith(exts)]
        if not image_files:
            continue
        
        # Pick up to n random images
        sample_files = random.sample(image_files, min(n, len(image_files)))
        
        # Determine relative path to recreate subfolder in destination
        rel_path = os.path.relpath(root, src_root)
        dst_folder = os.path.join(dst_root, rel_path)
        os.makedirs(dst_folder, exist_ok=True)
        
        # Copy sampled files
        for f in sample_files:
            shutil.copy(os.path.join(root, f), os.path.join(dst_folder, f))
        print(f"Copied {len(sample_files)} images to {dst_folder}")


sample_images_recursive('data/img-to-latex/converted/crohme/crohme2012_train',
                        'notebooks/samples/crohme/2012', n=100)

sample_images_recursive('data/img-to-latex/converted/crohme/crohme2013_train',
                        'notebooks/samples/crohme/2013', n=100)

sample_images_recursive('data/img-to-latex/converted/mathwriting/train',
                        'notebooks/samples/mathwriting', n=100)

sample_images_recursive('data/img-to-latex/HME100K/train/train_images',
                        'notebooks/samples/HME100K', n=100)
