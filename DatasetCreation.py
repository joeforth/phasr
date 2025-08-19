import os
import shutil
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

# This import was in the original file, so it's kept for consistency,
# though cv2.imwritemulti is used for saving TIFFs in this version.


def create_combined_dataset(source_dirs, output_dir, splits=['train', 'valid', 'test']):
    """
    Processes multiple datasets by concatenating image triplets from each source,
    and saves them sequentially into a single output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- CHANGE: Initialize a counter for each split ---
    # This ensures unique filenames across all combined datasets.
    file_counters = {split: 0 for split in splits}

    for split in splits:
        print(f"--- Processing split: {split} ---")

        # --- CHANGE: Set up destination paths once per split ---
        dest_image_path = os.path.join(output_dir, split, 'images')
        dest_label_path = os.path.join(output_dir, split, 'labels')
        os.makedirs(dest_image_path, exist_ok=True)
        os.makedirs(dest_label_path, exist_ok=True)

        # --- CHANGE: Loop through each source dataset directory ---
        for source_dir in source_dirs:
            print(f"\nProcessing source: {source_dir}")

            source_image_path = os.path.join(source_dir, split, 'images')
            source_label_path = os.path.join(source_dir, split, 'labels')

            if not os.path.exists(source_image_path):
                print(f"Source path not found for split '{split}' in '{source_dir}', skipping.")
                continue

            image_files = sorted(os.listdir(source_image_path))

            if len(image_files) < 3:
                print(f"Not enough images to form a triplet in {source_image_path}, skipping.")
                continue

            for i in tqdm(range(len(image_files) - 2), desc=f"Concatenating from {os.path.basename(source_dir)}"):
                img1_name, img2_name, img3_name = image_files[i], image_files[i + 1], image_files[i + 2]

                try:
                    # Open and process images
                    img1 = Image.open(os.path.join(source_image_path, img1_name)).convert('RGB')
                    img2 = Image.open(os.path.join(source_image_path, img2_name)).convert('RGB')
                    img3 = Image.open(os.path.join(source_image_path, img3_name)).convert('RGB')

                    img1_arr, img2_arr, img3_arr = np.array(img1), np.array(img2), np.array(img3)

                    # Concatenate along the channel axis (axis=2)
                    concatenated_array = np.concatenate([img1_arr, img2_arr, img3_arr], axis=2)

                    # --- CHANGE: Generate a new, unique filename using the counter ---
                    # Formats the number with leading zeros (e.g., 000001, 000002)
                    output_basename = os.path.splitext(img1_name)[0]
                    output_image_file = os.path.join(dest_image_path, f"{output_basename}.tiff")

                    # Transpose from (H, W, C) to (C, H, W) for saving
                    array_for_cv2 = concatenated_array.transpose(2, 0, 1)

                    # Save the 9-channel TIFF image
                    cv2.imwritemulti(output_image_file, array_for_cv2)

                    # --- CHANGE: Process the corresponding label ---
                    # The original script associates the label of the *first* image in the triplet.
                    # We maintain that logic here.
                    label_to_copy_name = os.path.splitext(img1_name)[0] + '.txt'
                    source_label_file = os.path.join(source_label_path, label_to_copy_name)

                    if os.path.exists(source_label_file):
                        # The destination label name must match the new image name
                        dest_label_file = os.path.join(dest_label_path, f"{output_basename}.txt")
                        shutil.copy(source_label_file, dest_label_file)

                    # --- CHANGE: Increment the counter for the next file ---
                    file_counters[split] += 1

                except Exception as e:
                    print(f"Could not process triplet ending with {img3_name} from {source_dir}. Error: {e}")

    print("\n--- Combined dataset processing complete! ---")
    print(f"New combined dataset created at: {output_dir}")


# --- How to Use ---

if __name__ == '__main__':
    # 1. --- CHANGE: Define a LIST of paths to your original dataset directories ---
    SOURCE_DATASET_DIRS = [
        'Shampoo-segmentation-6',
        'Shampoo-segmentation-7',
        'Shampoo-segmentation-8',
        'Shampoo-segmentation-9'
          # Example: And your third one
        # Add as many dataset paths as you need
    ]

    # 2. Define the path where the new combined dataset will be saved
    OUTPUT_DATASET_DIR = 'Shampoo9C_Combined'

    # 3. Run the processing script
    create_combined_dataset(SOURCE_DATASET_DIRS, OUTPUT_DATASET_DIR)