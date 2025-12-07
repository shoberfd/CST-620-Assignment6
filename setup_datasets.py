import os
import cv2
import glob
import shutil

# --- CONFIGURATION ---
# Path to the folder you just downloaded/extracted (must contain 'train' folder with images)
# Example: If you extracted to a folder named 'maps', put "./maps" here.
SOURCE_PATH = "./maps" 
OUTPUT_ROOT = "./data"
IMG_COUNT = 500  # Limit to 50 images for <5 min training

def prepare_data():
    # Create directories expected by your assignment code
    # Structure: data/satellite/images/ and data/map/images/
    os.makedirs(f"{OUTPUT_ROOT}/satellite/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_ROOT}/map/images", exist_ok=True)

    # Get list of files
    files = glob.glob(f"{SOURCE_PATH}/train/*.jpg") + glob.glob(f"{SOURCE_PATH}/train/*.png")
    files = files[:IMG_COUNT] # Keep only the first 50

    print(f"Processing {len(files)} images...")

    for i, file_path in enumerate(files):
        # Read the side-by-side image
        img = cv2.imread(file_path)
        if img is None: continue

        # Split: Original Pix2Pix maps are [Satellite | Map] (1200x600)
        h, w, _ = img.shape
        half_w = w // 2
        
        # Satellite is usually the left half, Map is the right half
        sat_img = img[:, :half_w, :]
        map_img = img[:, half_w:, :]

        # Save to separate folders
        # We use a dummy subfolder "images" because ImageFolder requires strictly "root/class/img"
        cv2.imwrite(f"{OUTPUT_ROOT}/satellite/images/{i}.jpg", sat_img)
        cv2.imwrite(f"{OUTPUT_ROOT}/map/images/{i}.jpg", map_img)

    print("✅ Data preparation complete!")
    print(f"   Satellite images saved to: {OUTPUT_ROOT}/satellite/images")
    print(f"   Map images saved to:       {OUTPUT_ROOT}/map/images")
    print("   You are ready to run pix2pix.py!")

if __name__ == "__main__":
    prepare_data()