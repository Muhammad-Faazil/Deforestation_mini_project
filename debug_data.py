import os
import cv2
import numpy as np
from PIL import Image

def debug_data_structure():
    data_dir = r"C:\Users\faazi\Desktop\Deforestation_CNN\data"
    
    print("🔍 Debugging data structure...")
    print(f"Data directory exists: {os.path.exists(data_dir)}")
    
    if os.path.exists(data_dir):
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                print(f"📁 Folder: {item}")
                # Count images in this folder
                image_count = 0
                for file in os.listdir(item_path):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_count += 1
                        if image_count == 1:  # Check first image
                            img_path = os.path.join(item_path, file)
                            try:
                                img = Image.open(img_path)
                                print(f"   First image: {file} - Size: {img.size} - Mode: {img.mode}")
                            except:
                                print(f"   First image: {file} - CORRUPTED")
                
                print(f"   Total images: {image_count}")
            else:
                print(f"📄 File: {item}")

def check_sample_images():
    """Check if we can actually load and display some images"""
    data_dir = r"C:\Users\faazi\Desktop\Deforestation_CNN\data"
    
    for category in ['deforestation', 'no_deforestation']:
        path = os.path.join(data_dir, category)
        if os.path.exists(path):
            print(f"\n📸 Checking {category} images:")
            images_checked = 0
            for filename in os.listdir(path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(path, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        print(f"   ✅ {filename}: {img.shape}")
                        images_checked += 1
                        if images_checked >= 3:  # Check first 3 images
                            break
                    else:
                        print(f"   ❌ {filename}: FAILED TO LOAD")

if __name__ == "__main__":
    debug_data_structure()
    check_sample_images()