import os

def check_sync(image_dir, label_dir):
    images = {os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))}
    labels = {os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.txt')}
    
    missing_labels = images - labels
    missing_images = labels - images
    
    print(f"Total Images: {len(images)} | Total Labels: {len(labels)}")
    print(f"Missing Labels (Images without .txt): {len(missing_labels)}")
    print(f"Missing Images (.txt without image): {len(missing_images)}")

# Cek folder train
print("--- Checking Train Folder ---")
check_sync('train/images', 'train/labels')

# Cek folder valid
print("\n--- Checking Valid Folder ---")
check_sync('valid/images', 'valid/labels')

# Cek folder test
print("\n--- Checking Test Folder ---")
check_sync('test/images', 'test/labels')