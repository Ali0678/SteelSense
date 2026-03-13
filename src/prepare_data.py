import os
import shutil

source_dir = '../data/NEU-DET/IMAGES' 
dest_dir = '../data/NEU_Clean'

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
    print(f"Created clean directory: {dest_dir}")

files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
print(f"Found {len(files)} images. Sorting them now...")

count = 0
for filename in files:
    if "_" in filename:
        
        class_name = filename.split('_')[0]
        class_folder = os.path.join(dest_dir, class_name)
        
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
            
        src_path = os.path.join(source_dir, filename)
        dst_path = os.path.join(class_folder, filename)
        shutil.copy(src_path, dst_path)
        count += 1
    else:
        print(f"Skipping weird file: {filename}")

print(f"Success! Organized {count} images into '{dest_dir}'")
