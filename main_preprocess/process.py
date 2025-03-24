import os

# Path to the 'hsi' folder
hsi_path = "/workspace/src/extra_norm/hsi"  
rgb_path = "/workspace/src/extra_norm/rgb"

# Get list of subfolders
subfolders = sorted([f for f in os.listdir(hsi_path) if os.path.isdir(os.path.join(hsi_path, f))])

# Rename subfolders and their contents
for idx, old_folder_name in enumerate(subfolders, start=1):
    new_folder_name = f"s1_ext_norm{idx}"
    old_folder_path = os.path.join(hsi_path, old_folder_name)
    new_folder_path = os.path.join(hsi_path, new_folder_name)

    # Rename the folder
    os.rename(old_folder_path, new_folder_path)

    # Function to rename files in all subdirectories
    for root, dirs, files in os.walk(new_folder_path, topdown=True):
        for filename in files:
            if old_folder_name in filename:
                old_file_path = os.path.join(root, filename)
                new_filename = filename.replace(old_folder_name, new_folder_name)
                new_file_path = os.path.join(root, new_filename)
                os.rename(old_file_path, new_file_path)

print("Renaming HSI completed successfully.")


rgb_files = sorted([f for f in os.listdir(rgb_path) if f.endswith('.png')]) 

# Make sure we have the same number of RGB files as HSI folders
if len(rgb_files) != len(subfolders):
    print(f"Warning: Number of RGB files ({len(rgb_files)}) doesn't match number of HSI folders ({len(subfolders)})")


for idx, old_filename in enumerate(rgb_files, start=1):
    new_filename = f"s1_ext_norm{idx}.png"
    old_file_path = os.path.join(rgb_path, old_filename)
    new_file_path = os.path.join(rgb_path, new_filename)
    
    if os.path.exists(new_file_path):
        print(f"Warning: {new_file_path} already exists, skipping")
        continue
        
    os.rename(old_file_path, new_file_path)

print("Renaming RGB completed successfully.")
