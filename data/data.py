import kagglehub
import shutil
import os

# Download dataset (returns cache path)
path = kagglehub.dataset_download("rodsaldanha/arketing-campaign")

# Define your project data folder
local_path = "raw/marketing-campaign"

# Create it if needed
os.makedirs(local_path, exist_ok=True)

# Move or copy data to your project folder
shutil.copytree(path, local_path, dirs_exist_ok=True)

print(f"Dataset copied to: {local_path}")
