import os
import shutil
from tqdm import tqdm

# Source and destination directories
split_type = 'test'
source_path = f"datasets/scooter-bicycle-detectionv4.v1i.yolov8/{split_type}"
source_image_dir = source_path + '/images'
source_label_dir = source_path + '/labels'

dest_path = f'cocodataset/bicycle/{split_type}'
destination_image_dir = dest_path + '/images'
destination_label_dir = dest_path + '/labels'

# Copy image files
for filename in tqdm(os.listdir(source_image_dir)):
    shutil.copy(os.path.join(source_image_dir, filename), destination_image_dir)

# Copy label files
for filename in tqdm(os.listdir(source_label_dir)):
    shutil.copy(os.path.join(source_label_dir, filename), destination_label_dir)