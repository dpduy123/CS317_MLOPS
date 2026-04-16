import os
from tqdm import tqdm

# data_path = '../dataset/yolo_plate_dataset'

# list_data = os.listdir(data_path)

# category = ['CarLongPlateGen', 'xemay', 'boderngoaigiao', 'CarLongPlate', 'brightnessquandoi', 'xemayBigPlate', 'cropquandoi', 'quandoi', 'boderquandoi', 'ngoaigiao', 'rotatequandoi', 'rotatengoaigiao', 'cropngoaigiao', 'brightnessngoaigiao']

# for cat in category:
#     if not os.path.exists(os.path.join('../dataset/yolo_plate_dataset_v1', cat)):
#         os.makedirs(os.path.join('../dataset/yolo_plate_dataset_v1', cat))
#         os.makedirs(os.path.join('../dataset/yolo_plate_dataset_v1', cat, 'images'))
#         os.makedirs(os.path.join('../dataset/yolo_plate_dataset_v1', cat, 'labels'))

# for name in tqdm(list_data):
#     tmp = name.split('.')
#     cat = tmp[0]
#     end = tmp[1]

#     while cat[-1].isdigit():
#         cat = cat[:-1]
    
#     if end == 'jpg':
#         os.system(f'cp {os.path.join(data_path, name)} {os.path.join("../dataset/yolo_plate_dataset_v1", cat, "images", name)}')
#     else:
#         os.system(f'cp {os.path.join(data_path, name)} {os.path.join("../dataset/yolo_plate_dataset_v1", cat, "labels", name)}')


# sum = 0
# for cat in category:
#     print(cat,": ", len(os.listdir(os.path.join('../dataset/yolo_plate_dataset_v1', cat, 'labels'))))
#     sum += len(os.listdir(os.path.join('../dataset/yolo_plate_dataset_v1', cat, 'labels')))
# print("Total: ", sum)

import os
import shutil
import random

# ===== CONFIG =====

root_datasets = "../dataset/yolo_plate_dataset_split"
output_dir = "../dataset/yolo_plate_dataset_v1"

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

random.seed(42)

# ==================

def create_dirs(base):

    os.makedirs(
        os.path.join(base, "images"),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(base, "labels"),
        exist_ok=True
    )


# Create output folders
create_dirs(os.path.join(output_dir, "train"))
create_dirs(os.path.join(output_dir, "val"))
create_dirs(os.path.join(output_dir, "test"))


def copy_pairs(pairs, split_name):
    for img_path, label_path in pairs:
        img_name = os.path.basename(img_path)
        label_name = os.path.basename(label_path)

        shutil.copy(
            img_path,
            os.path.join(
                output_dir,
                split_name,
                "images",
                img_name
            )
        )

        shutil.copy(
            label_path,
            os.path.join(
                output_dir,
                split_name,
                "labels",
                label_name
            )
        )


# ===== Split từng folder =====

total_train = 0
total_val = 0
total_test = 0


for folder in os.listdir(root_datasets):

    folder_path = os.path.join(root_datasets, folder)

    images_dir = os.path.join(folder_path, "images")
    labels_dir = os.path.join(folder_path, "labels")

    if not os.path.isdir(images_dir):
        continue

    pairs = []

    for img_name in os.listdir(images_dir):

        if img_name.endswith(".jpg"):

            img_path = os.path.join(images_dir, img_name)

            label_name = img_name.replace(".jpg", ".txt")
            label_path = os.path.join(labels_dir, label_name)

            if os.path.exists(label_path):

                pairs.append(
                    (img_path, label_path)
                )

    random.shuffle(pairs)

    n = len(pairs)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]

    print(
        folder,
        "| Train:", len(train_pairs),
        "| Val:", len(val_pairs),
        "| Test:", len(test_pairs)
    )

    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs, "val")
    copy_pairs(test_pairs, "test")

    total_train += len(train_pairs)
    total_val += len(val_pairs)
    total_test += len(test_pairs)


print("\n===== FINAL COUNT =====")
print("Train:", total_train)
print("Val:", total_val)
print("Test:", total_test)

print("\nDone splitting dataset.")