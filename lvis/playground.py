from lvis import LVIS
from collections import defaultdict


# category_id = 301 # 301 = convertible
# category_id = 21 # 21 = armor
train_obj = LVIS("../dataset/lvis_v1_train.json")
val_obj = LVIS("../dataset/lvis_v1_val.json")

category_id = 45 # 45 = banana
category_index = 0

images_folder_path = "test/banana_armor/images/train"
labels_folder_path = "test/banana_armor/labels/train"
train_obj.download_images_for_category(images_folder_path, category_id)
train_obj.export_labels(labels_folder_path, category_id, category_index)

images_folder_path = "test/banana_armor/images/val"
labels_folder_path = "test/banana_armor/labels/val"
val_obj.download_images_for_category(images_folder_path, category_id)
val_obj.export_labels(labels_folder_path, category_id, category_index)

category_id = 21 # 21 = armor
category_index = 1

images_folder_path = "test/banana_armor/images/train"
labels_folder_path = "test/banana_armor/labels/train"
train_obj.download_images_for_category(images_folder_path, category_id)
train_obj.export_labels(labels_folder_path, category_id, category_index)

images_folder_path = "test/banana_armor/images/val"
labels_folder_path = "test/banana_armor/labels/val"
val_obj.download_images_for_category(images_folder_path, category_id)
val_obj.export_labels(labels_folder_path, category_id, category_index)