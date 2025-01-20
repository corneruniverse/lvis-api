from lvis import LVIS
from collections import defaultdict

category_id = 45 # 45 = banana
# category_id = 301 # 301 = convertible
# category_id = 21 # 21 = armor
obj = LVIS("../dataset/lvis_v1_train.json")
images_folder_path = "test/lvis_top1/images/train"
labels_folder_path = "test/lvis_top1/labels/train"

category_index = 0

# obj.download_images_for_category(images_folder_path, category_id)
obj.export_labels(labels_folder_path, category_id, category_index)