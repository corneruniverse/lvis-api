from lvis import LVIS
from collections import defaultdict

obj = LVIS("../dataset/lvis_v1_train.json")

# obj.write_categories_to_csv("lvis_val_categories.csv")
# obj.write_annotations_for_category(21,"lvis_train_armor_annotations.csv")

# obj.download_images_for_category(301, 1, "convertible_dataset/images/train", "convertible_dataset/labels/train")
category_id = 301
images_folder_path = "convertible2/images/train"

image_ids_for_category = set()
image_ids_for_category = obj.get_image_ids(category_id)
print("Image IDs for Category:")
print(image_ids_for_category)

obj.download_images_for_category(images_folder_path, category_id)