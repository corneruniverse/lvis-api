from lvis import LVIS

obj = LVIS("../dataset/lvis_v1_train.json")

# obj.write_categories_to_csv("lvis_val_categories.csv")
obj.write_annotations_for_category(21,"lvis_train_armor_annotations.csv")

