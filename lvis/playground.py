from lvis import LVIS

obj = LVIS("../dataset/lvis_v1_val.json")

obj.write_categories_to_csv("lvis_val_categories.csv")


