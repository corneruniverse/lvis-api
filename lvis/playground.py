from lvis import LVIS

obj = LVIS("../dataset/lvis_v1_train.json")

obj.write_categories_to_csv("lvis_categories.csv")


