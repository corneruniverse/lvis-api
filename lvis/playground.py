from lvis import LVIS

obj = LVIS("../dataset/lvis_v1_train.json")

category_names = obj.get_cat_names()

print(category_names)

