from lvis import LVIS

obj = LVIS("./lvis_v1_train.json")

list_cat_ids = obj.get_cat_ids()

print(list_cat_ids)

