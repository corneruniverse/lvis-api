from lvis import LVIS



train_obj = LVIS("../dataset/lvis_v1_train.json")

train_obj.write_categories_to_csv("lvis_train.csv")