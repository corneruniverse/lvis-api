import json
from lvis import LVIS
from collections import defaultdict

# the goal is to correspond the category names and category ids
# and save them to a json file
train_obj = LVIS("../dataset/lvis_v1_train.json")
val_obj = LVIS("../dataset/lvis_v1_val.json")

# categories = train_obj.get_cat()
# with open('lvis_categories.json', 'w', encoding='utf-8') as f:
#     json.dump(categories, f, ensure_ascii=False, indent=4)

# read json file 'lvis_categories.json'
# return the 25 objects where the field "image_count" is highest
def get_top_25_categories_by_image_count(json_path):
    """
    Reads a JSON file containing LVIS categories and returns the
    25 categories with the highest 'image_count'.
    
    The JSON is expected to be a list of objects, where each object
    at least has an 'image_count' field.
    
    Args:
        json_path (str): Path to the 'lvis_categories.json' file.
    
    Returns:
        list: Top 25 category objects (sorted descending by image_count).
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        categories = json.load(f)

    # Sort by image_count descending
    categories_sorted = sorted(categories, key=lambda cat: cat['image_count'], reverse=True)

    # Return the top 25
    return categories_sorted[:25]

def write_to_json(json_path, data):
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Example usage:
if __name__ == "__main__":
    json_path = "lvis_categories_train.json"
    top_25 = get_top_25_categories_by_image_count(json_path)
    for cat in top_25:
        print(f"ID: {cat['id']} | image_count: {cat['image_count']} | name/synset: {cat['synset']}")

    top_25_json_path = "lvis_categories_train_top_25.json"
    write_to_json(top_25_json_path, top_25)