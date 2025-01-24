import json
from lvis import LVIS
from collections import defaultdict
from custom_categories import lvis_category_names_govivid_95


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

def get_categories_by_name(json_path, name_array):
    """
    Accepts an string array containing names of LVIS categories and 
    reads a JSON file containing LVIS categories.

    The JSON is expected to be a list of objects, where each object
    at least has an 'name' field.

    The function returns the JSON objects where the 'name' field matches a name in the string array.
    
    Args:
        json_path (str): Path to the 'lvis_categories.json' file.
    
    Returns:
        list: objects of matching LVIS categories (sorted descending by image_count).
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        categories = json.load(f)

    # Filter categories by name
    filtered_categories = [cat for cat in categories if cat.get('synset') in name_array]
    
    # # Sort by image_count in descending order
    # sorted_categories = sorted(filtered_categories, key=lambda x: x.get('image_count', 0), reverse=True)
    
    # Return the results
    return filtered_categories


def write_to_json(json_path, data):
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Example usage:
if __name__ == "__main__":
    json_path = "lvis_categories_train.json"
    categories_govivid_95 = get_categories_by_name(json_path, lvis_category_names_govivid_95)
    for cat in categories_govivid_95:
        print(f"ID: {cat['id']} | image_count: {cat['image_count']} | name/synset: {cat['synset']}")

    govivid_95_json_path = "lvis_categories_train_govivid_95.json"
    write_to_json(govivid_95_json_path, categories_govivid_95)