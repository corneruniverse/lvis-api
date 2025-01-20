from lvis import LVIS
from collections import defaultdict
import json


def get_categories_alphabetized(json_path):
    """
    Reads a JSON file containing LVIS categories and returns the
    categories alphabetized.
    
    The JSON is expected to be a list of objects, where each object
    at least has an 'name' field.
    
    Args:
        json_path (str): Path to the 'lvis_categories.json' file.
    
    Returns:
        list: Top 25 category objects (sorted descending by image_count).
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        categories = json.load(f)

    # Sort by name ascending
    categories_sorted = sorted(categories, key=lambda cat: cat['name'], reverse=False)

    # Return all
    return categories_sorted

def write_to_json(json_path, data):
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Example usage:
if __name__ == "__main__":
    json_path = "lvis_categories_train_top_25.json"
    top_25 = get_categories_alphabetized(json_path)
    for cat in top_25:
        print(f"ID: {cat['id']} | image_count: {cat['image_count']} | name/synset: {cat['synset']}")
    
    train_obj = LVIS("../dataset/lvis_v1_train.json")
    val_obj = LVIS("../dataset/lvis_v1_val.json")
    images_folder_path = "test/top_25/images/train"
    labels_folder_path = "test/top_25/labels/train"

    images_folder_path = "test/top_25/images/val"
    labels_folder_path = "test/top_25/labels/val"
    detect_yaml = defaultdict(list)
    category_index = 0
    for category in top_25:
        print(f"name: {category["name"]} , category_index = {category_index}")
        # Group categories by their index for easy lookup in yaml
        detect_yaml[category_index] = category["name"]
        category_id = category["id"]
        train_obj.download_images_for_category(images_folder_path, category_id)
        train_obj.export_labels(labels_folder_path, category_id, category_index)

        val_obj.download_images_for_category(images_folder_path, category_id)
        val_obj.export_labels(labels_folder_path, category_id, category_index)

        #end of the loop, increase category index
        category_index += 1

    write_to_json("top_25_detect_yaml.json", detect_yaml)