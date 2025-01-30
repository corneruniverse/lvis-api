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
    json_path = "lvis_categories_train_govivid_62.json"
    govivid_62 = get_categories_alphabetized(json_path)
    for cat in govivid_62:
        print(f"ID: {cat['id']} | image_count: {cat['image_count']} | name/synset: {cat['synset']}")
    
    train_obj = LVIS("../dataset/lvis_v1_train.json")
    val_obj = LVIS("../dataset/lvis_v1_val.json")
    test_obj = LVIS("../dataset/lvis_v1_image_info_test_dev.json")

    images_folder_path_train = "test/govivid_62/images/train"
    labels_folder_path_train = "test/govivid_62/labels/train"

    images_folder_path_val = "test/govivid_62/images/val"
    labels_folder_path_val = "test/govivid_62/labels/val"

    images_folder_path_test = "test/govivid_62/images/test"
    labels_folder_path_test = "test/govivid_62/labels/test"

    detect_yaml = defaultdict(list)
    category_index = 0
    for category in govivid_62:
        print(f"name: {category['name']} , category_index = {category_index}")
        # Group categories by their index for easy lookup in yaml
        detect_yaml[category_index] = category["name"]
        category_id = category["id"]
        train_obj.download_images_for_category(images_folder_path_train, category_id)
        train_obj.export_labels(labels_folder_path_train, category_id, category_index)

        val_obj.download_images_for_category(images_folder_path_val, category_id)
        val_obj.export_labels(labels_folder_path_val, category_id, category_index)

        test_obj.download_images_for_category(images_folder_path_test, category_id)
        test_obj.export_labels(labels_folder_path_test, category_id, category_index)

        #end of the loop, increase category index
        category_index += 1

    write_to_json("govivid_62_detect_yaml.json", detect_yaml)