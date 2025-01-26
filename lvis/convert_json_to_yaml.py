import json
import yaml
# pip install pyyaml


def convert_to_yaml(json_path):
    """
    Reads a JSON file. The object in the JSON file would be formatted as "id" : "name", 
    where the id is string of an integer, and the name is a string of a name.
    
    Write the values to a YAML file called detect.yaml correctly formatted for Ultralytics Hub
    
    Args:
        json_path (str): Path to the 'lvis_categories.json' file.
    
    Returns:
        nil
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

   # Format the data into the required YAML structure
    yolo_data = {
        'train': 'images/train',
        'val': 'images/val',
        'names': {int(k): v.lower() for k, v in json_data.items()}
    }

    # Write to YAML file
    with open('yolo_classes.yaml', 'w') as yaml_file:
        yaml.dump(yolo_data, yaml_file, default_flow_style=False, sort_keys=False)

    print("YAML file has been created successfully.")

convert_to_yaml("govivid_95_detect_yaml.json")