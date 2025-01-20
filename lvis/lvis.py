"""
API for accessing LVIS Dataset: https://lvisdataset.org.

LVIS API is a Python API that assists in loading, parsing and visualizing
the annotations in LVIS. In addition to this API, please download
images and annotations from the LVIS website.
"""

import json
import os
import logging
from collections import defaultdict
from urllib.request import urlretrieve

import pycocotools.mask as mask_utils

import csv

import requests


class LVIS:
    def __init__(self, annotation_path):
        """Class for reading and visualizing annotations.
        Args:
            annotation_path (str): location of annotation file
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Loading annotations.")

        self.dataset = self._load_json(annotation_path)

        assert (
            type(self.dataset) == dict
        ), "Annotation file format {} not supported.".format(type(self.dataset))
        self._create_index()

    def _load_json(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def _create_index(self):
        self.logger.info("Creating index.")

        self.img_ann_map = defaultdict(list)
        self.cat_img_map = defaultdict(list)

        self.anns = {}
        self.cats = {}
        self.imgs = {}

        for ann in self.dataset["annotations"]:
            self.img_ann_map[ann["image_id"]].append(ann)
            self.anns[ann["id"]] = ann

        for img in self.dataset["images"]:
            self.imgs[img["id"]] = img

        for cat in self.dataset["categories"]:
            self.cats[cat["id"]] = cat

        for ann in self.dataset["annotations"]:
            self.cat_img_map[ann["category_id"]].append(ann["image_id"])

        self.logger.info("Index created.")

    def get_ann_ids(self, img_ids=None, cat_ids=None, area_rng=None):
        """Get ann ids that satisfy given filter conditions.

        Args:
            img_ids (int array): get anns for given imgs
            cat_ids (int array): get anns for given cats
            area_rng (float array): get anns for a given area range. e.g [0, inf]

        Returns:
            ids (int array): integer array of ann ids
        """
        anns = []
        if img_ids is not None:
            for img_id in img_ids:
                anns.extend(self.img_ann_map[img_id])
        else:
            anns = self.dataset["annotations"]

        # return early if no more filtering required
        if cat_ids is None and area_rng is None:
            return [_ann["id"] for _ann in anns]

        cat_ids = set(cat_ids)

        if area_rng is None:
            area_rng = [0, float("inf")]

        ann_ids = [
            _ann["id"]
            for _ann in anns
            if _ann["category_id"] in cat_ids
            and _ann["area"] > area_rng[0]
            and _ann["area"] < area_rng[1]
        ]
        return ann_ids

    def get_cat_ids(self):
        """Get all category ids.

        Returns:
            ids (int array): integer array of category ids
        """
        return list(self.cats.keys())
        

    def get_img_ids(self):
        """Get all img ids.

        Returns:
            ids (int array): integer array of image ids
        """
        return list(self.imgs.keys())

    def _load_helper(self, _dict, ids):
        if ids is None:
            return list(_dict.values())
        else:
            return [_dict[id] for id in ids]

    def load_anns(self, ids=None):
        """Load anns with the specified ids. If ids=None load all anns.

        Args:
            ids (int array): integer array of annotation ids

        Returns:
            anns (dict array) : loaded annotation objects
        """
        return self._load_helper(self.anns, ids)

    def load_cats(self, ids):
        """Load categories with the specified ids. If ids=None load all
        categories.

        Args:
            ids (int array): integer array of category ids

        Returns:
            cats (dict array) : loaded category dicts
        """
        return self._load_helper(self.cats, ids)

    def load_imgs(self, ids):
        """Load categories with the specified ids. If ids=None load all images.

        Args:
            ids (int array): integer array of image ids

        Returns:
            imgs (dict array) : loaded image dicts
        """
        return self._load_helper(self.imgs, ids)

    def download(self, save_dir, img_ids=None):
        """Download images from mscoco.org server.
        Args:
            save_dir (str): dir to save downloaded images
            img_ids (int array): img ids of images to download
        """
        imgs = self.load_imgs(img_ids)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for img in imgs:
            file_name = os.path.join(save_dir, img["coco_url"].split("/")[-1])
            if not os.path.exists(file_name):
                urlretrieve(img["coco_url"], file_name)

    def ann_to_rle(self, ann):
        """Convert annotation which can be polygons, uncompressed RLE to RLE.
        Args:
            ann (dict) : annotation object

        Returns:
            ann (rle)
        """
        img_data = self.imgs[ann["image_id"]]
        h, w = img_data["height"], img_data["width"]
        segm = ann["segmentation"]
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = mask_utils.frPyObjects(segm, h, w)
            rle = mask_utils.merge(rles)
        elif isinstance(segm["counts"], list):
            # uncompressed RLE
            rle = mask_utils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = ann["segmentation"]
        return rle

    def ann_to_mask(self, ann):
        """Convert annotation which can be polygons, uncompressed RLE, or RLE
        to binary mask.
        Args:
            ann (dict) : annotation object

        Returns:
            binary mask (numpy 2D array)
        """
        rle = self.ann_to_rle(ann)
        return mask_utils.decode(rle)

# -- custom functions below ---

    def get_cat(self):
        """Get all category info.

        Returns:
            ? (? array): string array of category names
        """
        return self.dataset["categories"]

    def write_categories_to_csv(self, csv_file_path):
        """
        Writes the LVIS categories to a CSV file.

        Args:
            csv_file_path (str): The path (including filename) where the CSV should be saved.
        """
        # Extract category info from the dataset
        categories = self.dataset.get('categories', [])

        # Define the CSV header
        # If you prefer a different name than 'def' for the CSV column, change it here (e.g., "definition")
        fieldnames = [
            'id',
            'synset',
            'synonyms',
            'def',
            'instance_count',
            'image_count',
            'frequency'
        ]

        # Open the file and create a writer
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            
            # Write the header
            writer.writeheader()
            
            # Write each category to the CSV
            for cat in categories:
                # Convert list of synonyms to a single string (e.g., joined by '|')
                synonyms_str = '|'.join(cat.get('synonyms', []))
                
                row_data = {
                    'id': cat.get('id', ''),
                    'synset': cat.get('synset', ''),
                    'synonyms': synonyms_str,
                    'def': cat.get('def', ''),
                    'instance_count': cat.get('instance_count', ''),
                    'image_count': cat.get('image_count', ''),
                    'frequency': cat.get('frequency', '')
                }
                
                writer.writerow(row_data)

        print(f"Categories successfully written to {csv_file_path}")


    def write_annotations_for_category(self, category_id, csv_file_path):
        """
        1) Gather all image IDs that have annotations for the specified category_id.
        2) From those images, collect only the annotations that match the same category_id.
        3) Write those annotations to a CSV file.

        The CSV will include:
        - id
        - image_id
        - category_id
        - segmentation
        - area
        - bbox

        Args:
            category_id (int): The category ID to filter on.
            csv_file_path (str): The path (including filename) for the CSV file.
        """

        # 1) Identify all image_ids that contain this category.
        image_ids_for_category = set()
        for ann in self.dataset['annotations']:
            if ann['category_id'] == category_id:
                image_ids_for_category.add(ann['image_id'])

        # 2) Filter annotations for those images AND matching the category_id.
        relevant_annotations = []
        for ann in self.dataset['annotations']:
            if (ann['image_id'] in image_ids_for_category) and (ann['category_id'] == category_id):
                relevant_annotations.append(ann)

        # 3) Write the filtered annotations to CSV.
        fieldnames = ['id', 'image_id', 'category_id', 'segmentation', 'area', 'bbox']
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for ann in relevant_annotations:
                row_data = {
                    'id': ann.get('id', ''),
                    'image_id': ann.get('image_id', ''),
                    'category_id': ann.get('category_id', ''),
                    # segmentation can be a list (polygon) or RLE. Convert as needed.
                    'segmentation': ann.get('segmentation', ''),
                    'area': ann.get('area', ''),
                    'bbox': ann.get('bbox', '')  # Usually [x, y, width, height]
                }
                writer.writerow(row_data)

        print(f"Found {len(image_ids_for_category)} images for category_id={category_id}. "
            f"Wrote {len(relevant_annotations)} annotations to '{csv_file_path}'.")


    def get_image_ids(self, category_id):
        # 1) Identify all image_ids containing this category
        image_ids_for_category = set()
        for ann in self.dataset['annotations']:
            if ann['category_id'] == category_id:
                image_ids_for_category.add(ann['image_id'])
        return image_ids_for_category

    def get_annotations(self, category_id, image_ids_for_category):
        # 2) Collect annotations that match both the category and the image_ids
        relevant_annotations = []
        for ann in self.dataset['annotations']:
            if (ann['image_id'] in image_ids_for_category) and (ann['category_id'] == category_id):
                relevant_annotations.append(ann)
        # Group annotations by their image_id for easy lookup
        image_id_to_annotations = defaultdict(list)
        for ann in relevant_annotations:
            image_id_to_annotations[ann['image_id']].append(ann)
        return image_id_to_annotations

    
    
    def download_images_for_category(self, images_folder_path, category_id):
        """
        1) Gather all image IDs that have annotations for the specified category_id.
        2) Download the images of those annotations to the specified images folder path.

        Args:
            images_folder_path (str): The directory to save the images.
            category_id (int): The category ID to filter on.
        """

        # Ensure output directories exist
        os.makedirs(images_folder_path, exist_ok=True)
        # os.makedirs(labels_folder_path, exist_ok=True)

        # 1) Identify all image_ids containing this category
        image_ids_for_category = set()
        image_ids_for_category = self.get_image_ids(category_id)

        # 2) Collect annotations that match both the category and the image_ids
        # Group annotations by their image_id for easy lookup
        # image_id_to_annotations = defaultdict(list)
        # image_id_to_annotations = self.get_annotations(category_id, image_ids_for_category)

        # 3) Download each image
        self.download(images_folder_path, image_ids_for_category)
        
        print(f"Downloaded images for category_id={category_id}.\n"
            f"Images in: {images_folder_path}\n")
        # print(f"Downloaded images and created YOLO labels for category_id={category_id}.\n"
        #     f"Images in: {images_folder_path}\nLabels in: {labels_folder_path}")

    def export_labels(self, labels_folder_path, category_id, category_index):
        """
        1) Gather all image IDs that have annotations for the specified category_id.
        2) From those images, collect only the annotations that match the same category_id.
        3) Label the images in YOLO format, exporting one *.txt file per image (if no objects in image, no *.txt file).
        Format: class x_center y_center width height
        - Box coordinates must be in normalized xywh.
        - Class numbers are zero-indexed (start from 0).

        Args:
            category_id (int): The category ID to filter on.
            images_folder_path (str): The directory to save the images.
            labels_folder_path (str): The directory to save the YOLO label files.
        """
        # Ensure output directories exist
        # os.makedirs(images_folder_path, exist_ok=True)
        os.makedirs(labels_folder_path, exist_ok=True)
        # 1) Identify all image_ids containing this category
        image_ids_for_category = set()
        image_ids_for_category = self.get_image_ids(category_id)

        # 2) Collect annotations that match both the category and the image_ids
        # Group annotations by their image_id for easy lookup
        image_id_to_annotations = defaultdict(list)
        image_id_to_annotations = self.get_annotations(category_id, image_ids_for_category)


        # 3) Export YOLO labels
        imgs = self.load_imgs(image_ids_for_category)
        for img in imgs:
            # file_name still includes .jpg at the end instead of .txt
            coco_url = img["coco_url"]
            image_name = coco_url.split("/")[-1]
            # replace the characters .jpg with .txt
            base_name = os.path.splitext(image_name)[0]  # Removes the extension
            label_file_name = base_name + '.txt'
            # Construct the label file path (same base name, .txt extension)
            label_file_path = os.path.join(labels_folder_path, label_file_name)
            img_width = img["width"]
            img_height = img["height"]

            # If missing critical info, skip
            if not (coco_url and img_width and img_height):
                continue

            # Gather bounding boxes for this image
            annotations_for_image = image_id_to_annotations[img["id"]]

            # If no bounding boxes for this category, do not create a label file
            if not annotations_for_image:
                continue


            for ann in annotations_for_image:
                # 'bbox' is typically [x, y, width, height] in pixel coordinates
                bbox = ann.get('bbox', None)
                if not bbox or len(bbox) != 4:
                    continue

                x, y, w, h = bbox

                # Convert to YOLO-style normalized coordinates
                x_center = (x + w / 2.0) / img_width
                y_center = (y + h / 2.0) / img_height
                w_norm = w / img_width
                h_norm = h / img_height

                # Categories are zero-indexed
                class_idx = category_index

                # Write one line per bounding box
                # Format as floats; you can refine precision as needed
                # Construct the line to be added
                # Construct the single bounding-box line we want to add
                bbox_line = f"{class_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                # label_file.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

                # If the file exists, make sure to not overwrite. Do not delete the information that is already there.
                # Gather existing lines if the file already exists
                existing_lines = set()
                if os.path.exists(label_file_path):
                    with open(label_file_path, 'r') as existing_file:
                        for line in existing_file:
                            existing_lines.add(line.strip())  # strip to remove trailing newline
                # If the file already contains what was to be written, skip
                # If this line isn't already in the file, append it
                if bbox_line not in existing_lines:
                    with open(label_file_path, 'a') as label_file:
                        label_file.write(bbox_line + "\n")
    
        

    def download_and_export_labels(self, image_ids_for_category, image_id_to_annotations):
        """
        1) Gather all image IDs that have annotations for the specified category_id.
        2) From those images, collect only the annotations that match the same category_id.
        3) Download the images of those annotations to the specified images folder path.
        4) Label the images in YOLO format, exporting one *.txt file per image (if no objects in image, no *.txt file).
        Format: class x_center y_center width height
        - Box coordinates must be in normalized xywh.
        - Class numbers are zero-indexed (start from 0).

        Args:
            category_id (int): The category ID to filter on.
            images_folder_path (str): The directory to save the images.
            labels_folder_path (str): The directory to save the YOLO label files.
        """
        # 3) Download each image and 4) export YOLO labels
        for image_id in image_ids_for_category:
            # Find the image info from self.dataset['images']
            image_info = next((img for img in self.dataset['images'] if img['id'] == image_id), None)
            if not image_info:
                continue

            coco_url = image_info.get('coco_url', None)
            # Use the original file_name or fall back to something like f"{image_id}.jpg"
            file_name = image_info.get('file_name', f"{image_id}.jpg")
            img_width = image_info.get('width', None)
            img_height = image_info.get('height', None)

            # If missing critical info, skip
            if not (coco_url and img_width and img_height):
                continue

            # Download the image
            try:
                response = requests.get(coco_url, timeout=15)
                if response.status_code != 200:
                    print(f"Warning: Failed to download image {image_id} from {coco_url}")
                    continue
            except requests.exceptions.RequestException as e:
                print(f"Warning: Request error for image {image_id}: {e}")
                continue

            # Save the image file
            image_path = os.path.join(images_folder_path, file_name)
            with open(image_path, 'wb') as img_file:
                img_file.write(response.content)

            # Gather bounding boxes for this image
            annotations_for_image = image_id_to_annotations[image_id]

            # If no bounding boxes for this category, do not create a label file
            if not annotations_for_image:
                continue

            # Construct the label file path (same base name, .txt extension)
            label_file_name = os.path.splitext(file_name)[0] + ".txt"
            label_file_path = os.path.join(labels_folder_path, label_file_name)

            with open(label_file_path, 'w') as label_file:
                for ann in annotations_for_image:
                    # 'bbox' is typically [x, y, width, height] in pixel coordinates
                    bbox = ann.get('bbox', None)
                    if not bbox or len(bbox) != 4:
                        continue

                    x, y, w, h = bbox

                    # Convert to YOLO-style normalized coordinates
                    x_center = (x + w / 2.0) / img_width
                    y_center = (y + h / 2.0) / img_height
                    w_norm = w / img_width
                    h_norm = h / img_height

                    # Categories are zero-indexed
                    class_idx = category_index

                    # Write one line per bounding box
                    # Format as floats; you can refine precision as needed
                    label_file.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    
    

    def get_cat_names(self):
        """
        Returns a list of category names (strings) from the LVIS dataset.
        By default, this returns the first synonym in each category's 'synonyms' list.
        """
        cat_names = []
        categories = self.dataset.get('categories', [])
        
        for cat in categories:
            # Make sure 'synonyms' exists and has at least one element
            if 'synonyms' in cat and cat['synonyms']:
                cat_names.append(cat['synonyms'][0])
            else:
                # Fallback in case 'synonyms' is missing or empty
                cat_names.append(None)
                
        return cat_names

    def get_cat_names(self):
        """
        Returns a list of category names (strings) from the LVIS dataset.
        By default, this returns the first synonym in each category's 'synonyms' list.
        """
        cat_names = []
        categories = self.dataset.get('categories', [])
        
        for cat in categories:
            # Make sure 'synonyms' exists and has at least one element
            if 'synonyms' in cat and cat['synonyms']:
                cat_names.append(cat['synonyms'][0])
            else:
                # Fallback in case 'synonyms' is missing or empty
                cat_names.append(None)
                
        return cat_names

        

