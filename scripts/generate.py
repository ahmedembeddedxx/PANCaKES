import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import choice
import random
from matplotlib.patches import Polygon
import torch
import json
import os
import numpy as np
import copy
import cv2
from tqdm import tqdm
from random import choice, randint


def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    return rotated, matrix

def flip_image(image, flip_code):
    return cv2.flip(image, flip_code)

def transform_segmentation(segmentation, matrix, img_w, img_h, flip_code=None):
    transformed_segmentation = []
    for polygon in segmentation:
        points = np.array(polygon).reshape(-1, 2)
        if matrix is not None:
            ones = np.ones((points.shape[0], 1))
            points_homogeneous = np.hstack([points, ones])
            transformed_points = points_homogeneous @ matrix.T
            points = transformed_points[:, :2]
        if flip_code is not None:
            if flip_code == 1:  
                points[:, 0] = img_w - points[:, 0]
            elif flip_code == 0:  
                points[:, 1] = img_h - points[:, 1]
            elif flip_code == -1:  
                points[:, 0] = img_w - points[:, 0]
                points[:, 1] = img_h - points[:, 1]
        transformed_segmentation.append(points.flatten().tolist())
    return transformed_segmentation

def augment_dataset(coco_json, image_folder, output_json, class_id=3, num_augment=3, restrict_class=[]):
    with open(coco_json, "r") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    annotations = data["annotations"]

    class_annotations = [ann for ann in annotations if ann["category_id"] == class_id and ann["category_id"] not in restrict_class]
    class_images = list(set(ann["image_id"] for ann in class_annotations))

    new_images = []
    new_annotations = []

    image_id_offset = max(images.keys()) + 1
    annotation_id_offset = max(ann["id"] for ann in annotations) + 1

    os.makedirs(image_folder, exist_ok=True)

    for img_id in tqdm(class_images, desc="Augmenting Images"):
        img_data = images[img_id]
        if "image" not in img_data["file_name"]:
            continue
        img_path = os.path.join(image_folder, img_data["file_name"])
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue

        img_h, img_w = img.shape[:2]
        img_annotations = [ann for ann in annotations if ann["image_id"] == img_id and ann["category_id"] not in restrict_class]

        new_images.append({
            "id": img_data["id"],
            "file_name": img_data["file_name"],
            "width": img_w,
            "height": img_h
        })
        for ann in img_annotations:
            new_ann = copy.deepcopy(ann)
            new_annotations.append(new_ann)

        for i in range(num_augment):
            aug_img = img.copy()
            aug_img_id = image_id_offset
            aug_ann_id = annotation_id_offset

            aug_type = random.choice(["rotate_180", "flip_h"])
            matrix = None
            flip_code = None

            if aug_type == "rotate_180":
                aug_img, matrix = rotate_image(aug_img, 180)
            elif aug_type == "flip_h":
                aug_img = flip_image(aug_img, 1)
                flip_code = 1

            aug_img_name = f"aug_{aug_img_id}.png"
            aug_img_path = os.path.join(image_folder, aug_img_name)
            cv2.imwrite(aug_img_path, aug_img)

            new_images.append({
                "id": aug_img_id,
                "file_name": aug_img_name,
                "width": img_w,
                "height": img_h
            })

            for ann in img_annotations:
                new_ann = copy.deepcopy(ann)
                new_ann["id"] = aug_ann_id
                new_ann["image_id"] = aug_img_id
                new_ann["segmentation"] = transform_segmentation(
                    new_ann["segmentation"], matrix, img_w, img_h, flip_code
                )
                new_annotations.append(new_ann)
                aug_ann_id += 1

            image_id_offset += 1
            annotation_id_offset = aug_ann_id

    data["images"].extend(new_images)
    data["annotations"].extend(new_annotations)

    with open(output_json, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Augmented dataset saved to {output_json}")