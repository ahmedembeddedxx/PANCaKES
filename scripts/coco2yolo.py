import json
import os
from tqdm import tqdm

def convert_coco_to_yolo_segmentation(json_file, folder_name = "labels"):
    folder_name = folder_name
    with open(json_file, 'r') as file:
        coco_data = json.load(file)

    output_folder = os.path.join(os.path.dirname(json_file), folder_name)
    os.makedirs(output_folder, exist_ok=True)

    annotations = coco_data['annotations']
    for annotation in tqdm(annotations):
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        segmentation = annotation['segmentation']
        bbox = annotation['bbox']

        for image in coco_data['images']:
            if image['id'] == image_id:
                image_filename = os.path.basename(image['file_name'])
                image_filename = os.path.splitext(image_filename)[0] # Removing the extension. (In our case, it is the .jpg or .png part.)
                image_width = image['width']
                image_height = image['height']
                break

        x_center = (bbox[0] + bbox[2] / 2) / image_width
        y_center = (bbox[1] + bbox[3] / 2) / image_height
        bbox_width = bbox[2] / image_width
        bbox_height = bbox[3] / image_height

        yolo_segmentation = [f"{(x) / image_width:.5f} {(y) / image_height:.5f}" for x, y in zip(segmentation[0][::2], segmentation[0][1::2])]
        yolo_segmentation = ' '.join(yolo_segmentation)

        yolo_annotation = f"{category_id} {yolo_segmentation}"

        output_filename = os.path.join(output_folder, f"{image_filename}.txt")
        with open(output_filename, 'a+') as file:
            file.write(yolo_annotation + '\n')

    print("Conversion completed. YOLO segmentation annotations saved in 'labels' folder.")