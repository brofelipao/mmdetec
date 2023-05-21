
# Import the necessary libraries
import json
import os
from pycocotools import mask

# Class name
classname = "larvae"

# Path of the json files
path = r"dataset/annotations_points/"
#coco_path = r"data/annotations_coco/_annotations.coco.json" # Nao vou mais usar pois quero separar os arquivos por teste, validacao e treino
test_path = r'dataset/filesJSON/_annotations_test.json'
val_path = r'dataset/filesJSON/_annotations_val.json'
train_path = r'dataset/filesJSON/_annotations_train.json'
all_annotations_path = r'dataset/all/train/_annotations.coco.json'
all_paths = [test_path, val_path, train_path, all_annotations_path]

all_files = [file for file in os.listdir(path) if file.endswith('.json')]
total_of_images = len(all_files)

number_of_test = int(total_of_images * 0.2)
number_of_val = int(total_of_images * 0.3)

test = all_files[0:number_of_test]
val = all_files[number_of_test:number_of_test + number_of_val]
train = all_files[number_of_test + number_of_val:]
all_images = [test, val, train, all_files]


CATEGORIES = [{
            "id": 1,
            "name": "larvae",
            "supercategory": "larvae"
        }]

LICENSES = [
    {
        "id": 1,
        "name": "",
        "url": ""
    }
]


def write_json(files):
    # Create an empty dictionary to store the data in the COCO format
    coco = {}
    # Define the keys and values of the coco dictionary according to the COCO format documentation
    # https://cocodataset.org/#format-data
    coco["info"] = {} # Information about the dataset
    coco["licenses"] = LICENSES # Information about the licenses of the images
    coco["images"] = [] # Information about the images
    coco["annotations"] = [] # Information about the annotations of the bounding boxes of the objects
    coco["categories"] = CATEGORIES # Information about the categories of the objects
    # Iterate over all the .json files in the current folder
    for file in files:
        if file.endswith(".json"):
            # Get the full path of the file
            json_file = path + file
            # Open the .json file and load its content as a python object
            with open(json_file, "r") as f:
                content = json.load(f)

                # For each .json file, assume that there is a corresponding image with the same name, but with .jpg extension
                # For example, if the .json file is called "image1.jpg.json", the corresponding image is called "image1.jpg"
                # Get the name of the image without the extension
                image_name = file[:-9]

                # Create a dictionary to store the information of the image
                image = {}
                # Define the id of the image as the name of the image without the extension
                image["id"] = image_name[1:]
                # Define the file name of the image as the name of the image with .jpg extension
                image["file_name"] = image_name + ".jpg"
                image["coco_url"] = ""
                image["flickr_url"] = ""

                # Get the width and height of the image
                image["width"], image["height"] = content['size']['width'], content['size']['height']

                bbsize = int(int(image['width']) * 0.01)
                if (image["width"] > image["height"]):
                    bbsize = int(int(image['height']) * 0.01)

                # Define the license of the image as 1 (can be changed according to the actual license)
                image["license"] = 1
                # Add the dictionary of the image to the list of images of the coco dictionary
                coco["images"].append(image)

                # For each .json file, iterate over the bounding boxes of annotated objects
                for box in content["objects"]:
                    # Create a dictionary to store information of annotation
                    annotation = {}
                    # Define id of image of annotation as id of corresponding image
                    annotation["image_id"] = image["id"]
                    # Define category id of annotation as label of bounding box
                    annotation["category_id"] = 1

                    x = int(box["points"]["exterior"][0][0]) - bbsize//2
                    y = int(box["points"]["exterior"][0][1]) - bbsize//2

                    # Define area of annotation as product of width and height of bounding box
                    annotation["area"] = bbsize*bbsize
                    # Define bounding box of annotation as a list of four numbers: [x, y, width, height]
                    # x, y is the point, so we get the left top corner of bbox
                    annotation["bbox"] = [x, y, bbsize,bbsize]
                    # Define iscrowd of annotation as 0 (can be changed according to different logic)
                    annotation["iscrowd"] = 0
                    # Add dictionary of annotation to list of annotations of coco dictionary
                    # Define id of annotation as a sequential number (can be changed according to different logic)
                    annotation["id"] = len(coco["annotations"]) + 1
                    # Define segmentation of annotation as an empty list (can be changed according to different logic)
                    annotation["segmentation"] = [[x, y, x + bbsize, y, x + bbsize, y + bbsize, x, y + bbsize]]
                    coco["annotations"].append(annotation)
    return coco

# Open a .json file called _annotations.coco.json in write mode
for i in range(4):
    with open(all_paths[i], 'w') as f:
        print(f'As informacoes de {len(all_images[i])} imagens foram armazenadas em {all_paths[i]}')

        # Write content of coco dictionary to .json file, formatted with indentation of 4 spaces
        coco = write_json(all_images[i])
        json.dump(coco, f, indent=4)

print('Finalizado com sucesso!')
