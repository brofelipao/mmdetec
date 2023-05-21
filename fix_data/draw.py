import cv2

def coco_annotation(img, points):
    for point in points:
        xi = int(point[0])
        yi = int(point[1])
        xf = xi + int(point[2])
        yf = yi + int(point[3])
        cv2.rectangle(img, (xi, yi), (xf, yf), (0, 0, 255), 1)

if __name__ == "__main__":
    import json, os

    images_path = r'data/images'
    tes = r'data/annotations_coco/_annotations_test.json'
    val = r'data/annotations_coco/_annotations_val.json'
    tra = r'data/annotations_coco/_annotations_train.json'

    jsons = [tes, val, tra]

    for i in jsons:
        with open(i) as f:
            content = json.load(f)
        
        images = content['images']
        annotations = content['annotations']

        for image in images:
            image_id = image['id']
            image_name = image['file_name']
            image_path = os.path.join(images_path, image_name)

            points = [ann['bbox'] for ann in annotations if ann['image_id'] == image_id]

            img_cv2 = cv2.imread(image_path)
            image_result = coco_annotation(img_cv2, points)

            width = int(img_cv2.shape[1] * 0.3)
            height = int(img_cv2.shape[0] * 0.3)

            image_result = cv2.resize(image_result, (width, height), interpolation=cv2.INTER_AREA)
            cv2.imshow(image_name, image_result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # for image in images:
    #     Draw().coco_annotation(image)