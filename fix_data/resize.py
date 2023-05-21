import cv2
import glob
import draw
import json

all_images = glob.glob('data/images/*.jpg')

def rotate(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

def rotate_points(img, json_file = 'images'):
    pass

def resize(img):
    return cv2.resize(img, (800, 1333), interpolation = cv2.INTER_AREA)

def imshow(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

for image_path in all_images:
    image = cv2.imread(image_path)
    h, w = image.shape[:-1]
    if w > h:
        image = rotate(image)
        imshow(image)
        break

