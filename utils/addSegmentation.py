import json

jsonStr = ''

with open('../dataset/all/train/_annotations.coco.json', 'r+', encoding='UTF-8') as file:
    jsonStr = json.load(file)

with open('../dataset/all/train/_annotations.coco.json', 'w', encoding='UTF-8') as file:
    annotations = jsonStr['annotations']
    for i in range(len(annotations)):
        ann = annotations[i]
        bbox = ann['bbox']
        x = bbox[0]
        y = bbox[1]
        xsize = bbox[2]
        ysize = bbox[3]
        annotations[i]["segmentation"] = [[x, y, x + xsize, y, x + xsize, y + ysize, x, y + ysize]]
    
    jsonStr['annotations'] = annotations
    
    print(jsonStr['annotations'])
    json.dump(jsonStr, file, indent=4)
    file.truncate()