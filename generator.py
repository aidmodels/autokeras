import os
import json

def main():
    annotation = []
    train_dir = './tiny-imagenet-200/train'
    test_dir = './tiny-imagenet-200/test/images'
    val_dir = './tiny-imagenet-200/val/images'
    for root, dirs, files in os.walk(train_dir):
        for file in files:
            if file.endswith('.JPEG'):
                label = file.split('_')[0]
                filepath = os.path.join(label, 'images', file)
                data = {
                    "folder": "train",
                    "filename": filepath,
                    "class": {
                        "label": label
                    }
                }
                annotation.append(data)
    with open('annotations.json', 'w+') as f:
        f.write(json.dumps(annotation))
if __name__=='__main__':
    main()