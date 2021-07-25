import os
import cv2
import matplotlib.pyplot as plt

img_path = "/home/sanchit/Downloads/Compressed/second_ann_batch-20210521T094921Z-001/second_ann_batch"
img_size = 256
txt_path = "/media/sanchit/Workspace/idealai/external_pipelines/yolov5/runs/detect/exp23/labels"

classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
]
decoder = {k: v for k, v in enumerate(classes)}


def center_to_x1y1(box):
    x_c = box[0]
    y_c = box[1]
    w = box[2]
    h = box[3]
    x1 = x_c - w // 2
    y1 = y_c - h // 2
    x2 = x_c + w // 2
    y2 = y_c + h // 2
    return [x1, y1, x2, y2]


for i in os.listdir(txt_path):
    txt_file = open(os.path.join(txt_path, i), "r")
    image = cv2.imread(os.path.join(img_path, i.replace("txt", "png")))
    h, w, _ = image.shape
    for b in txt_file.readlines():
        classe = int(b[0])
        box = [int(float(i) * img_size) for i in b.split(" ")[1:]]
        box = center_to_x1y1(box)
        x1 = int(box[0] * w / img_size)
        y1 = int(box[1] * h / img_size)
        x2 = int(box[2] * w / img_size)
        y2 = int(box[3] * h / img_size)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=1)
        image = cv2.putText(
            image,
            decoder[classe],
            (x1 - 2, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 0, 0),
            thickness=1,
        )
    plt.imshow(image)
    plt.show()
