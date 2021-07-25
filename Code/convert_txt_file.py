import os
import cv2

# path to text files produced by detect
path_to_detect_txt = "runs/detect/exp16/labels"
result_path = "./result_evaluation"
txt_file_yolov4 = "/media/sanchit/Workspace/Projects/IIIT_hyd_research_fellowship/Pothole_detection/phase1_collection/test_sets/merged.txt"

os.makedirs(os.path.join(result_path, "detections"), exist_ok=True)
os.makedirs(os.path.join(result_path, "groundtruths"), exist_ok=True)

def center_to_x1y1(box):
    x_c = box[0]
    y_c = box[1]
    w = box[2]
    h = box[3]
    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2
    return [x1, y1, x2, y2]


def convert_bbox(box, img_shape):
    h, w, _ = img_shape
    x1 = int(box[0] * w)
    y1 = int(box[1] * h)
    x2 = int(box[2] * w)
    y2 = int(box[3] * h)
    return [x1, y1, x2, y2]


def process_yolov4_txt(txt_path):
    txt_file = open(txt_path, "r")
    out_dict = {}
    for i in txt_file.readlines():
        out_dict.update(
            {
                i.split(" ")[0]
                .split("/")[-1]
                .replace(".png", ".txt")
                .replace(".jpg", ".txt"): {
                    "path": i.split(" ")[0],
                    "box": [[int(y) for y in x.split(",")] for x in i.split(" ")[1:]],
                }
            }
        )
    return out_dict


dict_yolo = process_yolov4_txt(txt_file_yolov4)
import matplotlib.pyplot as plt
for i in os.listdir(path_to_detect_txt):
    txt_file = open(os.path.join(path_to_detect_txt, i), "r")
    detect_txt_file = open(os.path.join(result_path, "detections", i), "w+")
    ground_truth_text_file = open(os.path.join(result_path, "groundtruths", i), "w+")
    for box in dict_yolo[i]["box"]:
        ground_truth_text_file.write(
            "{} {} {} {} {}\n".format(box[4], box[0], box[1], box[2]-box[0], box[3]-box[1])
        )

    image = cv2.imread(dict_yolo[i]["path"])
    image_shape = image.shape

    for box in txt_file.readlines():
        box = box.split(" ")
        classe = int(box[0])
        conf_score = float(box[5])
        box = [float(i) for i in box[1:5]]
        box = center_to_x1y1(box)
        box = convert_bbox(box, image_shape)
        detect_txt_file.write(
            "{} {} {} {} {} {}\n".format(
                classe, conf_score, box[0], box[1], box[2], box[3]
            )
        )

from object_detection_metrics_calculation.main import get_coco_metrics_from_path
_,all = get_coco_metrics_from_path(result_path)
print(all)