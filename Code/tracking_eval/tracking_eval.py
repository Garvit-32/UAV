from pymot import MOTEvaluation
import json
import os
import pandas as pd
from formatchecker import FormatChecker
import numpy as np
import argparse

ap = argparse.ArgumentParser()

ap.add_argument(
    "-p",
    "--path_to_yolo_id",
    required=True,
    type=str,
    help="Path to yolo id ground truth folder",
)
ap.add_argument(
    "-v",
    "--video_name",
    required=True,
    type=str,
    help="name of the video without extension",
)
ap.add_argument(
    "--iou",
    default=0.5,
    type=float,
    help="IoU threshold between 0-1",
)
args = ap.parse_args()


path_to_yolo_txt = args.path_to_yolo_id

# 2 car 5 bus 7 trucks
path_to_predicted_csv = "Output/" + args.video_name + "_deep_sort_tracking.csv"
csv_data = pd.read_csv(path_to_predicted_csv)
image_size = (1920, 1080)

# txt format xc,yc,w,h
groundtruths = {"frames": [], "class": "video", "filename": "./a"}

for i in sorted(os.listdir(path_to_yolo_txt)):
    if i.endswith("txt"):
        data = open(os.path.join(path_to_yolo_txt, i), "r")
        frame_id = float(i.replace(".txt", ""))
        dict_single_frame = {
            "timestamp": frame_id / 30,
            "num": 0,
            "class": "video",
            "annotations": [],
        }

        for j in data.readlines():
            dict_single_frame["annotations"].append(
                {
                    "dco": False,
                    "height": int(np.round(float(j.split(" ")[5]) * image_size[1])),
                    "width": int(np.round(float(j.split(" ")[4]) * image_size[0])),
                    "id": j.split(" ")[0]+'_'+j.split(" ")[1],
                    "y": int(np.round(float(j.split(" ")[3]) * image_size[1]
                                      - float(j.split(" ")[5]) * image_size[1] / 2)),
                    "x": int(np.round(float(j.split(" ")[2]) * image_size[0]
                                      - float(j.split(" ")[4]) * image_size[0] / 2)),
                }
            )
        groundtruths["frames"].append(dict_single_frame)

hypotheses = {"frames": [], "class": "video", "filename": ""}

for j in sorted(csv_data['frame id'].unique()):
    dict_single_frame = {
        "timestamp": j / 30,
        "class": "video",
        "hypotheses": [],
    }
    for i in csv_data.iloc[:, :].values:
        if i[0] == j:
            dict_single_frame["hypotheses"].append(
                {
                    "dco": False,
                    "height": i[6]-i[4],
                    "width": i[5]-i[3],
                    "id": f'{i[2]}_{i[1]}',
                    "y": i[4],
                    "x": i[3],
                }
            )
    hypotheses["frames"].append(dict_single_frame)

groundtruths['frames'] = groundtruths['frames'][2:]


formatChecker = FormatChecker(groundtruths, hypotheses)
success = formatChecker.checkForExistingIDs()
success |= formatChecker.checkForAmbiguousIDs()
success |= formatChecker.checkForCompleteness()


evaluator = MOTEvaluation(groundtruths, hypotheses, overlap_threshold=args.iou)
print("Results for IoU threshold: 0.5")
print('MOTA:', evaluator.getMOTA())
print('MOTP:', evaluator.getMOTP())
evaluator.printResults()
