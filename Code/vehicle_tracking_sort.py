import os
import cv2
import argparse
import numpy as np
import torch
from PIL import Image
import scipy.spatial.distance as dist
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import (
    check_img_size,
    check_requirements,
    check_imshow,
    non_max_suppression,
    apply_classifier,
    scale_coords,
    xyxy2xywh,
    strip_optimizer,
    set_logging,
    increment_path,
    save_one_box,
)
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from sort.sort import *

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")

colors = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 0, 0), 3: (255, 255, 0)}


def detect_object_yolov5(model, image, count):
    """accepts yolov4 model and current frame and returns bounding boxes in format x1,y1, x2,y2, score

    Args:
        model ([type]): [description]
        image ([type]): [description]

    Returns:
        [type]: [description]
    """
    ori_shape = image.shape[:2]
    image_ori = image.copy()
    image, _, _ = letterbox(image, new_shape=(960, 960))
    plt.imshow(image)
    plt.show()
    new_shape = image.shape[:2]
    img = torch.from_numpy(image.copy()).to("cuda").permute(2, 0, 1)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img, augment=False)[0]
    pred = non_max_suppression(
        pred,
        0.4,
        0.1,
        max_det=1000,
    )[0]
    scores = list(pred[:, 4].cpu().numpy())
    classes = list(pred[:, 5].cpu().numpy())
    scaled_coords = list(
        scale_coords(new_shape, pred[:, :4], ori_shape).round().cpu().numpy()
    )
    scaled_coords = np.array(
        [
            np.array(list(i) + [s] + [c])
            for i, s, c in zip(scaled_coords, scores, classes)
        ]
    )
    print(scaled_coords)
    import ipdb

    ipdb.set_trace()
    return scaled_coords, count, classes


def frame_extract(path):
    vidObj = cv2.VideoCapture(path)
    success = 1
    while success:
        success, image = vidObj.read()
        if success:
            yield image


def ymean(centroids):
    return np.mean([x[1] for x in centroids[:-1]])


import matplotlib.pyplot as plt


def get_centroid(boxes):
    return (int((boxes[0] + boxes[2]) / 2), int((boxes[1] + boxes[3]) / 2))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-p",
        "--path_to_video",
        required=True,
        help="path to Caffe 'deploy' prototxt file",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="",
    )
    args = ap.parse_args()

    yolo = (
        attempt_load("runs/train/exp11/weights/best.pt", map_location="cpu")
        .eval()
        .cuda()
    )
    # weights = torch.load("runs/train/exp27/weights/best.pt")
    # print(
    #     "epoch {} training results {} best fitness {}".format(
    #         weights["epoch"], weights["training_results"], weights["bestfitness"]
    #     )
    # )
    counted_object_ids = {}
    flag = 0
    count1 = 0
    current_video = cv2.VideoCapture(args.path_to_video)
    fps = current_video.get(cv2.CAP_PROP_FPS)
    out_video = cv2.VideoWriter(
        os.path.join(args.path_to_video.split("/")[-1][:-3] + "avi"),
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        fps,
        (
            1920,
            1080,
        ),  # change it to width,height if there is no cropping operation performed on image
    )
    from tqdm import tqdm

    for idx, frame in tqdm(enumerate(frame_extract(args.path_to_video))):
        if flag == 0:
            H, W, _ = frame.shape
            tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.4)
            ypos = 3 * frame.shape[0] // 4
            flag = 1
        bboxes, count1, classes = detect_object_yolov5(yolo, frame[:, :, ::-1], count1)
        trackers = tracker.update(bboxes)
        if args.debug:

            for i, c in zip(trackers, classes):
                cv2.rectangle(
                    frame,
                    (int(i[0]), int(i[1])),
                    (int(i[2]), int(i[3])),
                    (255, 255, 0),
                    2,
                )
                centroid = get_centroid(i)
                cv2.putText(
                    frame,
                    str(int(i[4])),
                    (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                )
            # for k, v in trackers.items():
            #     centroid = v["centroids"][-1]

            #     cv2.putText(
            #         frame,
            #         str(k),
            #         (centroid[0] - 10, centroid[1] - 10),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         0.5,
            #         (0, 0, 0),
            #         2,
            #     )
            #     # cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 0), -1)
            # for bbox in bboxes:
            #     cv2.rectangle(
            #         frame,
            #         (bbox[0], bbox[1]),
            #         (bbox[2], bbox[3]),
            #         colors[bbox[5]],
            #         2,
            #     )

            out_video.write(frame)
            # cv2.imshow("video", frame)
            # key = cv2.waitKey(1) & 0xFF
            # if key == ord("q"):
            #     break

    out_video.release()
    # print(len(counted_object_ids))


if __name__ == "__main__":
    main()