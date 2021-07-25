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


class CentroidTracker:
    def __init__(
        self,
        image_size,
        maxDisappearedframes=20,
        centroid_distance_threshold=15,
        roi=[0, 1],
    ):
        """constructor
        Args:
            maxDisappearedframes (int, optional): [number of frames to keep the object id saved after it disappeared]. Defaults to 50.
            centroid_distance_threshold (int, optional): [euclidean distance between two objects for which it is same object]. Defaults to 10.
        """
        self.max_disappeared = maxDisappearedframes
        self.centroid_distance_threshold = centroid_distance_threshold
        self.objects_tracked = (
            {}
        )  # key will be the object id, values will be list of all the centroids and number of frames for which it is disappeared
        self.count = 0
        self.roi = [image_size * roi[0], image_size * roi[1]]

    def register(self, centroid):
        if self.roi[0] < centroid[1] < self.roi[1]:
            self.objects_tracked.update(
                {self.count: {"centroids": [centroid], "disappeared_count": 0}}
            )
            self.count += 1

    def deregister(self, objectid):
        self.disappeared_object[objectid] = self.objects_tracked[objectid]
        del self.objects_tracked[objectid]

    def update(self, boxes):
        self.disappeared_object = {}
        centroids = self.get_centroid(boxes)
        if len(boxes) == 0:
            for object_id in list(self.objects_tracked.keys()):
                self.objects_tracked[object_id]["disappeared_count"] += 1

                if (
                    self.objects_tracked[object_id]["disappeared_count"]
                    > self.max_disappeared
                ):
                    self.deregister(object_id)

            return self.objects_tracked, self.disappeared_object

        if len(self.objects_tracked) == 0:
            for i in centroids:
                self.register(i)
            return self.objects_tracked, self.disappeared_object

        objectIDs = list(self.objects_tracked.keys())
        objectCentroids = [x["centroids"][-1] for _, x in self.objects_tracked.items()]
        D = dist.cdist(np.array(objectCentroids), np.array(centroids))
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        usedRows = set()
        usedCols = set()

        for (row, col) in zip(rows, cols):

            if row in usedRows or col in usedCols:
                continue

            objectID = objectIDs[row]

            self.objects_tracked[objectID]["centroids"].append(centroids[col])
            self.objects_tracked[objectID]["disappeared_count"] = 0
            usedRows.add(row)
            usedCols.add(col)

        # compute both the row and column index we have NOT yet
        # examined
        unusedRows = set(range(0, D.shape[0])).difference(usedRows)
        unusedCols = set(range(0, D.shape[1])).difference(usedCols)

        if D.shape[0] >= D.shape[1]:

            for row in unusedRows:

                objectID = objectIDs[row]
                self.objects_tracked[objectID]["disappeared_count"] += 1
                if (
                    self.objects_tracked[objectID]["disappeared_count"]
                    > self.max_disappeared
                ):
                    self.deregister(objectID)

        else:
            for col in unusedCols:
                self.register(centroids[col])
        return self.objects_tracked, self.disappeared_object

    def get_centroid(self, boxes):
        return [(int((x[0] + x[2]) / 2), int((x[1] + x[3]) / 2)) for x in boxes]


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
    )
    boxes = []
    for i in pred:
        if len(i):
            scaled_coords = list(
                scale_coords(new_shape, i, ori_shape).round().cpu().numpy()
            )
            scaled_coords = [list(i) for i in scaled_coords]

            boxes += scaled_coords
    return boxes, count


def frame_extract(path):
    vidObj = cv2.VideoCapture(path)
    success = 1
    while success:
        success, image = vidObj.read()
        if success:
            yield image


def ymean(centroids):
    return np.mean([x[1] for x in centroids[:-1]])


def count_object(disappeared_object, counted_object_ids, frame_count_to_count=3):
    for k, v in disappeared_object.items():
        if len(disappeared_object[k]["centroids"]) > frame_count_to_count:
            counted_object_ids.update({k: 1})
        # direction = v["centroids"][-1][1] - ymean(v["centroids"])

        # if direction > 0 and v["centroids"][-1][1] > y_pos:
        #     counted_object_ids.update({k: "unload"})

        # if direction < 0 and v["centroids"][-1][1] < y_pos:
        #     counted_object_ids.update({k: "load"})

    return counted_object_ids


import matplotlib.pyplot as plt


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
        os.path.join(args.path_to_video.split("/")[-1][:-4] + "centroid" + ".avi"),
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
            tracker = CentroidTracker(image_size=H)
            ypos = 3 * frame.shape[0] // 4
            flag = 1
        bboxes, count1 = detect_object_yolov5(yolo, frame[:, :, ::-1], count1)
        objects, disappeared_object = tracker.update(bboxes)

        if len(list(disappeared_object.keys())) > 0:
            counted_object_ids = count_object(disappeared_object, counted_object_ids)

        if args.debug:
            # cv2.line(frame, (0, 7 * H // 8), (W, 7 * H // 8), (0, 255, 255), 2)
            # cv2.line(frame, (0, 1 * H // 8), (W, 1 * H // 8), (0, 255, 255), 2)
            # cv2.putText(
            #     frame,
            #     "count: " + str(len(list(counted_object_ids.keys()))),
            #     (7 * W // 16, 7 * H // 8 - 40),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     2.0,
            #     (255, 0, 0),
            #     2,
            # )
            for k, v in objects.items():
                centroid = v["centroids"][-1]

                cv2.putText(
                    frame,
                    str(k),
                    (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                )
                # cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 0), -1)
            for bbox in bboxes:
                cv2.rectangle(
                    frame,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    colors[bbox[5]],
                    2,
                )

            out_video.write(frame)
            # cv2.imshow("video", frame)
            # key = cv2.waitKey(1) & 0xFF
            # if key == ord("q"):
            #     break

    out_video.release()
    # print(len(counted_object_ids))


if __name__ == "__main__":
    main()