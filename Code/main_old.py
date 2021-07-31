from enum import unique

import os
import cv2
import gc
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
from math import cos, asin, sqrt, pi
from movestar.python.movestar import movestar
import pandas as pd
import csv
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from sort.sort import *
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

import matplotlib.pyplot as plt
import matplotlib

import warnings
warnings.filterwarnings('ignore')

matplotlib.use("TkAgg")

colors = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 0, 0), 3: (255, 255, 0)}
decoder = {0: "car", 1: "truck", 2: "bus", 3: "heavy_truck"}


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
    pred = model(img, augment=True)[0]
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
    return scaled_coords, count


def frame_extract(path):
    vidObj = cv2.VideoCapture(path)
    success = 1
    while success:
        success, image = vidObj.read()
        if success:
            yield image


def ymean(centroids):
    return np.mean([x[1] for x in centroids[:-1]])


class count_object(object):
    def __init__(
        self,
        disappear_count=70,
        n_classes=4,
        corner_percent=10,
        image_size=(1080, 1920),
    ):
        self.disappear_count = disappear_count
        self.track_id_centroid = {}
        self.class_count = {class_id: 0 for class_id in range(n_classes)}
        self.counted_track_ids = {}

        corner_strips = [
            ((0, image_size[1]),
             (0, int(corner_percent * image_size[0] / 100))),
            (
                (0, image_size[1]),
                (
                    image_size[0] - int(corner_percent * image_size[0] / 100),
                    image_size[0],
                ),
            ),
            ((0, int(corner_percent *
                     image_size[1] / 100)), (0, image_size[0])),
            (
                (
                    image_size[1] - int(corner_percent * image_size[1] / 100),
                    image_size[1],
                ),
                (0, image_size[0]),
            ),
        ]
        self.corner_centroid_list = []
        for i in corner_strips:
            corner_strip_list = []
            for x in range(i[0][0], i[0][1]):
                for y in range(i[1][0], i[1][1]):
                    corner_strip_list.append((x, y))
            self.corner_centroid_list.append(corner_strip_list)

    def update_tracker(self, trackers, frame_idx):
        unique_ids = []
        current_counted_objects = {}
        for i in trackers:
            unique_ids.append(i[4])
            if i[4] in self.track_id_centroid.keys():
                self.track_id_centroid[i[4]]["centroids"].append(
                    self.get_centroid(i[:4])
                )
                self.track_id_centroid[i[4]]["frame_indexes"].append(frame_idx)

                self.track_id_centroid[i[4]]["distance"].append(
                    self.get_distance(self.track_id_centroid[i[4]]["centroids"][-1],
                                      self.track_id_centroid[i[4]]["centroids"][-2])
                )

                self.track_id_centroid[i[4]]["speeds"].append(
                    self.get_speed(
                        self.track_id_centroid[i[4]]["centroids"][-1],
                        self.track_id_centroid[i[4]]["centroids"][-2],
                        (
                            self.track_id_centroid[i[4]]["frame_indexes"][-1]
                            - self.track_id_centroid[i[4]]["frame_indexes"][-2]
                        )
                        / fps,
                    )
                )
                if len(self.track_id_centroid[i[4]]["speeds"]) > 2:
                    self.track_id_centroid[i[4]]["acceleration"].append(
                        self.get_acceleration(
                            self.track_id_centroid[i[4]]["speeds"][-2],
                            self.track_id_centroid[i[4]]["speeds"][-1],
                            (
                                self.track_id_centroid[i[4]
                                                       ]["frame_indexes"][-1]
                                - self.track_id_centroid[i[4]]["frame_indexes"][-2]
                            )
                            / fps,
                        )
                    )
                else:
                    self.track_id_centroid[i[4]]["acceleration"].append(0)
            else:

                self.track_id_centroid.update(
                    {
                        i[4]: {
                            "centroids": [self.get_centroid(i[:4])],
                            "disappear_count": 0,
                            "class": i[5],
                            "updated_id": i[6],
                            "frame_indexes": [frame_idx],
                            "speeds": [0],
                            "acceleration": [0],
                            "distance": [0],
                        }
                    }
                )

        disappeared_keys = []
        for key in self.track_id_centroid.keys():
            if key in unique_ids:
                continue
            else:
                self.track_id_centroid[key]["disappear_count"] += 1
            if self.track_id_centroid[key]["disappear_count"] >= self.disappear_count:
                current_counted_objects = self.count_object(
                    key, current_counted_objects
                )
                disappeared_keys.append(key)

        for key in disappeared_keys:
            if key in current_counted_objects.keys():
                current_idx_average_speed = self.get_average_speed(
                    self.track_id_centroid[key]["centroids"],
                    (
                        frame_idx
                        - self.track_id_centroid[key]["frame_indexes"][0]
                        - self.disappear_count
                    )
                    / fps,
                )

                current_counted_objects[key].update(
                    {"average_speed": current_idx_average_speed}
                )
                current_counted_objects[key].update(
                    {"total_distance": sum(
                        self.track_id_centroid[key]['distance'])}
                )
                if self.track_id_centroid[key]["acceleration"][2:]:
                    current_counted_objects[key].update(
                        {
                            "average_acceleration": np.mean(
                                self.track_id_centroid[key]["acceleration"][2:]
                            )
                        }
                    )
                else:
                    current_counted_objects[key].update(
                        {
                            "average_acceleration": 0
                        }
                    )

                del self.track_id_centroid[key]
                gc.collect()
        return self.class_count, current_counted_objects

    def count_object(self, object_key, current_counted_object):
        if self.check_corner(self.track_id_centroid[object_key]["centroids"][-1]):
            self.class_count[self.track_id_centroid[object_key]["class"]] += 1

            # self.counted_track_ids.update(
            #     {object_key: self.track_id_centroid[object_key]}
            # )
            current_counted_object.update(
                {object_key: self.track_id_centroid[object_key]}
            )
        return current_counted_object

    def get_average_speed(self, centroids, total_time):
        distance = 0
        for c1, c2 in zip(centroids[:-1], centroids[1:]):
            c1_geo = localize(c1)
            c2_geo = localize(c2)
            distance += self.distance(c1_geo, c2_geo)
        return distance / total_time

    def get_speed(
        self,
        centroid1,
        centroid2,
        time,
    ):
        centroid1_geo = localize(centroid1)
        centroid2_geo = localize(centroid2)

        dist = self.distance(centroid1_geo, centroid2_geo)
        return dist / time

    def get_distance(
        self,
        centroid1,
        centroid2,
    ):
        centroid1_geo = localize(centroid1)
        centroid2_geo = localize(centroid2)

        dist = self.distance(centroid1_geo, centroid2_geo)
        return dist

    def get_acceleration(self, speed1, speed2, time):
        return (speed2 - speed1) / time

    def get_euclidean_distance(self, c1, c2):
        return dist.euclidean(c1, c2)

    def distance(self, c1, c2):
        lat1, lon1 = c1
        lat2, lon2 = c2
        p = pi / 180
        a = (
            0.5
            - cos((lat2 - lat1) * p) / 2
            + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
        )
        return 12742 * asin(sqrt(a)) * 1000

    def get_centroid(self, boxes):
        return (int((boxes[0] + boxes[2]) / 2), int((boxes[1] + boxes[3]) / 2))

    def check_corner(self, centroid):
        for i in self.corner_centroid_list:
            if centroid in i:
                return True
        return False


def get_centroid(boxes):
    return (int((boxes[0] + boxes[2]) / 2), int((boxes[1] + boxes[3]) / 2))


def bbox_rel(xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0], xyxy[2]])
    bbox_top = min([xyxy[1], xyxy[3]])
    bbox_w = abs(xyxy[0] - xyxy[2])
    bbox_h = abs(xyxy[1] - xyxy[3])
    x_c = bbox_left + bbox_w / 2
    y_c = bbox_top + bbox_h / 2
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def write_tracking_txt(txt_file, trackers, frame_idx):

    for i in trackers:
        txt_file.write(
            "{} {} {} {} {} {} {} {}\n".format(
                frame_idx, i[4], i[6], i[0], i[1], i[2], i[3], i[5]
            )
        )
    return txt_file


def write_tracking_csv(counted_objects, decoder):

    csv_file_list = []

    for k, v in counted_objects.items():
        updated_id = v["updated_id"]
        centroids = v["centroids"]
        average_speed = v["average_speed"]
        frame_indexes = v["frame_indexes"]
        speeds = v["speeds"]
        acceleration = v["acceleration"]
        distance_covered = v["distance"]
        average_acceleration = v["average_acceleration"]
        class_id = v["class"]
        total_distance = v["total_distance"]
        for centroid, frame_id, speed, acceleration, distance in zip(
            centroids, frame_indexes, speeds, acceleration, distance_covered
        ):
            localized_centroid = localize(centroid)
            csv_file_list.append(
                [
                    frame_id,
                    k,
                    updated_id,
                    centroid[0],
                    centroid[1],
                    localized_centroid[0],
                    localized_centroid[1],
                    decoder[class_id],
                    speed,
                    acceleration,
                    distance,
                    average_speed,
                    average_acceleration,
                    total_distance,
                ]
            )
    return csv_file_list


def localize(
    centroid,
    origin_coords=(37.47646052806184, 126.89894687996158),
    pixel_delta=(-2.00e-06, -2.00e-06),
    image_shape=(1080, 1920),
):

    return (
        centroid[0] * pixel_delta[0] * 854 / image_shape[1] + origin_coords[0],
        centroid[1] * pixel_delta[1] * 378 / image_shape[0] + origin_coords[1],
    )


# output txt format:
# <frame idx> <target id> <target id updated> centroid_x centroid_y localized_centroid_x localized_centroid_y <category>


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-p",
        "--path_to_video",
        # required=True,
        # default='/home/hack/dl_ws/freelancing/UAV_code/DJI_0004_gt.mp4',
        help="path to Caffe 'deploy' prototxt file",
    )
    ap.add_argument(
        "--debug",
        action="store_false",
        help="",
    )
    ap.add_argument(
        "--no-txt",
        action="store_false",
        help="",
    )
    args = ap.parse_args()

    yolo = (
        attempt_load(
            "weights/best.pt", map_location="cpu")
        .eval()
        .cuda()
    )

    flag = 0
    count1 = 0
    current_video = cv2.VideoCapture(args.path_to_video)
    global fps
    fps = round(current_video.get(cv2.CAP_PROP_FPS))
    length = int(current_video.get(cv2.CAP_PROP_FRAME_COUNT))
    decoder = {0: "car", 1: "truck", 2: "bus", 3: "heavy_truck"}
    corner_percent = 10
    out_video = cv2.VideoWriter(
        os.path.join('Output', args.path_to_video.split(
            "/")[-1][:-4] + "_deep_sort" + ".avi"),
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        fps,
        (
            1920,
            1080,
        ),  # change it to width,height if there is no cropping operation performed on image
    )
    # out_txt_file = open(
    #     args.path_to_video.split("/")[-1][:-4] + "deep_sort" + ".txt", "w+"
    # )
    csv_file_list_full = []
    csv_file_list_full1 = []
    from tqdm import tqdm

    flush_count = 0
    flush_count_thresh = 50
    for idx, frame in tqdm(enumerate(frame_extract(args.path_to_video)), total=length):
        if flag == 0:
            current_counted_object_old = {}
            H, W, _ = frame.shape
            cfg = get_config()
            cfg.merge_from_file(
                "Code/deep_sort_pytorch/configs/deep_sort.yaml")
            deepsort = DeepSort(
                cfg.DEEPSORT.REID_CKPT,
                max_dist=cfg.DEEPSORT.MAX_DIST,
                min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                max_age=cfg.DEEPSORT.MAX_AGE,
                n_init=cfg.DEEPSORT.N_INIT,
                nn_budget=cfg.DEEPSORT.NN_BUDGET,
                num_classes=cfg.DEEPSORT.N_classes,
                use_cuda=True,
            )
            object_counter = count_object(
                cfg.DEEPSORT.MAX_AGE, cfg.DEEPSORT.N_classes, corner_percent
            )
            flag = 1
        bboxes, count1 = detect_object_yolov5(yolo, frame[:, :, ::-1], count1)

        if len(bboxes):
            xyxys = bboxes[:, :4]
            scores = bboxes[:, 4]
            classes = bboxes[:, 5]
            bbox_xywh = []
            confs = []
            updated_classes = []
            for xyxy, conf, cls in zip(xyxys, scores, classes):
                x_c, y_c, bbox_w, bbox_h = bbox_rel(list(xyxy))
                obj = [x_c, y_c, bbox_w, bbox_h]
                bbox_xywh.append(obj)
                confs.append([conf])
                updated_classes.append([cls])
            xywhs = torch.Tensor(bbox_xywh)
            confss = torch.Tensor(confs)
            updated_classes = torch.Tensor(updated_classes)

            trackers = deepsort.update(
                xywhs, confss, frame.copy(), updated_classes)

            class_wise_count, current_counted_object = object_counter.update_tracker(
                trackers, idx
            )

            if len(current_counted_object) > 0:
                csv_file_list_full += write_tracking_csv(
                    current_counted_object, decoder=decoder
                )
                if flush_count > flush_count_thresh:
                    current_counted_object_old = current_counted_object
                else:
                    flush_count = 0
                    current_counted_object_old.update(current_counted_object)
        else:
            deepsort.increment_ages()
        flush_count += 1

        if args.debug:
            cv2.line(
                frame,
                (int(1920 * corner_percent / 100), 0),
                (int(1920 * corner_percent / 100), 1080),
                color=(255, 0, 0),
                thickness=2,
            )  # vertical right
            cv2.line(
                frame,
                (0, int(1080 * corner_percent / 100)),
                (1920, int(1080 * corner_percent / 100)),
                color=(255, 0, 0),
                thickness=2,  # horizontal bottom
            )
            cv2.line(
                frame,
                (1920 - int(1920 * corner_percent / 100), 0),
                (1920 - int(1920 * corner_percent / 100), 1080),
                color=(255, 0, 0),
                thickness=2,
            )  # vertical left
            cv2.line(
                frame,
                (0, 1080 - int(1080 * corner_percent / 100)),
                (1920, 1080 - int(1080 * corner_percent / 100)),
                color=(255, 0, 0),
                thickness=2,
            )  # horizontal top

            line = "Counter: "

            for k, v in class_wise_count.items():
                line += decoder[k] + " "
                line += str(v) + " "

            cv2.putText(
                frame,
                line,
                (25, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 0),
                2,
            )
            # line_counted_objects = "Current counted_object: "
            # # for k, v in current_counted_object_old.items():
            # #     line_counted_objects += "{} {} ".format(
            # #         decoder[v["class"]], v["updated_id"]
            # #     )

            cv2.putText(
                frame,
                str(idx),
                (55, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                2,
            )

            if trackers is None:
                print(True)

            for i in trackers:
                csv_file_list_full1.append(
                    [idx, int(i[6]), i[5], int(i[0]), int(
                        i[1]), int(i[2]), int(i[3])]
                )
                color = colors[i[5]]
                cv2.rectangle(
                    frame,
                    (int(i[0]), int(i[1])),
                    (int(i[2]), int(i[3])),
                    color,
                    2,
                )

                cv2.putText(
                    frame,
                    str(int(i[6])),
                    (int(i[2]), int(i[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
            out_video.write(frame)

    with open(
        'Output/' +
            args.path_to_video.split(
                "/")[-1][:-4] + "_deep_sort_tracking" + ".csv",
        "w",
    ) as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(
            ["frame id", "tracking id", "class", "x1", "y1", "x2", "y2"])

        # writing the data rows
        csvwriter.writerows(csv_file_list_full1)

    decoder = {"car": 1, "truck": 1, "bus": 2, "heavy_truck": 2}

    rate_list = []
    factor_list = []
    for i in csv_file_list_full:
        id_out = movestar(decoder[i[7]], [i[8]])
        rate_list.append(list(i) + id_out["Emission Rate"][0])
        factor_list.append(list(i) + id_out["Emission Factor"][0])

    with open('Output/' + args.path_to_video.split("/")[-1][:-4] + "_deep_sort_rate_per_frame" + ".csv", "w") as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(
            [
                "frame id",
                "tracking id main",
                "tracking id class wise",
                "centroid x",
                "centroid y",
                "localized centroid x",
                "localized centroid y",
                "category",
                "speed",
                "acceleration",
                "distance covered",
                "average speed",
                "average acceleration",
                "total distance",
                "CO(g/m)",
                "HC(g)",
                "NOx(g)",
                "PM2.5_Ele(g)",
                "PM2.5_Org(g)",
                "Energy(KJ)",
                "CO2(g)",
                "Fuel(g)",
                "TT(s)",
            ]
        )

        # writing the data rows
        csvwriter.writerows(rate_list)

    with open('Output/' + args.path_to_video.split("/")[-1][:-4] + "_deep_sort_factor_per_frame" + ".csv", "w") as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(
            [
                "frame id",
                "tracking id main",
                "tracking id class wise",
                "centroid x",
                "centroid y",
                "localized centroid x",
                "localized centroid y",
                "category",
                "speed",
                "acceleration",
                "distance covered",
                "average speed",
                "average acceleration",
                "total distance covered",
                "CO(g/mi)",
                "HC(g/mi)",
                "NOx(g/mi)",
                "PM2.5_Ele(g/mi)",
                "PM2.5_Org(g/mi)",
                "Energy(KJ/mi)",
                "CO2(g/mi)",
                "Fuel(g/mi)",
                "TD(mi)",
            ]
        )

        # writing the data rows
        csvwriter.writerows(factor_list)

    df = pd.read_csv('Output/' + args.path_to_video.split("/")
                     [-1][:-4] + "_deep_sort_rate_per_frame" + ".csv")

    decoder = {"car": 1, "truck": 1, "bus": 2, "heavy_truck": 2}
    sec_list = [i for i in range(1, length+1) if i % fps == 0]
    main_dict = {j: [] for j in sorted(df['tracking id main'].unique())}

    for i in df.values:
        main_dict[i[1]].append(i)

    rate_list = []
    factor_list = []
    sec_back = 0
    for sec in sec_list:
        for i in main_dict.keys():
            y = 0
            speed_list = []
            dist_list = []
            for x, j in enumerate(main_dict[i]):
                speed_list.append(j[8])
                dist_list.append(j[10])
                if sec_back < j[0] <= sec:
                    y = x

            if y != 0:
                id_out = movestar(decoder[j[7]], speed_list[:y+1])
                rate_list.append([sec_list.index(sec)+1] +
                                 main_dict[i][y].tolist()[3:10] + [sum(dist_list[:y+1])] + main_dict[i][y].tolist()[11:14] + id_out['Emission Rate'][0])
                factor_list.append([sec_list.index(sec)+1] +
                                   main_dict[i][y].tolist()[3:10] + [sum(dist_list[:y+1])] + main_dict[i][y].tolist()[11:14] + id_out['Emission Factor'][0])
        sec_back = sec

    with open('Output/' + args.path_to_video.split("/")[-1][:-4] + "_deep_sort_rate_per_sec" + ".csv", "w") as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(
            [
                "sec",
                # "frame id",
                # "tracking id main",
                # "tracking id class wise",
                "centroid x",
                "centroid y",
                "localized centroid x",
                "localized centroid y",
                "category",
                "speed",
                "acceleration",
                "distance covered",
                "average speed",
                "average acceleration",
                "total distance",
                "CO(g)",
                "HC(g)",
                "NOx(g)",
                "PM2.5_Ele(g)",
                "PM2.5_Org(g)",
                "Energy(KJ)",
                "CO2(g)",
                "Fuel(g)",
                "TT(s)",
            ]
        )

        csvwriter.writerows(rate_list)

    with open('Output/' + args.path_to_video.split("/")[-1][:-4] + "_deep_sort_factor_per_sec" + ".csv", "w") as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(
            [
                "sec",
                # "frame id",
                # "tracking id main",
                # "tracking id class wise",
                "centroid x",
                "centroid y",
                "localized centroid x",
                "localized centroid y",
                "category",
                "speed",
                "acceleration",
                "distance covered",
                "average speed",
                "average acceleration",
                "total distance",
                "CO(g/mi)",
                "HC(g/mi)",
                "NOx(g/mi)",
                "PM2.5_Ele(g/mi)",
                "PM2.5_Org(g/mi)",
                "Energy(KJ/mi)",
                "CO2(g/mi)",
                "Fuel(g/mi)",
                "TD(mi)",

            ]
        )

        csvwriter.writerows(factor_list)

    out_video.release()
    # print(object_counter.counted_track_ids)

    # print(len(counted_object_ids))


if __name__ == "__main__":
    main()
