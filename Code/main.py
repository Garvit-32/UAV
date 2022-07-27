import os
import cv2
import gc
import torch
import argparse
import numpy as np
from tqdm import tqdm
import scipy.spatial.distance as dist
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords

from math import cos, asin, sqrt, pi
import csv
from sort.sort import *
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import matplotlib
import ast

import warnings
warnings.filterwarnings('ignore')

matplotlib.use("TkAgg")


colors = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 0, 0), 3: (255, 255, 0)}
decoder = {0: "car", 1: "truck", 2: "bus", 3: "heavy_truck"}
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
os.makedirs('./Output', exist_ok=True)


def detect_object_yolov5(model, image, count):
    """accepts yolov4 model and current frame and returns bounding boxes in format x1,y1, x2,y2, score

    Args:
        model ([type]): [description]
        image ([type]): [description]

    Returns:
        [type]: [description]
    """
    ori_shape = image.shape[:2]  # (720,1280)
    # image_ori = image.copy()
    image, _, _ = letterbox(image, new_shape=(960, 960))

    new_shape = image.shape[:2]  # (544,960)

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
        self.image_size = image_size

        self.corner_strips = [
            (
                (0, 0),
                (int(corner_percent * image_size[1] / 100), image_size[0])
            ),

            (
                (0, 0),
                (image_size[1], int(corner_percent * image_size[0] / 100))
            ),

            (
                (0, image_size[0] - int(corner_percent * image_size[0] / 100)),
                (image_size[1], image_size[0])
            ),

            (
                (image_size[1] - int(corner_percent * image_size[1] /
                                     100), 0),
                (image_size[1], image_size[0])
            ),
        ]

    def update_tracker(self, trackers, frame_idx):
        unique_ids = []
        current_counted_objects = {}
        for i in trackers:
            unique_ids.append(i[4])
            if i[4] in self.track_id_centroid.keys():
                self.track_id_centroid[i[4]]["centroids"].append(
                    self.get_centroid(i[:4])
                )  # x1,y1 shape equal to image (scaled)
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
            c1_geo = localize(c1, origin_coords, pixel_delta,
                              image_shape=self.image_size)
            c2_geo = localize(c2, origin_coords, pixel_delta,
                              image_shape=self.image_size)
            distance += self.distance(c1_geo, c2_geo)
        return distance / total_time

    def get_speed(
        self,
        centroid1,
        centroid2,
        time,
    ):
        centroid1_geo = localize(
            centroid1, origin_coords, pixel_delta, image_shape=self.image_size)
        centroid2_geo = localize(
            centroid2, origin_coords, pixel_delta, image_shape=self.image_size)

        dist = self.distance(centroid1_geo, centroid2_geo)
        return dist / time

    def get_distance(
        self,
        centroid1,
        centroid2,
    ):
        centroid1_geo = localize(
            centroid1, origin_coords, pixel_delta, image_shape=self.image_size)
        centroid2_geo = localize(
            centroid2, origin_coords, pixel_delta, image_shape=self.image_size)

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
        x, y = centroid
        for pt1, pt2 in self.corner_strips:
            if x > pt1[0] and x < pt2[0] and y > pt1[1] and y < pt2[1]:
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


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def write_tracking_csv(counted_objects, decoder, shape):

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
            localized_centroid = localize(
                centroid, origin_coords, pixel_delta, image_shape=shape)
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


def write_csv_tracking(args, csv_file_list_full1, init=False):
    with open(
        'Output/' +
            args.path_to_video.split(
                "/")[-1][:-4] + "_deep_sort_tracking" + ".csv",
        "a",
    ) as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        if init:
            csvwriter.writerow(
                ["frame id", "tracking id", "class", "x1", "y1", "x2", "y2"])

        # writing the data rows
        else:
            csvwriter.writerows(csv_file_list_full1)


def remove(video_name):
    if os.path.exists(os.path.join("Output", video_name + "_deep_sort_tracking.csv")):
        os.remove(os.path.join("Output", video_name + "_deep_sort_tracking.csv"))
    if os.path.exists(os.path.join('Output', video_name + "_deep_sort.avi")):
        os.remove(os.path.join('Output', video_name + "_deep_sort.avi"))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("-p", "--path_to_video", required=True,
                    help="path to Video File", )
    ap.add_argument("--debug", action="store_false",
                    help="Argument to debug the code",)
    ap.add_argument("--corner_percent", default=10, type=int)
    ap.add_argument("-o", "--origin_coords",
                    default="37.47646052806184, 126.89894687996158", type=str)
    ap.add_argument("-pd", "--pixel_delta",
                    default="-2.00e-06, -2.00e-06", type=str)

    args = ap.parse_args()

    global fps
    global origin_coords
    global pixel_delta
    origin_coords = ast.literal_eval(args.origin_coords)
    pixel_delta = ast.literal_eval(args.pixel_delta)

    yolo = attempt_load("weights/best.pt", map_location="cuda").eval().cuda()

    remove(args.path_to_video.split("/")[-1][:-4])

    count = 0
    current_video = cv2.VideoCapture(args.path_to_video)
    width = current_video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = current_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    shape = (int(width), int(height))
    fps = round(current_video.get(cv2.CAP_PROP_FPS))
    length = int(current_video.get(cv2.CAP_PROP_FRAME_COUNT))
    decoder = {0: "car", 1: "truck", 2: "bus", 3: "heavy_truck"}
    corner_percent = args.corner_percent
    out_video = cv2.VideoWriter(
        os.path.join('Output', args.path_to_video.split(
            "/")[-1][:-4] + "_deep_sort" + ".avi"),
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        fps,
        shape,  # change it to width,height if there is no cropping operation performed on image
    )

    trackers_list = []
    # flush_count_thresh = 50

    write_csv_tracking(args, [], init=True)

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
        cfg.DEEPSORT.MAX_AGE, cfg.DEEPSORT.N_classes, corner_percent, shape[::-1]
    )

    for idx, frame in tqdm(enumerate(frame_extract(args.path_to_video)), total=length):

        bboxes, count = detect_object_yolov5(yolo, frame[:, :, ::-1], count)

        if len(bboxes):
            xyxys = bboxes[:, :4]  # scaled with coordinates
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
                write_tracking_csv(current_counted_object,
                                   decoder=decoder, shape=shape[::-1])

                # if flush_count > flush_count_thresh:
                #     current_counted_object_old = current_counted_object
                # else:
                #     flush_count = 0
                #     current_counted_object_old.update(current_counted_object)

        else:
            deepsort.increment_ages()

        if args.debug:
            cv2.line(
                frame,
                (int(shape[0] * corner_percent / 100), 0),
                (int(shape[0] * corner_percent / 100), shape[1]),
                color=(255, 0, 0),
                thickness=2,
            )  # vertical right
            cv2.line(
                frame,
                (0, int(shape[1] * corner_percent / 100)),
                (shape[0], int(shape[1] * corner_percent / 100)),
                color=(255, 0, 0),
                thickness=2,  # horizontal bottom
            )
            cv2.line(
                frame,
                (shape[0] - int(shape[0] * corner_percent / 100), 0),
                (shape[0] - int(shape[0] * corner_percent / 100), shape[1]),
                color=(255, 0, 0),
                thickness=2,
            )  # vertical left
            cv2.line(
                frame,
                (0, shape[1] - int(shape[1] * corner_percent / 100)),
                (shape[0], shape[1] - int(shape[1] * corner_percent / 100)),
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
            # for k, v in current_counted_object_old.items():
            #     line_counted_objects += "{} {} ".format(
            #         decoder[v["class"]], v["updated_id"]
            #     )

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

                trackers_list.append([idx, int(i[6]), i[5], int(i[0]), int(
                    i[1]), int(i[2]), int(i[3])])
                if (idx+1) % 50 == 0:
                    write_csv_tracking(args, trackers_list)
                    trackers_list = []

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

    out_video.release()


if __name__ == "__main__":
    main()
