import utm
import cv2
import numpy as np
from math import cos, asin, sqrt, pi


points = {
    "1": {
        "lat": {"degree": 37, "minutes": 28, "seconds": 29.4454},
        "lon": {"degree": 126, "minutes": 53, "seconds": 54.9389},
        "x": 820,
        "y": 147,
    },
    "2": {
        "lat": {"degree": 37, "minutes": 28, "seconds": 31.1112},
        "lon": {"degree": 126, "minutes": 53, "seconds": 55.0381},
        "x": 579,
        "y": 151,
    },
    "3": {
        "lat": {"degree": 37, "minutes": 28, "seconds": 32.2867},
        "lon": {"degree": 126, "minutes": 53, "seconds": 55.1798},
        "x": 405,
        "y": 150,
    },
    "4": {
        "lat": {"degree": 37, "minutes": 28, "seconds": 32.3551},
        "lon": {"degree": 126, "minutes": 53, "seconds": 54.1043},
        "x": 406,
        "y": 285,
    },
    "5": {
        "lat": {"degree": 37, "minutes": 28, "seconds": 33.0869},
        "lon": {"degree": 126, "minutes": 53, "seconds": 54.2147},
        "x": 297,
        "y": 283,
    },
    "6": {
        "lat": {"degree": 37, "minutes": 28, "seconds": 33.0174},
        "lon": {"degree": 126, "minutes": 53, "seconds": 56.1910},
        "x": 287,
        "y": 59,
    },
    "7": {
        "lat": {"degree": 37, "minutes": 28, "seconds": 30.5040},
        "lon": {"degree": 126, "minutes": 53, "seconds": 55.7630},
        "x": 654,
        "y": 60,
    },
    "8": {
        "lat": {"degree": 37, "minutes": 28, "seconds": 30.0721},
        "lon": {"degree": 126, "minutes": 53, "seconds": 53.7847},
        "x": 747,
        "y": 276,
    },
    "9": {
        "lat": {"degree": 37, "minutes": 28, "seconds": 30.6156},
        "lon": {"degree": 126, "minutes": 53, "seconds": 55.0242},
        "x": 642,
        "y": 150,
    },
    "10": {
        "lat": {"degree": 37, "minutes": 28, "seconds": 32.4978},
        "lon": {"degree": 126, "minutes": 53, "seconds": 56.061},
        "x": 352,
        "y": 52,
    },
    "11": {
        "lat": {"degree": 37, "minutes": 28, "seconds": 32.0082},
        "lon": {"degree": 126, "minutes": 53, "seconds": 55.1286},
        "x": 432,
        "y": 150,
    },
    "12": {
        "lat": {"degree": 37, "minutes": 28, "seconds": 33.384},
        "lon": {"degree": 126, "minutes": 53, "seconds": 55.1004},
        "x": 233,
        "y": 170,
    },
    "13": {
        "lat": {"degree": 37, "minutes": 28, "seconds": 34.866},
        "lon": {"degree": 126, "minutes": 53, "seconds": 56.547},
        "x": 0,
        "y": 0,
    },
    "14": {
        "lat": {"degree": 37, "minutes": 28, "seconds": 29.046},
        "lon": {"degree": 126, "minutes": 53, "seconds": 56.0862},
        "x": 840,
        "y": 0,
    },
    "15": {
        "lat": {"degree": 37, "minutes": 28, "seconds": 29.3808},
        "lon": {"degree": 126, "minutes": 53, "seconds": 53.3286},
        "x": 840,
        "y": 373,
    },
    "16": {
        "lat": {"degree": 37, "minutes": 28, "seconds": 35.0436},
        "lon": {"degree": 126, "minutes": 53, "seconds": 53.9946},
        "x": 0,
        "y": 373,
    },
}

# 1,4,7,
# train_points = ["2", "3",  "5", "6",  "8"]
# train_points = ["8", "10", "11", "12"]
# train_points = ["5", "6", "7", "8"]
# eval_points = ["1", "2", "3", "4"]
# train_points = ["2", "4", "5", "7",
#                 "8", "9", "11", "12"]
# eval_points = ["1", "3", "6", "10"]
# train_points = ["2", "3", "5", "6", "8"]
# eval_points = ["1", "4", "7"]

# train_points = ["2", "3", "5", "6", "8", "9", "10", "11", "12"]
# eval_points = ["1", "4", "7"]

# train_points = ["1", "13", "14", "15", "8", "16", "5", "9", "4",  "6", ]
# eval_points = ["2", "3", "7",  "10", "11", "12"]


# 4, 8, 7

train_points = [
    "1",
    "3",
    "2",
    "8",
    "7",
]
eval_points = ["4", "5", "6"]
# eval_points = ["5", "8", "7"]


src_list = []
dst_list = []

for i in train_points:
    src_list.append([points[i]["x"], points[i]["y"]])
    lat1 = (
        points[i]["lat"]["degree"]
        + points[i]["lat"]["minutes"] / 60
        + points[i]["lat"]["seconds"] / 3600
    )

    lon1 = (
        points[i]["lon"]["degree"]
        + points[i]["lon"]["minutes"] / 60
        + points[i]["lon"]["seconds"] / 3600
    )

    # utm_coord = utm.from_latlon(lat1, lon1)
    # zone_number = utm_coord[2]
    # zone_letter = utm_coord[3]
    # dst_list.append(utm_coord[:2])
    dst_list.append([lat1, lon1])


h, status = cv2.findHomography(np.array(src_list), np.array(dst_list), cv2.RANSAC, 5.0)
# print(h)


def distance(lat1, lon1, lat2, lon2):
    p = pi / 180
    a = (
        0.5
        - cos((lat2 - lat1) * p) / 2
        + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    )
    return 12742 * asin(sqrt(a)) * 1000


for i in eval_points:
    lat1 = (
        points[i]["lat"]["degree"]
        + points[i]["lat"]["minutes"] / 60
        + points[i]["lat"]["seconds"] / 3600
    )
    lon1 = (
        points[i]["lon"]["degree"]
        + points[i]["lon"]["minutes"] / 60
        + points[i]["lon"]["seconds"] / 3600
    )
    x1 = points[i]["x"]
    y1 = points[i]["y"]
    a = np.asarray([x1, y1, 1])
    p = np.dot(h, a)
    # xp = utm.to_latlon(p[:2][0] / p[2], p[:2][1] / p[2], zone_number, zone_letter)
    # lat_pred = xp[0]
    # lon_pred = xp[1]
    lat_pred = p[0] / p[2]
    lon_pred = p[1] / p[2]

    print("distance", distance(lat1, lon1, lat_pred, lon_pred))
    print("predicted", lat_pred, lon_pred)
    print("ground truth", lat1, lon1)
    print("=" * 60)
