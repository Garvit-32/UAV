from math import cos, asin, sqrt, pi
import statistics
import numpy as np
from itertools import combinations
points = {
    "1": {
        "lat": {"degree": 37, "minutes": 28, "seconds": 29.4454},
        "lon": {"degree": 126, "minutes": 53, "seconds": 54.9389},
        "x": 805,
        "y": 168,
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
        "lat": {"degree": 37, "minutes": 28, "seconds": 33.384},
        "lon": {"degree": 126, "minutes": 53, "seconds": 55.1004},
        "x": 263,
        "y": 166,
    },
    "10": {
        "lat": {"degree": 37, "minutes": 28, "seconds": 33.7764},
        "lon": {"degree": 126, "minutes": 53, "seconds": 55.5966},
        "x": 168,
        "y": 107,
    },
    "11": {
        "lat": {"degree": 37, "minutes": 28, "seconds": 31.9764},
        "lon": {"degree": 126, "minutes": 53, "seconds": 54.7074},
        "x": 461,
        "y": 196,
    },
    "12": {
        "lat": {"degree": 37, "minutes": 28, "seconds": 31.371},
        "lon": {"degree": 126, "minutes": 53, "seconds": 54.3084},
        "x": 560,
        "y": 236,
    }

}

given_img_size = (854, 378)
output_img_size = (1920, 1080)

train_points = ["2", "3", "5", "6", "8", "9", "10", "11", "12"]
eval_points = ["1", "4", "7"]

delta_x = []
delta_y = []

for x in list(combinations(train_points, 2)):
    i, j = x
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
    lat2 = (
        points[j]["lat"]["degree"]
        + points[j]["lat"]["minutes"] / 60
        + points[j]["lat"]["seconds"] / 3600
    )
    lon2 = (
        points[j]["lon"]["degree"]
        + points[j]["lon"]["minutes"] / 60
        + points[j]["lon"]["seconds"] / 3600
    )

    x2 = points[j]["x"]
    y2 = points[j]["y"]

    x = (lat2 - lat1) / (x2 - x1)
    y = (lon2 - lon1) / (y2 - y1)
    delta_x.append(x)
    delta_y.append(y)

# d_x = -1.7201035781872401e-06  # np.mean(delta_x)
# np.mean(delta_y)  # -2.000e-06  #
d_y = d_x = statistics.median(delta_x + delta_y)

# d_x = statistics.median(delta_y)  # -2.000e-06  # np.mean(delta_x)
# d_y = -3.4809782607310256e-06  # np.mean(delta_y)
print(d_y, d_x)
origin_lats = []
origin_longs = []

x2s = [0]
y2s = [0]

for x2, y2 in zip(x2s, y2s):

    for i in train_points:
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
        lat2 = lat1 + (x2 - x1) * d_x
        lon2 = lon1 + (y2 - y1) * d_y
        origin_lats.append(lat2)
        origin_longs.append(lon2)
    origin_lat = np.mean(origin_lats)
    origin_lon = np.mean(origin_longs)


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
    x2 = points[i]["x"]
    y2 = points[i]["y"]
    lat_preds = []
    lon_preds = []
    x1, y1 = 0, 0

    for j in train_points:
        origin_lat = (
            points[j]["lat"]["degree"]
            + points[j]["lat"]["minutes"] / 60
            + points[j]["lat"]["seconds"] / 3600
        )
        origin_lon = (
            points[j]["lon"]["degree"]
            + points[j]["lon"]["minutes"] / 60
            + points[j]["lon"]["seconds"] / 3600
        )
        x1, y1 = points[j]["x"], points[j]["y"]
        lat_preds.append(origin_lat + (x2 - x1) * d_x)
        lon_preds.append(origin_lon + (y2 - y1) * d_y)
    lat_pred = np.mean(lat_preds)
    lon_pred = np.mean(lon_preds)

    # lat_pred = origin_lat + (x2 - x1) * d_x
    # lon_pred = origin_lon + (y2 - y1) * d_y

    # print("distance", distance(lat1, lon1, lat_pred, lon_pred))
    print("ground truth", lat1, lon1)
    print("predicted", lat_pred, lon_pred)
