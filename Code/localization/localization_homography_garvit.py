from pyproj import Transformer
from pyproj import Transformer, transformer, Proj, transform
import pyproj
import utm
import cv2
import numpy as np

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
        "lat": {"degree": 37, "minutes": 28, "seconds": 33.7764},
        "lon": {"degree": 126, "minutes": 53, "seconds": 55.5966},
        "x": 168,
        "y": 112,
    },
    "14": {
        "lat": {"degree": 37, "minutes": 28, "seconds": 31.9764},
        "lon": {"degree": 126, "minutes": 53, "seconds": 54.7074},
        "x": 446,
        "y": 200,
    },
    "15": {
        "lat": {"degree": 37, "minutes": 28, "seconds": 31.371},
        "lon": {"degree": 126, "minutes": 53, "seconds": 54.3084},
        "x": 538,
        "y": 235,
    },
}

# 1,4,7,
train_points = ["1", "2", "3", "5", "6", "8"]
eval_points = ["4", "7"]
# train_points = ["8", "10", "11", "12"]
# train_points = ["2", "3", "4", "5", "6", "7",
#                 "8", "9", "11"]

src_list = []
dst_list = []
dst_lat_list = []
dst_lon_list = []

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
    # dst_lat_list.append(lat1)
    # dst_lon_list.append(lon1)


# print(dst_list[0][0], dst_list[0][1])
# print(dst_list[1][0], dst_list[1][1])
# print(dst_list[2][0], dst_list[2][1])

# 314186.95171885344 4149675.7064567567 utm 0


# #
# x = utm.from_latlon(np.array(dst_lat_list), np.array(dst_lon_list))
# print(x)


# x = utm.from_latlon(dst_list[2][0], dst_list[2][1])
# print(type(x[:2]))


# def lonlat_to_xy(lon, lat):
#     proj_latlon = pyproj.Proj(proj='latlong', datum='WGS84')
#     proj_xy = pyproj.Proj(proj="utm", zone=52, datum='WGS84')
#     xy = pyproj.transform(proj_latlon, proj_xy, lon, lat)

#     return xy[0], xy[1]


# P = pyproj.Proj(proj='utm', zone=52, ellps='WGS84', preserve_units=True)
# G = pyproj.Geod(ellps='WGS84')

# def LatLon_To_XY(Lat, Lon):
#     return P(Lat, Lon)


# def XY_To_LatLon(x, y):
#     return P(x, y, inverse=True)

# def xy_to_lonlat(x, y):
#     proj_latlon = pyproj.Proj(proj='latlong', datum='WGS84')
#     proj_xy = pyproj.Proj(proj="utm", zone=52, datum='WGS84')
#     lonlat = pyproj.transform(proj_xy, proj_latlon, x, y)

#     return lonlat[0], lonlat[1]


# print(XY_To_LatLon(820, 147))
# print(XY_To_LatLon(406, 285))
# print(XY_To_LatLon(654, 60))
# print(XY_To_LatLon(0, 0))
# print(XY_To_LatLon(dst_list[0][0], dst_list[0][1]))

# dst_list = []

# for x, y in zip(yy, xx):
#     dst_list.append([x, y])


# print(zone_number)
# print(xx, yy)

# (314186.95171885344, 4149675.7064567567), (314191.241458988, 4149711.860837194)

# print(np.asarray(dst_list)/1000)
# print(np.array(dst_list).astype(np.int32))


h, status = cv2.findHomography(np.array(src_list), np.array(dst_list), cv2.RANSAC, 5.0)
print(h)


a1 = np.asarray([820, 147, 1])
a2 = np.asarray([579, 151, 1])
a3 = np.asarray([820, 147, 1])
a4 = np.asarray([406, 285, 1])
a5 = np.asarray([297, 283, 1])
a6 = np.asarray([287, 59, 1])
a7 = np.asarray([654, 60, 1])

p = np.dot(h, a1)
print(p)
# xp = utm.to_latlon(p[:2][0], p[:2][1], zone_number, zone_letter)
# print(xp)
# p1 = np.matmul(h, np.asarray([820, 147, 1]))
# p2 = np.matmul(h, np.asarray([406, 285, 1]))
# print(p2)
# # p3 = np.matmul(h, np.asarray([654, 60, 1]))
# p4 = np.matmul(h, np.asarray([263, 166, 1]))

# xp1 = utm.to_latlon(p1[:2][0], p1[:2][1], zone_number, zone_letter)
# print(xp1)

# xp3 = utm.to_latlon(p3[:2][0], p3[:2][1], zone_number, zone_letter)
# print(xp3)
# xp4 = utm.to_latlon(p4[:2][0], p4[:2][1], zone_number, zone_letter)
# print(xp4)
# print(p2[:2])
# print(p1, p2, p3)


# (36.69227314260328, 126.84652385880037) me
# (38.04422974222827, 126.93677405096192) 1
# [ 36.98725014 125.2474058 ]   sanchit

# (40.48071888951883, 127.1050644560258) me
# (39.7057959417568, 127.0505711353907) 4
# [ 41.33515864 139.96719652] sanchit

# (35.54043239595785, 126.77146889153086) me
# (37.711957886385086, 126.91479662850242) 7
# [ 35.51104312 120.24813238] sanchit
