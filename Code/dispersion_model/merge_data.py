import os
import cv2
import pickle
import argparse
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


ap = argparse.ArgumentParser()

ap.add_argument(
    '--receptor_coord',
    # required=True,
    help='Coordinates of receptor'
)

ap.add_argument(
    '--data_path',
    required=True,
    type=str,
    help='Path to the pickle file'
)
ap.add_argument(
    '--video_path',
    type=str,
    required=True,
    help='Path of the video'
)

ap.add_argument(
    '--image_path',
    type=str,
    help='Path of the image'
)
ap.add_argument(
    '--clim',
    type=int,
    required=True,
    help='Scale limit of the map'
)

args = ap.parse_args()
if args.receptor_coord is not None:
    split = args.receptor_coord.split(',')
    receptor = (int(float(split[0])), int(float(split[1])))


# try:
pollutant_name = args.data_path.split('/')[-1]
current_video = cv2.VideoCapture(args.video_path)
width = int(current_video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(current_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
shape = (width, height)
xs = np.arange(width)
ys = np.arange(height)
[x, y] = np.meshgrid(xs, ys)


pkl_file = [np.asarray(pickle.load(open(i, 'rb')))
            for i in glob(os.path.join(args.data_path, 'data*.pkl'))]
n_file = [np.asarray(pickle.load(open(i, 'rb')))
          for i in glob(os.path.join(args.data_path, 'n*.pkl'))]

sum1 = np.zeros(pkl_file[0].shape)
sum2 = 0
for pkl, n in zip(pkl_file, n_file):
    sum1 += len(n) * pkl
    sum2 = + len(n)

sum1 /= sum2
sum1 *= 1e6


g = plt.pcolor(x, y, sum1, cmap='jet', shading='nearest')
plt.clim((0, args.clim))
if args.receptor_coord is not None:
    plt.scatter(receptor[0], receptor[1], color='black')
plt.axis('off')
plt.gca().invert_yaxis()
plt.savefig(os.path.join('PlotOutput', f'{pollutant_name}_image_to_merge.png'),
            bbox_inches='tight', pad_inches=0)
# plt.show()


g = plt.pcolor(x, y, sum1, cmap='jet', shading='nearest')
plt.clim((0, args.clim))
plt.title(pollutant_name)
plt.axis('on')
plt.xlabel('x (metres)')
plt.ylabel('y (metres)')
cb1 = plt.colorbar(g)
cb1.set_label('$\mu$ g m$^{-3}$')
if args.receptor_coord is not None:
    plt.scatter(receptor[0], receptor[1], color='black')
# plt.gca().invert_yaxis()
plt.savefig(os.path.join('PlotOutput', f'{pollutant_name}_colorbar.png'))
plt.show()


if args.image_path is not None:
    img1 = cv2.imread(args.image_path)
    img1 = cv2.resize(img1, (width, height))
    img2 = cv2.imread(os.path.join(
        'PlotOutput', f'{pollutant_name}_image_to_merge.png'))
    img2 = cv2.resize(img2, (width, height))
    alpha = 0.5
    added_image = cv2.addWeighted(img1, alpha, img2, 1-alpha, 0)

    cv2.imwrite(os.path.join(
        'PlotOutput', f'{pollutant_name}_merge.png'), added_image)
    cv2.imshow('added_image', added_image)
    cv2.waitKey(0)
