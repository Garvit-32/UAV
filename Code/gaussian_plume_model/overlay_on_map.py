import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.io as sio


def overlay_on_map():

    # Overlay concentrations on map
    # plt.ion()
    mat_contents = sio.loadmat('map_green_lane')
    plt.figure()
    plt.imshow(mat_contents['A'],
               extent=[np.min(mat_contents['ximage']),
               np.max(mat_contents['ximage']),
               np.min(mat_contents['yimage']),
               np.max(mat_contents['yimage'])])

    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    # cs = plt.contour(x, y, np.mean(C1, axis=2)*1e6, cmap='hot')
    # plt.clabel(cs, cs.levels, inline=True, fmt='%.1f', fontsize=5)
    plt.show()
    # return


if __name__ == "__main__":
    overlay_on_map()
