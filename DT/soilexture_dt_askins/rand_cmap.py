import colorsys
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def rand_cmap(nlabels, type='bright', first_color_black=True,
              last_color_black=False, verbose=False):

    if type == 'bright':
        randHSVcolors = [
            (np.random.uniform(0.0, 1.0),
             np.random.uniform(0.2, 1.0),
             np.random.uniform(0.9, 1.0))
            for _ in range(nlabels)
        ]
    elif type == 'soft':
        randHSVcolors = [
            (np.random.uniform(0.0, 1.0),
             np.random.uniform(0.2, 0.5),
             np.random.uniform(0.5, 0.8))
            for _ in range(nlabels)
        ]
    else:
        raise ValueError('type must be "bright" or "soft"')

    randRGBcolors = [colorsys.hsv_to_rgb(*c) for c in randHSVcolors]

    if first_color_black:
        randRGBcolors[0] = (0.0, 0.0, 0.0)
    if last_color_black:
        randRGBcolors[-1] = (0.0, 0.0, 0.0)

    if verbose:
        print('Number of labels: ' + str(nlabels))

    return LinearSegmentedColormap.from_list('rand_cmap', randRGBcolors, N=nlabels)
