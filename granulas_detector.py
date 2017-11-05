import cv2
import imutils
import pandas

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


def detect_granulas(path, show=False):
    granulas_data = {}
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_source = gray.copy()

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=10)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    def init_default_region_property(n):
        prop = list()
        for _ in range(0, n + 1):
            prop.append(0)
        return prop

    squares_regions = init_default_region_property(ret)
    contrasts_regions = init_default_region_property(ret)

    min_x = []
    max_x = []
    min_y = []
    max_y = []
    min_x.append(-1)
    max_x.append(-1)
    min_y.append(-1)
    max_y.append(-1)

    for i_label in range(1, ret + 1):
        region_matrix = markers == i_label
        min_x.append(-1)
        max_x.append(-1)
        min_y.append(-1)
        max_y.append(-1)

        for i in range(0, len(region_matrix)):
            x = region_matrix[i]
            for point_index in range(0, len(x)):
                if x[point_index]:
                    squares_regions[i_label] += 1
                    contrasts_regions[i_label] += img_source[i, point_index]
                    if min_x[i_label] < 0 or min_x[i_label] > point_index:
                        min_x[i_label] = point_index
                    if max_x[i_label] < 0 or max_x[i_label] < point_index:
                        max_x[i_label] = point_index
                    if min_y[i_label] < 0 or min_y[i_label] > point_index:
                        min_y[i_label] = point_index
                    if max_y[i_label] < 0 or max_y[i_label] < point_index:
                        max_y[i_label] = point_index

    for i in range(1, len(contrasts_regions)):
        if squares_regions[i] == 0:
            contrasts_regions[i] = 0
        else:
            contrasts_regions[i] = contrasts_regions[i] / squares_regions[i]

    for i in range(2, len(contrasts_regions)):
        if contrasts_regions[i] == 0:
            contrasts_regions[i] = 0
        else:
            contrasts_regions[i] = contrasts_regions[i] / contrasts_regions[1]

    granulas_data['contrast'] = contrasts_regions
    granulas_data['square'] = squares_regions
    granulas_data['min_x'] = min_x
    granulas_data['max_x'] = max_x
    granulas_data['min_y'] = min_y
    granulas_data['max_y'] = max_y

    granulas_table = pd.DataFrame(granulas_data)

    if show:
        print(granulas_table)

        if img.shape[1] > 1000:
            img = imutils.resize(img, width=1000)
        cv2.imshow('Granules', img)

        cv2.waitKey(0)

    return img, granulas_table, markers


def granules_filtration(table, contrast_limit=0):
    remove_id_rows = []
    for i in range(len(table)):
        if table.at[i, 'contrast'] < contrast_limit:
            remove_id_rows.append(i)
    filt_table = table.copy()
    filt_table = filt_table.drop(remove_id_rows)

    return filt_table


if __name__ == '__main__':
    import os

    tables = []

    i = 0
    for file in os.listdir("images/"):
        if file.endswith(".jpg"):
            gran_image, table, markers = detect_granulas(os.path.join('images', file), True)
            cv2.imwrite(os.path.join(os.path.join('detected', file)), gran_image)
            tables.append(granules_filtration(table))
            i += 1
