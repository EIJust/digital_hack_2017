import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils

img = cv2.imread('ANP5.jpg')
if img.shape[1] > 600:
    img = imutils.resize(img, width=600)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_source = gray.copy()

ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) #nahodit granitsu

# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=10)

# sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3) #binarnoe image
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5) #4e eto?
ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0) #4to takoe sure_fg?
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg) #ret= kol-vo granul.
print('ret', ret)
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


for i_label in range(1, ret + 1):
    region_matrix = markers == i_label
    for i in range(0, len(region_matrix)):
        x = region_matrix[i]
        for point_index in range(0, len(x)):
            if x[point_index]:
                squares_regions[i_label] += 1
                contrasts_regions[i_label] += img_source[i, point_index]

for i in range(1, len(contrasts_regions)):
    contrasts_regions[i] = contrasts_regions[i] / squares_regions[i]

for i in range(2, len(contrasts_regions)):
    contrasts_regions[i] = contrasts_regions[i] / contrasts_regions[1]

print("Contrast regions")
print(contrasts_regions)

print("Square regions")
print(squares_regions)

cv2.imshow('wtf', img)

cv2.waitKey(0)
