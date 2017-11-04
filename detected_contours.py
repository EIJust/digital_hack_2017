import cv2
import numpy
import imutils

numpy.set_printoptions(threshold=numpy.nan)

clicked = False


def nothing(x):
    pass


def crop_image(data):
    for coordinate in data:
        # нулевой y:высота - (сверху вниз), нулевая х: ширина - (снизу вверх)
        crop_img = img[coordinate[1]:coordinate[3], coordinate[0]:coordinate[2]]
        crop_imgs.append(crop_img)


def triger_fn(cont_list):
    for conture in cont_list:
        min_x = -1
        max_x = -1
        min_y = -1
        max_y = -1
        for pair in conture:
            pair = pair[0]
            if min_x > pair[0] or min_x == -1:
                min_x = pair[0]
            if min_y > pair[1] or min_y == -1:
                min_y = pair[1]
            if max_x < pair[0] or max_x == -1:
                max_x = pair[0]
            if max_y < pair[1] or max_y == -1:
                max_y = pair[1]
                coordinates.append([min_x, min_y, max_x, max_y])
    crop_image(coordinates)


img = cv2.imread('images/ANP5.jpg')
coordinates = []
crop_imgs = []

if img.shape[1] > 800:
    img = imutils.resize(img, width=800)

cv2.namedWindow('Treshed')

# create trackbars for treshold change
cv2.createTrackbar('Treshold', 'Treshed', 100, 255, nothing)

prev_tresh_value = 0
waitable_iterations = 0
while True:
    clone = img.copy()

    gray = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)

    # get current positions of four trackbars
    r = cv2.getTrackbarPos('Treshold', 'Treshed')

    ret, gray_threshed = cv2.threshold(gray, r, 255, cv2.THRESH_BINARY)

    # Blur an image
    bilateral_filtered_image = cv2.bilateralFilter(gray_threshed, 5, 175, 175)

    # Detect edges
    edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)

    # Find contours
    _, contours, _ = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_list = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        if (len(approx) > 8) & (area > 30):
            contour_list.append(contour)

    # Draw contours on the original image
    cv2.drawContours(clone, contour_list, -1, (255, 0, 0), 2)

    # there is an outer boundary and inner boundary for each eadge, so contours double
    # print('Number of found circles: {}'.format(int(len(contour_list) / 2)))

    cv2.imshow('Objects Detected', clone)
    cv2.imshow("Treshed", gray_threshed)

    k = cv2.waitKey(1) & 0xFF
    if prev_tresh_value == r:
        waitable_iterations += 1
    else:
        waitable_iterations = 0
        prev_tresh_value = r
        clicked = False
    if waitable_iterations > 150 and not clicked:
        clicked = True
        triger_fn(contour_list)
    for idx, image in enumerate(crop_imgs):
        cv2.imwrite('images/granules/' + str(idx) + '.jpg', image)

    if k == 27:
        break

cv2.destroyAllWindows()
