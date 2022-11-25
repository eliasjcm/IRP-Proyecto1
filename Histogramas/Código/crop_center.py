import numpy as np
import cv2


def filters(img):
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    img = cv2.bitwise_not(img)
    for i in range(3):
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)

    img = cv2.bitwise_not(img)

    return img


def center_image(img):
    minX = minY = maxX = maxY = -1
    height, width = img.shape
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    whiteCount = 0
    blackFlag = False
    for j in range(width-2):
        whiteFlag = False
        for i in range(height-2):
            if img[i, j] == 0:
                blackFlag = True
                whiteFlag = True
                whiteCount = 0
                minX = j if minX == -1 else min(minX, j)
                maxX = j if maxX == -1 else max(maxX, j)
                minY = i if minY == -1 else min(minY, i)
                maxY = i if maxY == -1 else max(maxY, i)
        if not whiteFlag and blackFlag:
            whiteCount += 1
        if whiteCount > 7:
            break

    newImage = img[minY:maxY, minX:maxX]
    newImage = cv2.copyMakeBorder(
        newImage, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    newImage = cv2.resize(newImage, (50, 67), interpolation=cv2.INTER_AREA)
    _, newImage = cv2.threshold(
        newImage, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return newImage


def crop_image(img: np.array, x1, x2, y1, y2, num, nameID):
    varName = nameID
    distanceX = (x2 - x1)//17
    distanceY = (y2 - y1)//3

    for i in range(3):
        for j in range(17):
            crop = img[(y1+12) + i*distanceY:(y1-3) + (i+1)*distanceY,
                       (x1+13) + j*distanceX:(x1-1) + (j+1)*distanceX]

            crop = center_image(crop)

            if np.mean(crop) != 255:
                crop = cv2.bitwise_not(crop)
                cv2.imwrite(f'numbers/{num}/'+str(num) + '_' +
                            str(varName) + '.png', crop)
            varName += 1
