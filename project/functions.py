import cv2
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from vector import *


def find_lines_on_frame(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    res_blue = cv2.bitwise_and(image, image, mask=mask_blue)
    cany_blue = cv2.Canny(res_blue, 50, 200, None, 3)
    lines_blue = cv2.HoughLinesP(cany_blue, 1, np.pi / 180, 50, None, 50, 20)
    line_points = [[(0, 0), (0, 0)],
                   [(0, 0), (0, 0)]]
    distance_max = 0
    if lines_blue is not None:
        for i in range(0, len(lines_blue)):
            line = lines_blue[i][0]
            distance = math.sqrt((line[2] - line[0]) ** 2 + (line[3] - line[1]) ** 2)
            if distance > distance_max:
                line_points[0][0] = (line[0], line[1])
                line_points[0][1] = (line[2], line[3])
                distance_max = distance
            cv2.line(image, (line_points[0][0][0], line_points[0][0][1]), (line_points[0][1][0], line_points[0][1][1]),
                     (0, 0, 255), 1, cv2.LINE_AA)
    return line_points


def find_countours_on_frame(frame):
    lower = np.array([230, 230, 230], dtype="uint8")
    upper = np.array([255, 255, 255], dtype="uint8")
    mask = cv2.inRange(frame, lower, upper)
    image = cv2.bitwise_and(frame, frame, mask=mask)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.GaussianBlur(gray, (5, 5), 0)
    im2, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_ret = []
    for i, c in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area > 20 and area < 1200 and h > 10:
            coordinates = (x, y, w, h)
            contours_ret.append(coordinates)
    return contours_ret


def calculate_center(contour):
    (x, y, w, h) = contour
    cx = int(x+w/2)
    cy = int(y+h/2)
    return int(cx), int(cy)


def find_element(elements, element):
    indexes = []
    i = 0
    for el in elements:
        (eX,eY) = element.center
        (hX,hY) = el.center
        distance = math.sqrt(math.pow((eX - hX),2) + math.pow((eY - hY),2))
        if distance < 20:
            indexes.append(i)
        i += 1
    return indexes


def crop_image(number):
    ret, thresh = cv2.threshold(number, 127, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    if len(areas) == 0:
        return number
    contourIndex = np.argmax(areas)
    [x, y, w, h] = cv2.boundingRect(contours[contourIndex])
    cropped = number[y:y + h + 1, x:x + w + 1]
    cropped = cv2.resize(cropped, (28,28), interpolation=cv2.INTER_AREA)
    return cropped


def crop(number):
    ret, thresh = cv2.threshold(number, 127, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    if len(areas) == 0:
        return number
    contourIndex = np.argmax(areas)
    [x, y, w, h] = cv2.boundingRect(contours[contourIndex])
    cropped = number[y:y + h + 1, x:x + w + 1]
    cropped = cv2.resize(cropped, (28,28), interpolation=cv2.INTER_AREA)
    return cropped


def predict_number(image, center, knn, contour=None):
    image_to_predict = cv2.resize(image, (28, 28), interpolation=cv2.INTER_CUBIC)
    image_to_predict = image_to_predict.reshape(784,)
    image_to_predict = np.reshape(image_to_predict, (1, -1))
    predicted = knn.predict(image_to_predict)[0]
    return predicted

def prepare_image(image):
    kernel_1 = np.ones((1, 1), np.uint8)
    kernel_2 = np.ones((2, 2), np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(gray, 195, 255, cv2.THRESH_BINARY)

    eroded = cv2.erode(img, kernel_1, 1)
    dilated = cv2.dilate(eroded, kernel_2, 1)

    return dilated, img


def find_contours(image, opsecanje, id, frameNumber):
    im2, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    crossed = 0
    found_instance = []
    list_of_found_instances = []

    for con in contours:
        x,y, width, height = cv2.boundingRect(con)
        number = opsecanje[y:y+height, x:x+width]
        center_x = x + ( width / 2 )
        center_y = y + ( height /2 )
        center = (center_x, center_y)
        found_instance = [id, frameNumber, center, crossed, number]
        list_of_found_instances.append(found_instance)

    return list_of_found_instances


def find_numbers_on_image(frameNumber, frame, id):
    er, opsecanje = prepare_image(frame)

    list_of_found_instances = find_contours(er, opsecanje, id, frameNumber)

    return list_of_found_instances


# funkcija koja uzima vrednost broja: kroz MNIST se gleda onaj sa najmanjom razlikom i uzima se kao prepoznat
def pretragaOkruzenja(vrednost, skup_vrednosti):
    prihvatljiv_broj = []

    for idx in skup_vrednosti:
        if (distance(idx[2], vrednost[2]) < 9): # 9 je random stavljeno :D
            prihvatljiv_broj.append(idx)

    # provera da li u okruzenju ima vise brojeva
    if len(prihvatljiv_broj) > 1: # u okruzenju ima vise brojeva
        # uzimaju se cenri cifara koji se potom porede
        najmanja_duzina = distance(vrednost[2], prihvatljiv_broj[0][2])
        temp = prihvatljiv_broj[0]
        lista_u_okruzenju = []
        for i in prihvatljiv_broj:
            if (distance(i[2], vrednost[2]) < najmanja_duzina):
                najmanja_duzina = distance(i[2], vrednost[2])
                temp = i
        lista_u_okruzenju.append(temp)
        return lista_u_okruzenju
    else:
        return prihvatljiv_broj # u okruzenju je samo jedan broj


def detected_numbers(list_of_found_numbers, frameNumber, numbers, id):
    for foundNum in list_of_found_numbers:
        detectedNumber = pretragaOkruzenja(foundNum, numbers)

        if len(detectedNumber) == 0:
            id = id + 1
            foundNum[0] = id
            foundNum[3] = 0
            numbers.append(foundNum)
        else:
            #center coordinates
            detectedNumber[0][2] = foundNum[2]
            detectedNumber[0][1] = frameNumber


def sum_numbers(numbers, frameNumber, line, sum, knnModel):
    for number in numbers:
        framesPassed = frameNumber - number[1]
        if(framesPassed < 7):
            dist = pnt2line(number[2], line[0][0], line[0][1])
            if(not number[3] and dist <= 7):
                number[3] = 1 #presao liniju
                predicted = predict_number(number[4], number[2], knnModel)
                sum = sum + predicted
    return sum
