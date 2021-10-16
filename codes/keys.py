#3.4.1
import cv2
from time import sleep
import numpy as np

#img = cv2.imread('ex2.jpg')
#tmp = cv2.imread('ex2.jpg')
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cap = cv2.VideoCapture(1)
sample = cv2.imread(r'C:\Users\TLab_01\Documents\synthesizer\images\sample.png')
sample1 = cv2.imread(r'C:\Users\TLab_01\Documents\synthesizer\images\sample1.png')
sample2 = cv2.imread(r'C:\Users\TLab_01\Documents\synthesizer\images\sample2.png')

def sort(array):
    n = len(array)
    for i in range(n-1):
        for j in range(0, n-i-1):
            if array[j] > array[j+1]:
                array[j], array[j+1] = array[j+1], array[j]
    return array

def findKeys(img, deviation,match_val,samples=[]):
    avg_arclenghth = 200
    keys = []
    sample_points = []
    match = []
    #finding points for all samples
    for i in range(len(samples)):
        sample_edges = cv2.Canny(samples[i], 50, 248)
        image, sample_contours, hierarchy = cv2.findContours(sample_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sample_points.append(sample_contours[0])
    img = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(img, 60, 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    dilation = cv2.dilate(edges, kernel, iterations=1)
    image, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #contours, image = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)-1):
        for j in range(len(sample_points)):
            #print(cv2.arcLength(contours[i], True))
            if avg_arclenghth-deviation < cv2.arcLength(contours[i], True) < avg_arclenghth+deviation:
                match.append(cv2.matchShapes(sample_points[j],contours[i],1,0.0))
                #print(cv2.matchShapes(sample_points[j],contours[i],1,0.0))
                if len(match) == len(samples):
                    match = sort(match)
                    if match[0] < match_val:
                        keys.append(contours[i])
        #print(match)
        match = []
    return keys

while 1:
    frame = cap.read()[1]
    roi = frame[320:425, 10:615]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    keys = findKeys(gray, 22, 8, samples=[sample, sample1, sample2])
    # img = cv2.drawContours(img, keys, -1, (0, 255, 0), 1)
    for i in keys:
        rect = cv2.minAreaRect(i)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        tmp = cv2.drawContours(roi, [box], 0, (0, 0, 255), 2)
    cv2.imshow('Image', roi)
    if cv2.waitKey(1) == 27:
        break
    sleep(0.8)

cv2.destroyAllWindows()
