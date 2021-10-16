#3.4.1
import cv2
from time import sleep
import numpy as np

#img = cv2.imread('ex2.jpg')
#tmp = cv2.imread('ex2.jpg')
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#initialising video capture and loading samples of keys for recognition
cap = cv2.VideoCapture(1)
sample = cv2.imread(r'C:\Users\TLab_02\Documents\synthesizer\images\sample.png')
sample1 = cv2.imread(r'C:\Users\TLab_02\Documents\synthesizer\images\sample1.png')
sample2 = cv2.imread(r'C:\Users\TLab_02\Documents\synthesizer\images\sample2.png')

test = []
num = 0

def sort(array):
    '''Bubble sort'''
    n = len(array)
    for i in range(n-1):
        for j in range(0, n-i-1):
            if array[j] > array[j+1]:
                array[j], array[j+1] = array[j+1], array[j]
    return array

def findKeys(img, deviation, match_val, samples=[]):
    '''Finding keys by matching contours to samples'''
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
    #finding keys by matching each contour to every sample. Then all data is sorted and if the lowest value is below match_val, it's counted. Everything above
    #gets rejected
    for i in range(len(contours)-1):
        for j in range(len(sample_points)):
            #print(cv2.arcLength(contours[i], True))
            if avg_arclenghth-deviation < cv2.arcLength(contours[i], True) < avg_arclenghth+deviation:
                match.append(cv2.matchShapes(sample_points[j],contours[i],1,0.0))
                #print(cv2.matchShapes(sample_points[j],contours[i],1,0.0))
                if len(match) == len(samples):
                    match = sort(match)
                    #print(min(match))
                    if match[0] < match_val:
                        keys.append(contours[i])
        #print(match)
        match = []
    return keys

tmp = []
tmp_frame = []
for i in range(20):
    frame = cap.read()[1]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keys = findKeys(gray, 22, 8, samples=[sample, sample1, sample2])
    if len(keys) == 29:
        tmp_frame = frame
    test.append(len(keys))

for i in test:
    tmp.append(test.count(i))

for i in test:
    if test.count(i) == max(tmp):
        num = i
print(num)
tmp = []
while 1:
    frame = cap.read()[1]
    print(cv2.absdiff(tmp_frame, frame))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keys = findKeys(gray, 22, 9, samples=[sample, sample1, sample2])
    # img = cv2.drawContours(img, keys, -1, (0, 255, 0), 1)
    if len(keys) == num:
        tmp = keys
        for i in keys:
            rect = cv2.minAreaRect(i)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            tmp_img = cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
    else:
        if len(tmp) > 0:
            for i in tmp:
                rect = cv2.minAreaRect(i)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                tmp_img = cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
    cv2.imshow('Image', frame)
    if cv2.waitKey(1) == 27:
        break
    #sleep(0.4)

cv2.destroyAllWindows()
