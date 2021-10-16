import cv2
import numpy as np

img = cv2.imread(r'C:\Users\TLab_01\Documents\synthesizer\images\ex4.jpg')
img = cv2.GaussianBlur(img,(5,5),0)
tmp_c = img

#gray = cv2.cvtColor(tmp_c, cv2.COLOR_BGR2GRAY)
#corners = cv2.goodFeaturesToTrack(gray, 110, 0.13, 6)

edges = cv2.Canny(img, 50, 248)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
#opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
dilation = cv2.dilate(edges,kernel,iterations = 1)
#kernel1 = np.ones((1,1),np.uint8)
#erosion = cv2.erode(edges,kernel1,iterations = 1)
#contours, image = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
image, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
n=12
img = cv2.drawContours(img, contours, n, (0, 255, 0), 1)
epsilon = 0.009*cv2.arcLength(contours[n],True)
#print(cv2.arcLength(contours[n],True))
approx = cv2.approxPolyDP(contours[n],epsilon,True)
rect = cv2.minAreaRect(approx)
print(rect[2])
box = cv2.boxPoints(rect)
box = np.int0(box)
img = cv2.drawContours(img,[box],0,(0,0,255),2)
#print(approx)
for i in approx:
    x,y = i.ravel()
#    #print(x,y)
    cv2.circle(img,(int(x),int(y)),3,255,-1)#print(contours[30])

background = np.zeros((200,200,3), np.uint8)
def none(n):
    pass

cv2.namedWindow('trackbars')

t1 = cv2.createTrackbar('t1','trackbars',0,255,none)
t2 = cv2.createTrackbar('t2','trackbars',0,255,none)

while 1:

    t1 = cv2.getTrackbarPos('t1', 'trackbars')
    t2 = cv2.getTrackbarPos('t2', 'trackbars')

    #ret,threshold = cv2.threshold(gray, t1, t2,  cv2.ADAPTIVE_THRESH_GAUSSIAN_C)

    #cv2.imshow('Bruh', threshold)
    cv2.imshow('Canny', img)
    #cv2.imshow('dilation/erosion', dilation)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()