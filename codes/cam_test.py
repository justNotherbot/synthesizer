import cv2
import numpy as np

cap = cv2.VideoCapture(1)
while 1:
    frame = cap.read()[1]
    roi = frame[320:425, 10:615]
    edges = cv2.Canny(roi, 60, 255)
    #frame[31:361, 579:466] = roi
    cv2.imshow('Frame!', roi)
    if cv2.waitKey(1) == 27: break
    if cv2.waitKey(1) == ord('t'):
        cv2.imwrite('image.png', frame)

cv2.destroyAllWindows()