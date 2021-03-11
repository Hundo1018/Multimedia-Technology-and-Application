import cv2
import numpy as np


circleFrame = cv2.imread("circle.jpg")
manFrame = cv2.imread("man.jpg")


temp ,nCircleFrame =cv2.threshold( circleFrame,200,255,cv2.THRESH_TOZERO)
temp ,nManFrame = cv2.threshold(manFrame,200,255,cv2.THRESH_TOZERO)

nCircleFrame = cv2.morphologyEx(nCircleFrame,cv2.MORPH_CLOSE,np.ones((3,3)),iterations = 9)
nManFrame2 = cv2.dilate(nManFrame,np.ones((3,3)),iterations = 1)
nManFrame = nManFrame2-nManFrame 


cv2.imshow("circle",np.hstack([circleFrame,nCircleFrame]))
cv2.imshow("man",np.hstack([manFrame, nManFrame]))
cv2.waitKey(0)
cv2.destroyAllWindows()