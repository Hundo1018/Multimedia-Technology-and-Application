import cv2
import numpy as np

#讀取
circleFrame = cv2.imread("circle.jpg")
manFrame = cv2.imread("man.jpg")

#二值化
temp ,nCircleFrame =cv2.threshold( circleFrame,200,255,cv2.THRESH_TOZERO)
temp ,nManFrame = cv2.threshold(manFrame,200,255,cv2.THRESH_TOZERO)

#形態學：閉運算
nCircleFrame = cv2.morphologyEx(nCircleFrame,cv2.MORPH_CLOSE,np.ones((3,3)),iterations = 10)

#膨脹存成新圖
nManFrame2 = cv2.dilate(nManFrame,np.ones((3,3)),iterations = 1)

#膨脹後的新圖減未膨脹的舊圖
nManFrame = nManFrame2-nManFrame 

#原圖與新圖並排顯示
cv2.imshow("circle",np.hstack([circleFrame,nCircleFrame]))
cv2.imshow("man",np.hstack([manFrame, nManFrame]))
cv2.waitKey(0)
cv2.destroyAllWindows()