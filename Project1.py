import cv2
import numpy as np

frame = cv2.imread("Test.jpg")

#取得長寬
h,w = frame.shape[0],frame.shape[1]

#做出跟原圖一樣大的np多維陣列
frameR = np.zeros((h,w,3),np.uint8)
frameG = np.zeros((h,w,3),np.uint8)
frameB = np.zeros((h,w,3),np.uint8)

#取得RGB通道，如果直接show frame 會只有灰階(1通道)
#ex: imshow("",frame[:,:,1])
frameR[:,:,2] = frame[:,:,2]
frameG[:,:,1] = frame[:,:,1]
frameB[:,:,0] = frame[:,:,0]

#灰階化
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
ycrcb = cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)
hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

#show圖
cv2.imshow("RGB",np.hstack([frameR,frameG,frameB]))
cv2.imshow("YCrCb,HSV",np.hstack([ycrcb,hsv]))
cv2.imshow("Gray",gray)
cv2.waitKey(0)