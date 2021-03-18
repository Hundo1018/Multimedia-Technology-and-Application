import cv2
import numpy as np

#read
floor = cv2.imread("floor.jpg")

#轉灰階
floor_p = cv2.cvtColor(floor, cv2.COLOR_RGB2GRAY)

#高斯模糊
floor_p = cv2.GaussianBlur(floor_p, (7, 7), 0)

#二值化
useless, floor_BIN = cv2.threshold(floor_p, 100, 255, cv2.THRESH_BINARY)

#畫出線條
floor_lines = cv2.HoughLinesP(floor_BIN, 1, np.pi/180, 200, 10, 5)
for p in floor_lines:
    p = p[0]
    cv2.line(floor, (p[0],p[1]), (p[2],p[3]), color=(0,255,0), thickness=2)

#resize
#floor = cv2.resize(floor, (562,1000), interpolation=cv2.INTER_AREA)

cv2.imwrite("floor_r.jpg", np.hstack([floor, floor_BIN]))
#cv2.waitKey()
#floor_lines = cv2.HoughLinesP(floor_resize_gray, 1, np.pi/180, 200, 10, 5)


