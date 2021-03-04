import cv2
import numpy as np

frame = cv2.imread("Test.jpg")

#line
cv2.line(frame, (3,3), (300,300), (200,200,255), 2)
#box
cv2.rectangle(frame, (50,200), (200,260), (0,127,0),2)
#circle
cv2.circle(frame, (50,50), 30, (255,0,0),1)

cv2.imshow("test",frame)
cv2.waitKey(0)
cv2.destroyAllWindows()