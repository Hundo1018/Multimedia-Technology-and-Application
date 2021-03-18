import cv2
import numpy as np

#讀取
coinFrame = cv2.imread("coin.jpg")

#resize
nH =(int)( coinFrame.shape[0]/8)
nW =(int)( coinFrame.shape[1]/8)
coinFrame = cv2.resize(coinFrame,(nW,nH),interpolation=cv2.INTER_AREA)

#灰階
gCoinFrame = cv2.cvtColor(coinFrame,cv2.COLOR_BGR2GRAY)
#cv2.imshow("o",gCoinFrame)

#濾波
fCoinFrame = cv2.bilateralFilter(gCoinFrame,100,75,75)
#fCoinFrame = cv2.bilateralFilter(gCoinFrame,100,60,10)

#二值化
temp ,bCoinFrame =cv2.threshold(fCoinFrame,75,255,cv2.THRESH_BINARY )
cv2.imshow("",np.hstack([bCoinFrame]))


#顯示
#cv2.imshow("test",np.hstack([fCoinFrame]))

cv2.waitKey(0)
cv2.destroyAllWindows()
