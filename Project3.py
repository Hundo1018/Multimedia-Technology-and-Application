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

#濾波
fCoinFrame = cv2.bilateralFilter(gCoinFrame,100,60,10)

#二值化
temp ,bCoinFrame =cv2.threshold(fCoinFrame,75,255,cv2.THRESH_BINARY )

#侵蝕
eCoinFrame = cv2.erode(bCoinFrame,np.ones((5,5)),iterations = 3)

#物件連通
num_labels,labels,states,centroids = cv2.connectedComponentsWithStats(eCoinFrame,connectivity=8)

#畫方框
nCoinFrame = coinFrame
for s in states:
    print(s)
    cv2.rectangle(nCoinFrame,(s[0],s[1]),(s[2]+s[0],s[3]+s[1]),(0,0,127),1)

#顯示
cv2.imshow("",np.hstack([nCoinFrame]))

cv2.waitKey(0)
cv2.destroyAllWindows()
