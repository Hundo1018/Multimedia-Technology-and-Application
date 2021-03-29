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

#HSV
hsvgCoinFrame = cv2.cvtColor(coinFrame, cv2.COLOR_BGR2HSV)

#濾波
fCoinFrame = cv2.bilateralFilter(gCoinFrame,100,60,10)

#二值化
temp ,bCoinFrame =cv2.threshold(fCoinFrame,75,255,cv2.THRESH_BINARY )

#侵蝕
eCoinFrame = cv2.erode(bCoinFrame,np.ones((3,3)),iterations = 6)

#物件連通
num_labels,labels,states,centroids = cv2.connectedComponentsWithStats(eCoinFrame,connectivity=8)

#畫方框
nCoinFrame = coinFrame
for s in states[1:]:
    #print(s)
    cv2.rectangle(nCoinFrame,(s[0],s[1]),(s[2]+s[0],s[3]+s[1]),(0,0,127),1)

l = []#資料點
lSets = []#資料點對應群集
kPoints = []#中心點位置


#k mean分群(嘗試)
#拆成list
for s in states[1:]:
    l.append(s[4])
    lSets.append(0)
    

#有1元5元10元50元，分成4群
#分類的點K，先放前四個
for i in l[:4]:
    kPoints.append(i)

#主循環
for loop in range(10):
    #每個點分給最近的
    pCount = 0#點的index，0~len(l)
    for p in l:
        kCount = 0#群集的index，0~3
        minDis = 999999#最小距離
        for k in kPoints:
            if abs(p-k)<= minDis:#如果這個點對某個中心最近
                lSets[pCount] = kCount#將此點設為該群集
                minDis = abs(p-k)
            kCount+=1
        pCount+=1

    #重新計算每個群的中心(kPoints)
    #各群數量與total
    kn = [0,0,0,0]
    kt = [0,0,0,0]
    for i in range(0,len(l)):#跑每個資料點的index比較方便
        kn[lSets[i]]+=1
        kt[lSets[i]]+=l[i]
    for i in range(0,4):
        if(kn[i]>0):
            kPoints[i] = kt[i]/kn[i]
    #顯示大小 群集
    #print("---")
    for i in range(0,len(kPoints)):
        print("中心："+str(kPoints[i]))
    for i in range(0,len(l)):
        print("數值："+str(l[i]),"群集："+str(lSets[i]))
    print("---")

for s in states[1:]:
    cv2.putText(nCoinFrame, str(s[4]), (s[0], s[1]), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 255), 1, cv2.LINE_AA)
cv2.imshow("e",eCoinFrame)
#顯示
cv2.imshow("",np.hstack([nCoinFrame]))

cv2.waitKey(0)
cv2.destroyAllWindows()
