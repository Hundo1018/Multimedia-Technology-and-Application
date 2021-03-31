import cv2
import numpy as np
from math import * 
import random

#
# 第一題
#
def food_1():
    usingKMean = False


    #讀取
    coinFrame = cv2.imread("coin.jpg")

    #resize
    nH =(int)( coinFrame.shape[0]/8)
    nW =(int)( coinFrame.shape[1]/8)
    coinFrame = cv2.resize(coinFrame,(nW,nH),interpolation=cv2.INTER_AREA)

    #灰階
    gCoinFrame = cv2.cvtColor(coinFrame,cv2.COLOR_BGR2GRAY)

    #HSV
    hsvCoinFrame = cv2.cvtColor(coinFrame, cv2.COLOR_BGR2HSV)

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
    total = 0
    if (usingKMean):
        l = []#資料點
        lSets = []#資料點對應群集
        kPoints = []#中心點位置


        #k mean分群(嘗試)
        #拆成list
        for s in states[1:]:
            l.append([s[4] , hsvCoinFrame[s[1]][s[0]][0]])
            lSets.append(0)


        #有1元5元10元50元，依照面積大小與Hue值分成4群
        #分類的點K，先放前四個
        for i in l[:4]:
            kPoints.append([i[0],i[1]])
        
        #主循環
        for loop in range(100):
            #每個點分給最近的
            pCount = 0#點的index，0~len(l)
            for p in l:
                kCount = 0#群集的index，0~3
                minDis = 999999#最小距離
                for k in kPoints:
                    er = sqrt((p[0]-k[0])**2 + (p[1]-k[1])**2)
                    if er <= minDis:
                        lSets[pCount] = kCount
                        minDis = er
                    '''
                    if abs(p-k)<= minDis:#如果這個點對某個中心最近
                        lSets[pCount] = kCount#將此點設為該群集
                        minDis = abs(p-k)
                    '''
                    kCount+=1
                pCount+=1

            #重新計算每個群的中心(kPoints)
            #各群數量與total
            kn = [0,0,0,0]
            ktx = [0,0,0,0]
            kty = [0,0,0,0]
            for i in range(0,len(l)):#跑每個資料點的index比較方便
                kn[lSets[i]]+=1
                ktx[lSets[i]]+=l[i][0]
                kty[lSets[i]]+=l[i][1]
            for i in range(0,4):
                if(kn[i]>0):
                    kPoints[i] = [ktx[i]/kn[i],kty[i]/kn[i]]

            #顯示大小 群集
            #print("---")
            for i in range(0,len(kPoints)):
                print("中心："+str(kPoints[i]))
            for i in range(0,len(l)):
                print("數值："+str(l[i][0])+","+str(l[i][1]),"群集："+str(lSets[i]))
            print("---")
        index = 0
        #畫畫
        for s in states[1:]:
            #標記數值
            cv2.putText(nCoinFrame, str(s[4]), (s[0], s[1]), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 255), 1, cv2.LINE_AA)
            #畫方框
            if lSets[index]==0:
                cv2.rectangle(nCoinFrame,(s[0],s[1]),(s[2]+s[0],s[3]+s[1]),(0,0,255),2)
            elif lSets[index]==1:
                cv2.rectangle(nCoinFrame,(s[0],s[1]),(s[2]+s[0],s[3]+s[1]),(0,0,127),2)
            elif lSets[index]==2:
                cv2.rectangle(nCoinFrame,(s[0],s[1]),(s[2]+s[0],s[3]+s[1]),(0,255,255),2)
            elif lSets[index]==3:
                cv2.rectangle(nCoinFrame,(s[0],s[1]),(s[2]+s[0],s[3]+s[1]),(0,255,0),2)
            index += 1
    else:
        for s in states[1:]:
            #標記數值
            #cv2.putText(nCoinFrame, str(s[4]), (s[0], s[1]), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 255), 1, cv2.LINE_AA)
            #畫方框
            if s[4]>=1900:
                total+=50
                cv2.rectangle(nCoinFrame,(s[0],s[1]),(s[2]+s[0],s[3]+s[1]),(0,255,0),2)
                cv2.putText(nCoinFrame, '50', (s[0]+s[2]//4, s[1]+s[3]//2), cv2.FONT_HERSHEY_SIMPLEX,0.5,    (0,0,0), 2, cv2.LINE_AA)

            elif s[4]>=1100:
                total+=10
                cv2.rectangle(nCoinFrame,(s[0],s[1]),(s[2]+s[0],s[3]+s[1]),(0,255,255),2)
                cv2.putText(nCoinFrame, '10', (s[0]+s[2]//4, s[1]+s[3]//2), cv2.FONT_HERSHEY_SIMPLEX,0.5,    (0,0,0), 2, cv2.LINE_AA)
            elif s[4]>=900:
                total+=5
                cv2.rectangle(nCoinFrame,(s[0],s[1]),(s[2]+s[0],s[3]+s[1]),(80,127,255),2)
                cv2.putText(nCoinFrame, '5', (s[0]+s[2]//4, s[1]+s[3]//2), cv2.FONT_HERSHEY_SIMPLEX,0.5,    (0,0,0), 2, cv2.LINE_AA)
            elif s[4]>=0:
                total+=1
                cv2.rectangle(nCoinFrame,(s[0],s[1]),(s[2]+s[0],s[3]+s[1]),(0,0,255),2)
                cv2.putText(nCoinFrame, '1', (s[0]+s[2]//4, s[1]+s[3]//2), cv2.FONT_HERSHEY_SIMPLEX,0.5,    (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(nCoinFrame, 'total:'+str(total), (12,20), cv2.FONT_HERSHEY_SIMPLEX,0.5,    (255,255,255), 1, cv2.LINE_AA)   

    #cv2.imshow("e",eCoinFrame)
    #顯示
    cv2.imshow("1",nCoinFrame)
    cv2.imwrite("coinResult.jpg", nCoinFrame)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

#
# 第二題
#
def food_2():
    # read
    floor = cv2.imread("floor.jpg")

    # resize
    floor = cv2.resize(floor, (562, 1000), interpolation=cv2.INTER_AREA)

    # 轉灰階
    floor_p = cv2.cvtColor(floor, cv2.COLOR_RGB2GRAY)

    # 高斯模糊
    floor_p = cv2.GaussianBlur(floor_p, (5, 5), 0)

    # 二值化
    useless, floor_BIN = cv2.threshold(floor_p, 120, 255, cv2.THRESH_BINARY_INV)

    # CLOSE
    floor_close = cv2.morphologyEx(floor_BIN, cv2.MORPH_CLOSE, np.ones((5, 5)), iterations=1)

    # 畫出線條
    floor_lines = cv2.HoughLinesP(floor_close, 10, 3*np.pi/180, 500, 50, 20)
    for p in floor_lines:
        p = p[0]
        cv2.line(floor, (p[0], p[1]), (p[2], p[3]),
                 color=(0, 0, 255), thickness=2)

    cv2.imshow("2", floor)
    cv2.imwrite("floorResult.jpg", floor)
    #cv2.waitKey()
    #print("food_2 is good")


#
# 第三題
#
def food_3():
    #讀取
    coinFrame = cv2.imread("coin2.jpg")

    #resize
    nH =(int)( coinFrame.shape[0]/8)
    nW =(int)( coinFrame.shape[1]/8)
    coinFrame = cv2.resize(coinFrame,(nW,nH),interpolation=cv2.INTER_AREA)

    #灰階
    gCoinFrame = cv2.cvtColor(coinFrame,cv2.COLOR_BGR2GRAY)

    #HSV
    hsvCoinFrame = cv2.cvtColor(coinFrame, cv2.COLOR_BGR2HSV)

    #濾波
    fCoinFrame = cv2.bilateralFilter(gCoinFrame,100,60,10)
    #cv2.imshow("f",fCoinFrame)

    #二值化
    temp ,bCoinFrame =cv2.threshold(fCoinFrame,75,255,cv2.THRESH_BINARY )



    eCoinFrame = bCoinFrame
    #開運算
    #把硬幣分離
    eCoinFrame = cv2.morphologyEx(eCoinFrame, cv2.MORPH_OPEN, np.ones((3,3)),iterations = 5)


    #物件連通
    num_labels,labels,states,centroids = cv2.connectedComponentsWithStats(eCoinFrame,connectivity=8)
    total = 0
    #畫方框
    nCoinFrame = coinFrame
    for s in states[1:]:
            #畫方框
            #cv2.rectangle(nCoinFrame,(s[0],s[1]),(s[2]+s[0],s[3]+s[1]),(0,255,0),2)
            #標記數值
            #cv2.putText(nCoinFrame, str(s[4]), (s[0], s[1]), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 255), 1, cv2.LINE_AA)
            
            if s[4]<=650:#1
                total += 1
                cv2.rectangle(nCoinFrame,(s[0],s[1]),(s[2]+s[0],s[3]+s[1]),(0,0,255),2)#紅色
                cv2.putText(nCoinFrame, '1', (s[0]+s[2]//4, s[1]+s[3]//2), cv2.FONT_HERSHEY_SIMPLEX,0.5,    (0,0,0), 2, cv2.LINE_AA)
            elif s[4]<=750:#5
                total += 5
                cv2.rectangle(nCoinFrame,(s[0],s[1]),(s[2]+s[0],s[3]+s[1]),(80,127,255),2)#橘色
                cv2.putText(nCoinFrame, '5', (s[0]+s[2]//4, s[1]+s[3]//2), cv2.FONT_HERSHEY_SIMPLEX,0.5,    (0,0,0), 2, cv2.LINE_AA)
            elif s[4]<=1100:#10
                total += 10
                cv2.rectangle(nCoinFrame,(s[0],s[1]),(s[2]+s[0],s[3]+s[1]),(0,255,255),2)#黃色
                cv2.putText(nCoinFrame, '10', (s[0]+s[2]//4, s[1]+s[3]//2), cv2.FONT_HERSHEY_SIMPLEX,0.5,   (0,0,0), 2, cv2.LINE_AA)
            elif s[4]<=1450:#50
                total += 50
                cv2.rectangle(nCoinFrame,(s[0],s[1]),(s[2]+s[0],s[3]+s[1]),(0,255,0),2)#綠色
                cv2.putText(nCoinFrame, '50', (s[0]+s[2]//4, s[1]+s[3]//2), cv2.FONT_HERSHEY_SIMPLEX,0.5,   (0,0,0), 2, cv2.LINE_AA)
            elif s[4]<=21000:#500
                total += 500
                cv2.rectangle(nCoinFrame,(s[0],s[1]),(s[2]+s[0],s[3]+s[1]),(127,0,127),2)#紫
                cv2.putText(nCoinFrame, '500', (s[0]+s[2]//4, s[1]+s[3]//2), cv2.FONT_HERSHEY_SIMPLEX,0.5,  (0,0,0), 2, cv2.LINE_AA)
            elif s[4]<=21400:#100
                total += 100
                cv2.rectangle(nCoinFrame,(s[0],s[1]),(s[2]+s[0],s[3]+s[1]),(255,0,0),2)#藍
                cv2.putText(nCoinFrame, '100', (s[0]+s[2]//4, s[1]+s[3]//2), cv2.FONT_HERSHEY_SIMPLEX,0.5,  (0,0,0), 2, cv2.LINE_AA)
            elif s[4]<=22000:#1000
                total += 1000
                cv2.rectangle(nCoinFrame,(s[0],s[1]),(s[2]+s[0],s[3]+s[1]),(255,255,255),2)#白
                cv2.putText(nCoinFrame, '1000', (s[0]+s[2]//4, s[1]+s[3]//2), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(nCoinFrame, 'total:'+str(total), (12,20), cv2.FONT_HERSHEY_SIMPLEX,0.5,    (255,255,255), 1, cv2.LINE_AA)   
            
    #顯示
    cv2.imshow("3",nCoinFrame)
    cv2.imwrite("coinResult2.jpg", nCoinFrame)

#
# 第四題
#
def food_4():
    appleFrame = cv2.imread("apple.jpg")
    orangeFrame = cv2.imread("orange.jpg")


    w = appleFrame.shape[0]

    #resize縮小影像
    w=w//2
    daINTER_LINEAR  =cv2.resize(appleFrame,(w,w),interpolation=cv2.INTER_LINEAR)
    doINTER_LINEAR  =cv2.resize(orangeFrame,(w,w),interpolation=cv2.INTER_LINEAR)
    
    ##組合
    aoINTER_LINEAR   = np.hstack([daINTER_LINEAR  [:,:w//2],doINTER_LINEAR  [:,w//2:]])

    smallW = w//2
    bigW = w*2
    blurV = 5
    blurV2 = 7

    border = 7

    #模糊
    BaoINTER_LINEAR  =cv2.blur(    aoINTER_LINEAR  ,(blurV,blurV))

    #複製
    IaoINTER_LINEAR  =aoINTER_LINEAR  

    #插到中間
    IaoINTER_LINEAR  [::,smallW-border:smallW+border] =  BaoINTER_LINEAR  [::,smallW-border:smallW+border]

    #放大
    UaoINTER_LINEAR  =cv2.resize(IaoINTER_LINEAR  ,(bigW,bigW),interpolation=cv2.INTER_LINEAR)

    #再次模糊
    NaoINTER_LINEAR  =cv2.blur(    UaoINTER_LINEAR  ,(blurV2,blurV2))
    

    #顯示
    cv2.imshow("4",NaoINTER_LINEAR  )

    cv2.imwrite("appleOrangeResult.jpg", NaoINTER_LINEAR)


#
# main function
#
def main():
    food_1()
    food_2()
    food_3()
    food_4()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
main()
