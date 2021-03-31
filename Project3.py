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
            elif s[4]>=1100:
                total+=10
                cv2.rectangle(nCoinFrame,(s[0],s[1]),(s[2]+s[0],s[3]+s[1]),(0,255,255),2)
            elif s[4]>=900:
                total+=5
                cv2.rectangle(nCoinFrame,(s[0],s[1]),(s[2]+s[0],s[3]+s[1]),(80,127,255),2)
            elif s[4]>=0:
                total+=1
                cv2.rectangle(nCoinFrame,(s[0],s[1]),(s[2]+s[0],s[3]+s[1]),(0,0,255),2)
    cv2.putText(nCoinFrame, 'total:'+str(total), (12,20), cv2.FONT_HERSHEY_SIMPLEX,0.5,    (255,255,255), 1, cv2.LINE_AA)   

    #cv2.imshow("e",eCoinFrame)
    #顯示
    cv2.imshow("1",np.hstack([nCoinFrame]))

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
    cv2.imshow("3",np.hstack([nCoinFrame]))


#
# 第四題
#
def food_4():
    appleFrame = cv2.imread("apple.jpg")
    orangeFrame = cv2.imread("orange.jpg")


    w = appleFrame.shape[0]

    #resize縮小影像
    w=w//2
    #daINTER_NEAREST =cv2.resize(appleFrame,(w,w),interpolation=cv2.INTER_NEAREST)
    daINTER_LINEAR  =cv2.resize(appleFrame,(w,w),interpolation=cv2.INTER_LINEAR)
    #daINTER_AREA    =cv2.resize(appleFrame,(w,w),interpolation=cv2.INTER_AREA)
    #daINTER_CUBIC   =cv2.resize(appleFrame,(w,w),interpolation=cv2.INTER_CUBIC)
    #daINTER_LANCZOS4=cv2.resize(appleFrame,(w,w),interpolation=cv2.INTER_LANCZOS4)

    #doINTER_NEAREST =cv2.resize(orangeFrame,(w,w),interpolation=cv2.INTER_NEAREST)
    doINTER_LINEAR  =cv2.resize(orangeFrame,(w,w),interpolation=cv2.INTER_LINEAR)
    #doINTER_AREA    =cv2.resize(orangeFrame,(w,w),interpolation=cv2.INTER_AREA)
    #doINTER_CUBIC   =cv2.resize(orangeFrame,(w,w),interpolation=cv2.INTER_CUBIC)
    #doINTER_LANCZOS4=cv2.resize(orangeFrame,(w,w),interpolation=cv2.INTER_LANCZOS4)
    
    ##組合
    #aoINTER_NEAREST  = np.hstack([daINTER_NEAREST [:,:w//2],doINTER_NEAREST [:,w//2:]])
    aoINTER_LINEAR   = np.hstack([daINTER_LINEAR  [:,:w//2],doINTER_LINEAR  [:,w//2:]])
    #aoINTER_AREA     = np.hstack([daINTER_AREA    [:,:w//2],doINTER_AREA    [:,w//2:]])
    #aoINTER_CUBIC    = np.hstack([daINTER_CUBIC   [:,:w//2],doINTER_CUBIC   [:,w//2:]])
    #aoINTER_LANCZOS4 = np.hstack([daINTER_LANCZOS4[:,:w//2],doINTER_LANCZOS4[:,w//2:]])
    smallW = w//2
    bigW = w*2
    blurV = 5
    blurV2 = 7

    border = 7
    #模糊
    #blurV = 17
    #BaoINTER_NEAREST =cv2.blur(    aoINTER_NEAREST ,(blurV,blurV))
    BaoINTER_LINEAR  =cv2.blur(    aoINTER_LINEAR  ,(blurV,blurV))

    #BaoINTER_AREA    =cv2.blur(    aoINTER_AREA    ,(blurV,blurV))
    #BaoINTER_CUBIC   =cv2.blur(    aoINTER_CUBIC   ,(blurV,blurV))
    #BaoINTER_LANCZOS4=cv2.blur(    aoINTER_LANCZOS4,(blurV,blurV))

    #複製
    #IaoINTER_NEAREST =aoINTER_NEAREST 
    IaoINTER_LINEAR  =aoINTER_LINEAR  
    #IaoINTER_AREA    =aoINTER_AREA    
    #IaoINTER_CUBIC   =aoINTER_CUBIC   
    #IaoINTER_LANCZOS4=aoINTER_LANCZOS4

    #插到中間
    #border = 9
    #IaoINTER_NEAREST [::,smallW-border:smallW+border] =  BaoINTER_NEAREST [::,smallW-border:smallW+border]
    IaoINTER_LINEAR  [::,smallW-border:smallW+border] =  BaoINTER_LINEAR  [::,smallW-border:smallW+border]
    #IaoINTER_AREA    [::,smallW-border:smallW+border] =  BaoINTER_AREA    [::,smallW-border:smallW+border]
    #IaoINTER_CUBIC   [::,smallW-border:smallW+border] =  BaoINTER_CUBIC   [::,smallW-border:smallW+border]
    #IaoINTER_LANCZOS4[::,smallW-border:smallW+border] =  BaoINTER_LANCZOS4[::,smallW-border:smallW+border]
    #w=w*2

    #放大
    
    #UaoINTER_NEAREST =cv2.resize(IaoINTER_NEAREST ,(bigW,bigW),interpolation=cv2.INTER_NEAREST)
    UaoINTER_LINEAR  =cv2.resize(IaoINTER_LINEAR  ,(bigW,bigW),interpolation=cv2.INTER_LINEAR)
    #UaoINTER_AREA    =cv2.resize(IaoINTER_AREA    ,(bigW,bigW),interpolation=cv2.INTER_AREA)
    #UaoINTER_CUBIC   =cv2.resize(IaoINTER_CUBIC   ,(bigW,bigW),interpolation=cv2.INTER_CUBIC)
    #UaoINTER_LANCZOS4=cv2.resize(IaoINTER_LANCZOS4,(bigW,bigW),interpolation=cv2.INTER_LANCZOS4)

    #再次模糊
    #blurV2 = 7
    #NaoINTER_NEAREST =cv2.blur(    UaoINTER_NEAREST ,(blurV2,blurV2))
    NaoINTER_LINEAR  =cv2.blur(    UaoINTER_LINEAR  ,(blurV2,blurV2))
    #NaoINTER_AREA    =cv2.blur(    UaoINTER_AREA    ,(blurV2,blurV2))
    #NaoINTER_CUBIC   =cv2.blur(    UaoINTER_CUBIC   ,(blurV2,blurV2))
    #NaoINTER_LANCZOS4=cv2.blur(    UaoINTER_LANCZOS4,(blurV2,blurV2))
    

    #顯示
    #cv2.imshow("aoINTER_NEAREST "+str(blurV)+","+str(blurV2)+","+str(border),NaoINTER_NEAREST )
    #cv2.imshow("aoINTER_LINEAR  "+str(blurV)+","+str(blurV2)+","+str(border),NaoINTER_LINEAR  )
    cv2.imshow("4",NaoINTER_LINEAR  )
    #cv2.imshow("aoINTER_AREA    "+str(blurV)+","+str(blurV2)+","+str(border),NaoINTER_AREA    )
    #cv2.imshow("aoINTER_CUBIC   "+str(blurV)+","+str(blurV2)+","+str(border),NaoINTER_CUBIC   )
    #cv2.imshow("aoINTER_LANCZOS4"+str(blurV)+","+str(blurV2)+","+str(border),NaoINTER_LANCZOS4)
    '''
    w = appleFrame.shape[0]
    w=w//4
    #apple向下
    danearFrame =   cv2.resize(appleFrame,(w,w),interpolation=cv2.INTER_NEAREST)
    dalinearFrame = cv2.resize(appleFrame,(w,w),interpolation=cv2.INTER_LINEAR)
    daareaFrame =   cv2.resize(appleFrame,(w,w),interpolation=cv2.INTER_AREA)
    dacubicFrame =  cv2.resize(appleFrame,(w,w),interpolation=cv2.INTER_CUBIC)
    dalanczsFrame = cv2.resize(appleFrame,(w,w),interpolation=cv2.INTER_LANCZOS4)

    #orange向下
    donearFrame = cv2.resize(orangeFrame,(w,w),interpolation=cv2.INTER_NEAREST)
    dolinearFrame = cv2.resize(orangeFrame,(w,w),interpolation=cv2.INTER_LINEAR)
    doareaFrame = cv2.resize(orangeFrame,(w,w),interpolation=cv2.INTER_AREA)
    docubicFrame = cv2.resize(orangeFrame,(w,w),interpolation=cv2.INTER_CUBIC)
    dolanczsFrame = cv2.resize(orangeFrame,(w,w),interpolation=cv2.INTER_LANCZOS4)

    #合併
    nearhsFrame = np.hstack([    danearFrame[:,0:w//2], donearFrame[:,w//2:w]  ])
    linehsFrame = np.hstack([    dalinearFrame[:,0:w//2],dolinearFrame[:,w//2:w]  ])
    areahsFrame = np.hstack([    daareaFrame[:,0:w//2],doareaFrame[:,w//2:w]  ])
    cubhsFrame = np.hstack([    dacubicFrame[:,0:w//2],docubicFrame[:,w//2:w]  ])
    lanhsFrame = np.hstack([    dalanczsFrame[:,0:w//2],dolanczsFrame[:,w//2:w]  ])
    
    #模糊
    ksize = 5
    BnearhsFrame = cv2.blur(nearhsFrame, (ksize, ksize))
    BlinehsFrame = cv2.blur(linehsFrame, (ksize, ksize))
    BareahsFrame = cv2.blur(areahsFrame, (ksize, ksize))
    BcubhsFrame  = cv2.blur(cubhsFrame , (ksize, ksize))
    BlanhsFrame  = cv2.blur(lanhsFrame , (ksize, ksize))
    NnearhsFrame = nearhsFrame
    NlinehsFrame = linehsFrame
    NareahsFrame = areahsFrame
    NcubhsFrame  = cubhsFrame 
    NlanhsFrame  = lanhsFrame 

    #中間換成模糊
    w=w//4
    border = 100
    #NnearhsFrame[::,w-border:w+border] =  BnearhsFrame[::,w-border:w+border]
    #NlinehsFrame[::,w-border:w+border] =  BlinehsFrame[::,w-border:w+border]
    #NareahsFrame[::,w-border:w+border] =  BareahsFrame[::,w-border:w+border]
    #NcubhsFrame [::,w-border:w+border] =  BcubhsFrame [::,w-border:w+border]
    #NlanhsFrame [::,w-border:w+border] =  BlanhsFrame [::,w-border:w+border]
    w=w*4


    #upFrame = cv2.pyrUp(hsFrame)
    w=w*4

    NnearhsFrame = cv2.resize(     NnearhsFrame ,(w,w),interpolation=cv2.INTER_NEAREST)
    NlinehsFrame = cv2.resize(   NlinehsFrame ,(w,w),interpolation=cv2.INTER_LINEAR)
    NareahsFrame = cv2.resize(     NareahsFrame ,(w,w),interpolation=cv2.INTER_AREA)
    NcubhsFrame  = cv2.resize(    NcubhsFrame  ,(w,w),interpolation=cv2.INTER_CUBIC)
    NlanhsFrame  = cv2.resize(   NlanhsFrame  ,(w,w),interpolation=cv2.INTER_LANCZOS4)



    #cv2.imshow("up", upFrame)
    cv2.imshow("near",  NnearhsFrame )
    cv2.imshow("line",  NlinehsFrame )
    cv2.imshow("area",  NareahsFrame )
    cv2.imshow("cub",   NcubhsFrame  )
    cv2.imshow("lan",   NlanhsFrame  )
    '''


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
