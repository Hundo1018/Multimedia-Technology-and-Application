import cv2
import numpy as np


#
# 第一題
#
def food_1():
    print("nothing in food_1")


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

    cv2.imshow("food", floor)
    cv2.waitKey()
    print("food_2 is good")


#
# 第三題
#
def food_3():
    print("nothing in food_3")


#
# 第四題
#
def food_4():
    print("nothing in food_4")


#
# main function
#
def main():
    # food_1()
    food_2()
    # food_3()
    # food_4()

main()
