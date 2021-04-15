import numpy as np
import cv2 as cv

from sklearn.model_selection import train_test_split
from sklearn import svm

from skimage.feature import hog
from skimage import data, exposure

from os import listdir


def example():
	img = cv.imread("example.jpg")
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	sift = cv.SIFT_create()
	kp, des = sift.detectAndCompute(gray, None)

	img = cv.drawKeypoints(gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	cv.imshow('wow', img)
	cv.waitKey()

def train(data,target):
	x_train,x_test,y_train,y_test = train_test_split(data,target,test_size=0.2,random_state=0)
	clf = svm.SVC(kernel="linear",C=1,gamma='auto')
	#reshape
	x_train = np.array(x_train)
	nsamples, nx, ny = x_train.shape
	d2_x_train = x_train.reshape((nsamples,nx*ny))#9072*1674

	#reshape2
	x_test = np.array(x_test)
	nsamplesTest ,nx,ny = x_test.shape
	d2_x_test = x_test.reshape((nsamplesTest,nx*ny))
	clf.fit(d2_x_train,y_train)
	return clf,d2_x_train,y_train,d2_x_test,y_test
def readData():
	imgList = []
	labelList = []
	label = 0

	for i in range(1,73):
		for j in range(1,11):
			image = cv.imread("CAVIAR4REID/CAVIARa/"+"%04d"%i+"%03d"%j+".jpg")
			image = cv.resize(image, (144, 72), interpolation=cv.INTER_AREA)
			#fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8,8), cells_per_block=(3,3),visualize=False, multichannel=True)


			imgList.append(image)
			labelList.append(label)
	rootpath = "voc2005_1/VOC2005_1/PNGImages/"
	for i in listdir(rootpath):
		if i == "TUGraz_person":
			label = 0
		else:
			label = 1
		for j in listdir(rootpath+i):
			image = cv.imread(rootpath+"/"+i+"/"+j)
			image = cv.resize(image, (144, 72), interpolation=cv.INTER_AREA)
			imgList.append(image)
			labelList.append(label)
	return imgList,labelList



#main
def main():
	example()


main()