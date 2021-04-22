import numpy as np
import cv2 as cv

from sklearn.model_selection import train_test_split
from sklearn import svm

from scipy.cluster.vq import *


from os import listdir


def example():
	img = cv.imread("example.jpg")
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	sift = cv.SIFT_create()
	kp, des = sift.detectAndCompute(gray, None)

	#draw
	img = cv.drawKeypoints(gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	cv.imshow('wow', img)
	cv.waitKey()

def imageToSift(imageList):
	#resultArray = np.array([])

	des_list = []
	sift = cv.SIFT_create()
	for img in imageList:
		#fd = hog(img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(3,3),visualize=False, multichannel=True)
		grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		kp, des = sift.detectAndCompute(grayImg, None)
		des_list.append(des)


	resultArray = []
	for descriptor in des_list:
		#resultArray.append([des])
		resultArray = np.append(resultArray, descriptor)
	return des_list, resultArray


def train(data,target):
	x_train,x_test,y_train,y_test = train_test_split(data,target,test_size=0.2,random_state=0)
	clf = svm.SVC(kernel="linear",C=1,gamma='auto')

	#reshape
	x_train = np.array(x_train, dtype=object)
	nsamples, nx, ny = x_train.shape
	d2_x_train = x_train.reshape((nsamples,nx*ny))#9072*1674

	#reshape2
	x_test = np.array(x_test, dtype=object)
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
	#example()
	images,target = readData()
	des_list, Sifts = imageToSift(images)

	#k-means
	k = 20
	voc, variance = kmeans(Sifts, k, 1)

	#生成特徵直方圖
	im_features = np.zeros((len(Sifts), k), "float32")
	for i in range(len(Sifts)):
		words, distance = vq(des_list[i])
		for w in words:
			im_features[i][w] += 1



	clf,x_train,y_train,x_test,y_test = train(Sifts,target)
	print("predict")
	print(clf.predict(x_train))
	print(clf.predict(x_test))
	print("Accuracy:")
	print(clf.score(x_train,y_train))
	print(clf.score(x_test,y_test))


main()