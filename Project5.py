from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np

import cv2

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm

from skimage.feature import hog
from skimage import data, exposure

from os import listdir

def example():
	image = data.astronaut()

	fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8,8),
		cells_per_block=(3,3), visualize=True, multichannel=True)

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

	ax1.axis('off')
	ax1.imshow(image, cmap=plt.cm.gray)
	ax1.set_title('input image')

	#resize
	hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

	ax2.axis('off')
	ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
	ax2.set_title('hog image')

	plt.show()


def imageToHog(imageList):
	resultArray = np.array([])
	for img in imageList:
		fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(3,3),
		visualize=False, multichannel=True)
		np.append(resultArray, [fd], axis=0)
	return resultArray

def train(data,target):
	x_train,x_test,y_train,y_test = train_test_split(data,target,test_size=0.2,random_state=0)
	slf = svm.SVC(kernel="linear",c=1,gamma='auto')
	clf.fit(x_train,y_train)
	return clf.score(x_test,y_test)
def readData():
	imgList = []
	labelList = []
	label = 0
	
	for i in range(1,73):
		for j in range(1,11):
			image = cv2.imread("CAVIAR4REID/CAVIARa/"+"%04d"%i+"%03d"%j+".jpg")
			image = cv2.resize(image, (72, 144), interpolation=cv2.INTER_AREA)
			imgList.append(image)
			labelList.append(label)
	rootpath = "voc2005_1/VOC2005_1/PNGImages/"
	for i in listdir(rootpath):
		if i == "TUGraz_person":
			label = 0
		else:
			label = 1
		for j in listdir(rootpath+i):
			image = cv2.imread(rootpath+"/"+i+"/"+j)
			image = cv2.resize(image, (72, 144), interpolation=cv2.INTER_AREA)
			imgList.append(image)
			labelList.append(label)
	return imgList,labelList



#main
def main():
	#example()
	images,target = readData()
	Hogs = imageToHog(images)
	print(train(Hogs,target))
main()