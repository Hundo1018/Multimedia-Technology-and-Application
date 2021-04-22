from posix import XATTR_CREATE
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import *
from imutils import paths

train_path = "dataset/train"
training_names = os.listdir(train_path)

image_paths = []
image_classes = []
class_id = 0
for training_name in training_names:
	dir = os.path.join(train_path, training_name)
	class_path = list(paths.list_images(dir))
	image_paths += class_path
	image_classes += [class_id] * len(class_path)
	class_id += 1

sift = cv2.SIFT_create()

des_list = []

for image_path in image_paths:
	im = cv2.imread(image_path)
	im = cv2.resize(im, (300, 300))
	kpts = sift.detect(im)
	kpts, des = sift.compute(im, kpts)
	des_list.append((image_path, des))
	print("Image file path : ", image_path)

descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
	descriptors = np.vstack((descriptors, descriptor))

k = 20
voc, variance = kmeans(descriptors, k, 1)

im_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
	words, distance = vq(des_list[i][1], voc)
	for w in words:
		im_features[i][w] += 1

x = im_features
y = np.array(image_classes)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

clf = svm.SVC(kernel="linear",C=1,gamma='auto')
clf.fit(x_train,y_train)

print("predict")
print(clf.predict(x_train))
print(clf.predict(x_test))

print("Accuracy:")
print(clf.score(x_train,y_train))
print(clf.score(x_test,y_test))

stdslr = StandardScaler().fit(im_features)
im_features = stdslr.transform(im_features)

