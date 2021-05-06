import numpy as np
import cv2
import os
import random
import joblib
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import *
from imutils import paths

# 0: user loss
# 1: user tie
# 2: user win
def resultRockPaperScissors(user, bot):
	if user == 0:
		if bot == 0:
			return 1
		elif bot == 1:
			return 0
		elif bot == 2:
			return 2
		else:
			return 'bot ERROR'
	elif user == 1:
		if bot == 0:
			return 2
		elif bot == 1:
			return 1
		elif bot == 2:
			return 0
		else:
			return 'bot ERROR'
	elif user == 2:
		if bot == 0:
			return 0
		elif bot == 1:
			return 2
		elif bot == 2:
			return 1
		else:
			return 'bot ERROR'
	else:
		return 'user ERROR'


# 手部判斷 需要SVM 回傳預測結果
def reconHand(img, svm):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.resize(img, (500, 500))
	sift = cv2.SIFT_create()
	kp, des = sift.detectAndCompute(img, None)
	result = svm.predict(des.flatten())
	return result


# 攝影機 回傳img
def Camara():
	# 選擇第二隻攝影機
	cap = cv2.VideoCapture(1)
	while(True):
		# 從攝影機擷取一張影像
		ret, frame = cap.read()
		# 顯示圖片
		cv2.imshow('Camara', frame)
		# 若按下 q 鍵則離開迴圈
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	# 釋放攝影機
	cap.release()
	# 關閉所有 OpenCV 視窗
	cv2.destroyAllWindows()
	return frame


# status
# 0: user loss
# 1: user tie
# 2: user win
def combineImg(imgUser, imgBot, status):
	# load game result imagine for show
	imgGameWin = cv2.imread("gameWin.png")
	imgGameTie = cv2.imread("gameTie.png")
	imgGameLose = cv2.imread("gameLose.png")
	resultImg = np.ndarray()

	if status == 0:
		resultImg = np.hstack([imgUser, imgGameLose, imgBot])
	elif status == 1:
		resultImg = np.hstack([imgUser, imgGameTie, imgBot])
	elif status == 2:
		resultImg = np.hstack([imgUser, imgGameWin, imgBot])
	else:
		return 'combine ERROR'

	resultImg = cv2.resize(resultImg, (900, 300))
	return resultImg





# 0: 剪刀
# 1: 石頭
# 2: 布
def RockPaperScissors():
	# load bot imagine for show
	imgBotRock = cv2.imread("botRock.jpg")
	imgBotPaper  = cv2.imread("botPaper.jpg")
	imgBotScissors = cv2.imread("botScissors.jpg")

	svm = joblib.load("hand_svm.pkl")		# load pre train svm
	userImg = cv2.imread("user.jpg")		# load user hand input imagine
	userHand = reconHand(userImg, svm)		# hand to digit
	botHand = random.randint(0, 2)			# generate bot digit

	gameResult = resultRockPaperScissors(userHand, botHand)	# user win or loss or tie

	# set botImg
	if botHand == 0:
		botImg = imgBotScissors
	elif botHand == 1:
		botImg = imgBotRock
	elif botHand == 2:
		botImg = imgBotPaper

	gameResultImg = combineImg(userImg, botImg, gameResult)
	cv2.imshow("遊戲結果~", gameResultImg)

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
		im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		im = cv2.resize(im, (300, 200))
		kpts = sift.detect(im)
		kpts, des = sift.compute(im, kpts)
		des_list.append((image_path, des))
		print("Image file path : ", image_path)

	descriptors = des_list[0][1]
	for image_path, descriptor in des_list[1:]:
		descriptors = np.vstack((descriptors, descriptor))

	k = 30
	voc, variance = kmeans(descriptors, k, 5)

	im_features = np.zeros((len(image_paths), k), "float32")
	for i in range(len(image_paths)):
		words, distance = vq(des_list[i][1], voc)
		for w in words:
			im_features[i][w] += 1

	x = im_features
	y = np.array(image_classes)
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

	clf = svm.SVC()
	clf.fit(x_train,y_train)

	print("predict")
	print(clf.predict(x_train))
	print(clf.predict(x_test))

	print("Accuracy:")
	print(clf.score(x_train,y_train))
	print(clf.score(x_test,y_test))

	stdslr = StandardScaler().fit(im_features)
	im_features = stdslr.transform(im_features)

	return clf

#svm訓練區
def svmTeacher():
	train_path = "Data"
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

	sift = cv2.xfeatures2d.SIFT_create()

	des_list = []

	for image_path in image_paths:
		im = cv2.imread(image_path)
		im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		im = cv2.resize(im, (300, 200))
		kpts = sift.detect(im)
		kpts, des = sift.compute(im, kpts)
		des_list.append((image_path, des))
		print("Image file path : ", image_path)

	descriptors = des_list[1][1]
	for image_path, descriptor in des_list[1:]:
		if descriptor is not None:
			descriptors = np.vstack((descriptors, descriptor))

	k = 40
	voc, variance = kmeans(descriptors, k, 20)

	im_features = np.zeros((len(image_paths), k), "float32")
	for i in range(len(image_paths)):
		if des_list[i][1] is None:
			continue
		words, distance = vq(des_list[i][1], voc)
		for w in words:
			im_features[i][w] += 1
	stdslr = StandardScaler().fit(im_features)
	im_features = stdslr.transform(im_features)
	x = im_features
	y = np.array(image_classes)
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

	clf = svm.SVC()
	clf.fit(x_train,y_train)

	print("predict")
	print(clf.predict(x_train))
	print(clf.predict(x_test))

	print("Accuracy:")
	print(clf.score(x_train,y_train))
	print(clf.score(x_test,y_test))



	return clf
	pass

def main():
	# 訓練
	#clf = joblib.load("hand_svm.pkl")
	clf = svmTeacher()
	joblib.dump(clf,"hand_svm.pkl")
	# 剪刀石頭布程式，不包含訓練
	RockPaperScissors()
	pass

main()


'''
#	dataset/train/ ┬ car/img.png
#				   └ person/img.png
#
def train():
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
		im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		im = cv2.resize(im, (300, 200))
		kpts = sift.detect(im)
		kpts, des = sift.compute(im, kpts)
		des_list.append((image_path, des))
		print("Image file path : ", image_path)

	descriptors = des_list[0][1]
	for image_path, descriptor in des_list[1:]:
		descriptors = np.vstack((descriptors, descriptor))

	k = 20
	voc, variance = kmeans(descriptors, k, 5)

	im_features = np.zeros((len(image_paths), k), "float32")
	for i in range(len(image_paths)):
		words, distance = vq(des_list[i][1], voc)
		for w in words:
			im_features[i][w] += 1

	x = im_features
	y = np.array(image_classes)
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

	clf = svm.SVC()
	clf.fit(x_train,y_train)

	print("predict")
	print(clf.predict(x_train))
	print(clf.predict(x_test))

	print("Accuracy:")
	print(clf.score(x_train,y_train))
	print(clf.score(x_test,y_test))

	stdslr = StandardScaler().fit(im_features)
	im_features = stdslr.transform(im_features)

	return clf
'''