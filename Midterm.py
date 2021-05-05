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


# 0: 剪刀
# 1: 石頭
# 2: 布
SCISSORS = 0
ROCK = 1
PAPER = 2
# 10: user loss
# 11: user tie
# 12: user win
LOSS = 10
TIE = 11
WIN = 12


# 剪刀石頭布遊戲邏輯
def resultRockPaperScissors(user, bot):
	if user == SCISSORS:
		if bot == SCISSORS:
			return TIE
		elif bot == ROCK:
			return LOSS
		elif bot == PAPER:
			return WIN
		else:
			return -1
	elif user == ROCK:
		if bot == SCISSORS:
			return WIN
		elif bot == ROCK:
			return TIE
		elif bot == PAPER:
			return LOSS
		else:
			return -1
	elif user == PAPER:
		if bot == SCISSORS:
			return LOSS
		elif bot == ROCK:
			return WIN
		elif bot == PAPER:
			return TIE
		else:
			return -1
	else:
		return -1


# 手部辨識
def identifyHand(img, svm):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.resize(img, (500, 500))
	sift = cv2.SIFT_create()
	kpts, des = sift.detectAndCompute(img, None)
	result = svm.predict(kpts.flatten())
	return result


# 勝負圖片合成
def combineImg(imgUser, imgBot, status):
	# 載入勝負圖片
	imgGameWin = cv2.imread("gameWin.png")
	imgGameTie = cv2.imread("gameTie.png")
	imgGameLose = cv2.imread("gameLose.png")
	resultImg = np.ndarray()

	if status == LOSS:
		resultImg = np.hstack([imgUser, imgGameLose, imgBot])
	elif status == TIE:
		resultImg = np.hstack([imgUser, imgGameTie, imgBot])
	elif status == WIN:
		resultImg = np.hstack([imgUser, imgGameWin, imgBot])
	else:
		return 'combine ERROR'

	resultImg = cv2.resize(resultImg, (900, 300))
	return resultImg


# 剪刀石頭布主程式
def RockPaperScissors(user_hand_img, model):
	# 載入機器人的手部圖片
	imgBotRock = cv2.imread("botRock.jpg")
	imgBotPaper  = cv2.imread("botPaper.jpg")
	imgBotScissors = cv2.imread("botScissors.jpg")
	# 手部辨識 & BOT隨機產生手勢
	userHand = identifyHand(user_hand_img, model)
	botHand = random.randint(0, 2)
	# 判斷勝負
	gameResult = resultRockPaperScissors(userHand, botHand)

	# 設定BOT手部圖片
	if botHand == SCISSORS:
		botImg = imgBotScissors
	elif botHand == ROCK:
		botImg = imgBotRock
	elif botHand == PAPER:
		botImg = imgBotPaper

	gameResultImg = combineImg(user_hand_img, botImg, gameResult)
	cv2.imshow("遊戲結果~", gameResultImg)


# svm訓練區
def svmTraining(X, Y, model_name):
	# SVM 訓練
	model = svm.SVC(kernel='linear', C=1, gamma='auto')
	model.fit(X, Y)

	print("此次訓練自我良好準確率為: ", model.score(X, Y))

	# 保存model
	joblib.dump(model, model_name + '.pkl')


# svm預測區
def svmPrediction():
	pass


# 訓練集產生器
def TrainingDataGenerator():


def main():
	# 訓練
	X, Y = TrainingDataGenerator()
	svmTraining()

	# 剪刀石頭布主程式，不包含訓練
	# RockPaperScissors()

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