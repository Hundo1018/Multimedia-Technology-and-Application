import numpy as np
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def hamer(data, target, kernelP, CP, gammaP, sizeTest):
	X = data
	Y = target

	#分割訓練集與測試集
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=sizeTest, random_state=0)

	#SVM
	result = svm.SVC(kernel=kernelP, C=CP, gamma=gammaP)
	#餵SVM訓練集的data, target
	result.fit(X_train, Y_train)

	# 回傳準確率
	return print(result.score(X_train, Y_train)), print(result.score(X_test, Y_test))


def main():
	#還要再修改
	hamer(datasets.load_boston().data, datasets.load_boston().target, 'linear', 1, 'auto', 0.2)

main()