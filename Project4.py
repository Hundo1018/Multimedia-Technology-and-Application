import numpy as np
from sklearn import datasets
from sklearn import metrics


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt



scores = {}
algorithms = {"KNN":["uniform","distance"],"AdaBoost":["SAMME","SAMME.R"]}




def Classifier(dataSet,func,size):
	x = dataSet.data
	y = dataSet.target
	x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=size,random_state=0)
	func.fit(x_train,y_train)
	
	y_pred = func.predict(x_test)
	print(str(y_pred)+"\n"+str(y_test))
	print("score:",func.score(x_test,y_test),"\n")
	return func.score(x_test,y_test)

def main():
	print("\nKNN\nweight:distance\nn_neighbors:10\ntest_size:0.1")
	Classifier(datasets.load_iris(),KNeighborsClassifier(n_neighbors=10,weights='distance'),0.1)
	print("AdaBoost\nalgorithm:SAMME\nn_estimators:30\ntest_size:0.2")
	Classifier(datasets.load_iris(),AdaBoostClassifier(n_estimators=30,algorithm='SAMME'),0.2)

	'''

	n_range = range(1,50)
	for i in algorithms.keys():
		scores[i] = {}
		for j in algorithms[i]:
			scores[i][j] = []
	#
	for n in n_range:
		scores["KNN"]['distance'].append( Classifier(datasets.load_iris(),KNeighborsClassifier(n_neighbors=n,weights='distance'),0.2))
		scores["KNN"]['uniform'].append ( Classifier(datasets.load_iris(),KNeighborsClassifier(n_neighbors=n,weights='uniform'),0.2))
		scores["AdaBoost"]['SAMME'].append(Classifier(datasets.load_iris(),AdaBoostClassifier(n_estimators=n,algorithm='SAMME'),0.2))
		scores["AdaBoost"]['SAMME.R'].append(Classifier(datasets.load_iris(),AdaBoostClassifier(n_estimators=n,algorithm='SAMME.R'),0.2))

	fig, axs = plt.subplots(4, 1, constrained_layout=True)

	axs[0].plot(n_range,scores['KNN']["uniform"],label = "uniform")
	axs[0].plot(n_range,scores['KNN']["distance"],label= "distance")
	axs[0].legend()
	axs[0].set_xlabel("n_neighbors")
	axs[0].set_ylabel("score")
	axs[0].set_title("KNN")
	axs[1].plot(n_range,scores['AdaBoost']["SAMME"],label = "SAMME")
	axs[1].plot(n_range,scores['AdaBoost']["SAMME.R"],label= "SAMME.R")
	axs[1].legend()
	axs[1].set_xlabel("n_estimators")
	axs[1].set_ylabel("score")
	axs[1].set_title("AdaBoost")

	size_range = np.arange(.1,1.0,.1)
	for i in algorithms.keys():
		scores[i] = {}
		for j in algorithms[i]:
			scores[i][j] = []
	for s in size_range:
		scores["KNN"]['distance'].append( Classifier(datasets.load_iris(),KNeighborsClassifier(n_neighbors=10,weights='distance'),s))
		scores["KNN"]['uniform'].append ( Classifier(datasets.load_iris(),KNeighborsClassifier(n_neighbors=10,weights='uniform'),s))
		scores["AdaBoost"]['SAMME'].append(Classifier(datasets.load_iris(),AdaBoostClassifier(n_estimators=30,algorithm='SAMME'),s))
		scores["AdaBoost"]['SAMME.R'].append(Classifier(datasets.load_iris(),AdaBoostClassifier(n_estimators=30,algorithm='SAMME.R'),s))

	axs[2].plot(size_range,scores['KNN']["uniform"],label = "uniform")
	axs[2].plot(size_range,scores['KNN']["distance"],label= "distance")
	axs[2].legend()
	axs[2].set_xlabel("size")
	axs[2].set_ylabel("score")
	axs[2].set_title("KNN")
	axs[3].plot(size_range,scores['AdaBoost']["SAMME"],label = "SAMME")
	axs[3].plot(size_range,scores['AdaBoost']["SAMME.R"],label= "SAMME.R")
	axs[3].legend()
	axs[3].set_xlabel("size")
	axs[3].set_ylabel("score")
	axs[3].set_title("AdaBoost")
	plt.show()
	'''
main()