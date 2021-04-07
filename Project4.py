import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split


#
# svm參數參考https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
#
# kernel: {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’}, default=’rbf’
# C     : float, default=1.0
# gamma : {‘scale’, ‘auto’} or float, default=’scale’
#
def foo(in_x, in_y, in_kernel, in_c, in_gamma, in_sizeTest):
    #分割為train,test
    x_train, x_test, y_train, y_test = train_test_split(in_x, in_y, test_size=in_sizeTest, random_state=0)

    svmModel = svm.SVC(kernel=in_kernel, C=in_c, gamma=in_gamma)

    svmModel.fit(x_train, y_train)
    #test資料及準確率
    return svmModel.score(x_test, y_test)


#
# main function
#
def main():
    kernelList = ['linear', 'poly', 'rbf', 'sigmoid']
    scoreList = []

    wine = datasets.load_wine()
    #example x=foo(wine.data, wine.target, 'linear', 1, 'auto', 0.8)


    #sizeTest
    plt.figure(1, (10, 6))
    for kernel in kernelList:
        scoreList = []
        for size in np.linspace(0.05, 0.95, 100):
            scoreList.append(foo(wine.data, wine.target, kernel, 1, 'auto', size))
        plt.plot(np.linspace(0.05, 0.95, 100), scoreList, label=kernel)

    plt.legend()
    plt.xlabel('test_size')
    plt.ylabel('score')
    plt.title("sizeTest")



    #example
    plt.figure(2)

    plt.legend()
    plt.xlabel('test_size')
    plt.ylabel('score')
    plt.title("sizeTest")


    plt.show()


main()
