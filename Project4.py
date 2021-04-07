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
    wine = datasets.load_wine()
    x = []
    y = []
    for i in range(len(wine.data[:,0])):
        x.append( [wine.data[:,0][i],wine.data[:,2][i]])
        y.append( wine.target[i])
    result = foo(x,y,'poly',5,0.5,0.2)[1]
    print('kernel:poly\nc:5\ngamma:0.5\nsize_test:0.2\n準確率',result)


    #sizeTest
    plt.figure(1, (10, 6))
    for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        scoreList = []
        for size in np.linspace(0.05, 0.95, 100):
            scoreList.append(foo(wine.data, wine.target, kernel, 1, 'auto', size))
        plt.plot(np.linspace(0.05, 0.95, 100), scoreList, label=kernel)

    plt.legend()
    plt.xlabel('test_size')
    plt.ylabel('score')
    plt.title("sizeTest")



main()
