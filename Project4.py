import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split


#
# svm參數參考https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
#
# kernel: {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
# C     : float, default=1.0
# gamma : {‘scale’, ‘auto’} or float, default=’scale’
#
def foo(in_x, in_y, in_kernel, in_c, in_gamma, in_sizeTest):
    #分割為train,test
    x_train, x_test, y_train, y_test = train_test_split(in_x, in_y, test_size=in_sizeTest, random_state=0)

    svmModel = svm.SVC(kernel=in_kernel, C=in_c, gamma=in_gamma)

    svmModel.fit(x_train, y_train)
    #train資料集準確率,test資料及準確率
    return [svmModel.score(x_train, y_train), svmModel.score(x_test, y_test)]


#
# main function
#
def main():
    #
    # wine = datasets.load_wine()
    # x=foo(wine.data, wine.target, 'linear', 1, 'auto', 0.8)


main()