# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 17:15:16 2018

@author: Kau
"""


import numpy as np
import matplotlib.pyplot as plt

def Manhattan(x, y):
    total = 0
    for i in range(len(x)):
        total += np.abs(x[i] - y[i])
    return total

def Euclidian(x, y):
    total = 0
    #print(y)
    for i in range(len(x)):
        total += np.square(np.abs(x[i] - y[i]))
    total = np.sqrt(total)
    return total

def knn_classifier(x_test, x_train, y_train, k, p):
    distances = []
    dist = 0;
    count2 = 0
    count4 = 0
    for i in range(len(x_train)):
        if p == 1:
            dist = Manhattan(x_test, x_train[i])
        elif p == 2:
            dist = Euclidian(x_test, x_train[i])
        distances.append((dist, y_train[i], i))
        #distances.append((dist, y_train[i]))
    distances.sort(key=lambda x: x[0]) #Sort to get distance values.
    neighbors = []
    for i in range(k):
        #neighbors.append(distances[i][1])
        neighbors.append((distances[i+1][1], distances[i+1][2]))
    for i in range(k):
        if neighbors[i][0] == 2:
            count2 += 1
        elif neighbors[i][0] == 4:
            count4 += 1
    if count2 > count4:
        return 2
    else:
        return 4
    
def main():
    BCan = np.genfromtxt('breast-cancer-wisconsin.data', usecols = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10), missing_values = '?', filling_values = '0', delimiter = ',', dtype = 'int')
    x_train = BCan[0:559, 0:9]
    y_train = BCan[0:559, 9]
    x_test = BCan[559:699, 0:9]
    y_test = BCan[559:699, 9]
    #print(x_test)
    k = 1
    p = 2
    True_Pos = 0
    False_Pos = 0
    True_Neg = 0
    False_Neg = 0
    y_pred_table = []
    numcorr = 0
    numtot = len(x_test)
    for i in range(len(x_test)):
        y_pred = knn_classifier(x_test[i], x_train, y_train, k, p)
        y_pred_table.append(y_pred)
    #print(y_pred_table)
    for i in range(len(y_pred_table)):
        if(y_pred_table[i] == y_test[i]):
            numcorr += 1
        if(y_pred_table[i] == y_test[i]) and y_train[i] == 4:
            True_Pos += 1
        if(y_pred_table[i] == y_test[i] and y_train[i] == 2):
            True_Neg += 1
        if(y_pred_table[i] != y_test[i] and y_train[i] == 4):
            False_Neg += 1
        if(y_pred_table[i] != y_test[i] and y_train[i] == 2):
            False_Pos += 1
    
    print('Accuracy: ' + str((True_Pos + True_Neg) / numtot)  +'\n')
    #print(numcorr/ numtot)
   
main()