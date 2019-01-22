# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 17:15:16 2018

@author: Kau
"""


import numpy as np
import statistics as stat
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
        if dist != 0:
            distances.append((dist, y_train[i], i))
        #distances.append((dist, y_train[i]))
    distances.sort(key=lambda x: x[0]) #Sort to get distance values.
            
    neighbors = []
    for i in range(k):
        #neighbors.append(distances[i][1])
        neighbors.append((distances[i][1], distances[i][2]))
    for i in range(k):
        if neighbors[i][0] == 2:
            count2 += 1
        elif neighbors[i][0] == 4:
            count4 += 1
    if count2 > count4:
        return 2
    else:
        return 4

def mean(x):
    total = 0;
    for i in range(len(x)):
        total += x[i]
    return total / len(x)

def main():
    BCan = np.genfromtxt('breast-cancer-wisconsin.data', usecols = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10), missing_values = '?', filling_values = '0', delimiter = ',', dtype = 'int')
    S_BCan = BCan
    np.random.shuffle(S_BCan)
    #p = x_test (partitioned data)
    p1 = S_BCan[0:70, 0:9]
    p2 = S_BCan[70:140, 0:9]
    p3 = S_BCan[140:210, 0:9]
    p4 = S_BCan[210:280, 0:9]
    p5 = S_BCan[280:350, 0:9]
    p6 = S_BCan[350:420, 0:9]
    p7 = S_BCan[420:490, 0:9]
    p8 = S_BCan[490:560, 0:9]
    p9 = S_BCan[560:630, 0:9]
    p10 = S_BCan[630:699, 0:9]
    #t = y_test (labels)
    t1 = S_BCan[0:70, 9] 
    t2 = S_BCan[70:140, 9]
    t3 = S_BCan[140:210, 9]
    t4 = S_BCan[210:280, 9]
    t5 = S_BCan[280:350, 9]
    t6 = S_BCan[350:420, 9]
    t7 = S_BCan[420:490, 9]
    t8 = S_BCan[490:560, 9]
    t9 = S_BCan[560:630, 9]
    t10 = S_BCan[630:699, 9]
    #full test cross-validation info
    Test1 = np.asarray(p2.tolist() + p3.tolist() + p4.tolist() + p5.tolist() + p6.tolist() + p7.tolist() + p8.tolist() + p9.tolist() + p10.tolist()) 
    Test2 = np.asarray(p1.tolist() + p3.tolist() + p4.tolist() + p5.tolist() + p6.tolist() + p7.tolist() + p8.tolist() + p9.tolist() + p10.tolist())
    Test3 = np.asarray(p1.tolist() + p2.tolist() + p4.tolist() + p5.tolist() + p6.tolist() + p7.tolist() + p8.tolist() + p9.tolist() + p10.tolist())
    Test4 = np.asarray(p1.tolist() + p2.tolist() + p3.tolist() + p5.tolist() + p6.tolist() + p7.tolist() + p8.tolist() + p9.tolist() + p10.tolist())
    Test5 = np.asarray(p1.tolist() + p2.tolist() + p3.tolist() + p4.tolist() + p6.tolist() + p7.tolist() + p8.tolist() + p9.tolist() + p10.tolist())
    Test6 = np.asarray(p1.tolist() + p2.tolist() + p3.tolist() + p4.tolist() + p5.tolist() + p7.tolist() + p8.tolist() + p9.tolist() + p10.tolist())
    Test7 = np.asarray(p1.tolist() + p2.tolist() + p3.tolist() + p4.tolist() + p5.tolist() + p6.tolist() + p8.tolist() + p9.tolist() + p10.tolist())
    Test8 = np.asarray(p1.tolist() + p2.tolist() + p3.tolist() + p4.tolist() + p5.tolist() + p6.tolist() + p7.tolist() + p9.tolist() + p10.tolist())
    Test9 = np.asarray(p1.tolist() + p2.tolist() + p3.tolist() + p4.tolist() + p5.tolist() + p6.tolist() + p7.tolist() + p8.tolist() + p10.tolist())
    Test10 = np.asarray(p1.tolist() + p2.tolist() + p3.tolist() + p4.tolist() + p5.tolist() + p6.tolist() + p7.tolist() + p8.tolist() + p9.tolist())
    # populate y_trains (labels for the training set)
    y1 = np.asarray(t2.tolist() + t3.tolist() + t4.tolist() + t5.tolist() + t6.tolist() + t7.tolist() + t8.tolist() + t9.tolist() + t10.tolist()) 
    y2 = np.asarray(t1.tolist() + t3.tolist() + t4.tolist() + t5.tolist() + t6.tolist() + t7.tolist() + t8.tolist() + t9.tolist() + t10.tolist())
    y3 = np.asarray(t1.tolist() + t2.tolist() + t4.tolist() + t5.tolist() + t6.tolist() + t7.tolist() + t8.tolist() + t9.tolist() + t10.tolist())
    y4 = np.asarray(t1.tolist() + t2.tolist() + t3.tolist() + t5.tolist() + t6.tolist() + t7.tolist() + t8.tolist() + t9.tolist() + t10.tolist())
    y5 = np.asarray(t1.tolist() + t2.tolist() + t3.tolist() + t4.tolist() + t6.tolist() + t7.tolist() + t8.tolist() + t9.tolist() + t10.tolist())
    y6 = np.asarray(t1.tolist() + t2.tolist() + t3.tolist() + t4.tolist() + t5.tolist() + t7.tolist() + t8.tolist() + t9.tolist() + t10.tolist())
    y7 = np.asarray(t1.tolist() + t2.tolist() + t3.tolist() + t4.tolist() + t5.tolist() + t6.tolist() + t8.tolist() + t9.tolist() + t10.tolist())
    y8 = np.asarray(t1.tolist() + t2.tolist() + t3.tolist() + t4.tolist() + t5.tolist() + t6.tolist() + t7.tolist() + t9.tolist() + t10.tolist())
    y9 = np.asarray(t1.tolist() + t2.tolist() + t3.tolist() + t4.tolist() + t5.tolist() + t6.tolist() + t7.tolist() + t8.tolist() + t10.tolist())
    y10 = np.asarray(t1.tolist() + t2.tolist() + t3.tolist() + t4.tolist() + t5.tolist() + t6.tolist() + t7.tolist() + t8.tolist() + t9.tolist())
    
    
    y_pred_table = [] #holds y-predicts for each iteration
    
    for p in range(1, 3): #get p trials: (p = 1, p = 2)
        acc_mean = [] #hold means for each acc_vals
        acc_std = [] #hold standard deviations for each mean
        sensi_vals = [] #holds sensitivities for each 10-f validation, = TP / (TP + FN) 
        sensi_mean = [] #hold means for each fold of sensi_means
        sensi_std = [] # hold standard deviation for each mean
        speci_vals = [] #holds specificities for each 10-f validation, = TN / (TN + FP)
        speci_mean = [] #hold means for each fold of speci_vals.
        speci_std = [] #hold standard deviations for each mean 
        print('For p = ' + str(p) + '\n')
        for k in range(1, 11): #get k, which is 1-10
            print('For neighbors k = ' + str(k) + '\n')
            acc_vals = [] #holds accuracies for each 10-f validation, = TP+TN / (TP + TN + FP + FN) [ assume 4 = positive, 2 = neg ]
            print('First Fold Validation:')
            numtot = len(p1)
            #reset values per validation loop.
            y_pred_table = []
            True_Pos = 0
            True_Neg = 0
            False_Pos = 0
            False_Neg = 0
            for i in range(len(p1)):
                y_pred = knn_classifier(p1[i], Test1, y1, k, p)
                y_pred_table.append(y_pred)
            for i in range(len(y_pred_table)):
                #record data, positive = 4 (malignant), negative = 2.
                if(y_pred_table[i] == t1[i]) and t1[i] == 4:
                    True_Pos += 1
                if(y_pred_table[i] == t1[i] and t1[i] == 2):
                    True_Neg += 1
                if(y_pred_table[i] != t1[i] and t1[i] == 4):
                    False_Neg += 1
                if(y_pred_table[i] != t1[i] and t1[i] == 2):
                    False_Pos += 1
            #Use recorded values to append values.
            acc_vals.append((True_Pos + True_Neg) / float(numtot))
            sensi_vals.append((True_Pos) / (True_Pos + False_Neg))
            speci_vals.append((True_Neg) / (True_Neg + False_Pos))
            
            print('Second Fold Validation:')
            numtot = len(p2)
            y_pred_table = []
            True_Pos = 0
            True_Neg = 0
            False_Pos = 0
            False_Neg = 0
            for i in range(len(p2)):
                y_pred = knn_classifier(p2[i], Test2, y2, k, p)
                y_pred_table.append(y_pred)
            for i in range(len(y_pred_table)):
                if(y_pred_table[i] == t2[i]) and t2[i] == 4:
                    True_Pos += 1
                if(y_pred_table[i] == t2[i] and t2[i] == 2):
                    True_Neg += 1
                if(y_pred_table[i] != t2[i] and t2[i] == 4):
                    False_Neg += 1
                if(y_pred_table[i] != t2[i] and t2[i] == 2):
                    False_Pos += 1
            acc_vals.append((True_Pos + True_Neg) / float(numtot))
            sensi_vals.append((True_Pos) / (True_Pos + False_Neg))
            speci_vals.append((True_Neg) / (True_Neg + False_Pos))
            
            
            print('Third Fold Validation:')
            numtot = len(p3)
            y_pred_table = []
            True_Pos = 0
            True_Neg = 0
            False_Pos = 0
            False_Neg = 0
            for i in range(len(p3)):
                y_pred = knn_classifier(p3[i], Test3, y3, k, p)
                y_pred_table.append(y_pred)
            for i in range(len(y_pred_table)):
                if(y_pred_table[i] == t3[i]) and t3[i] == 4:
                    True_Pos += 1
                if(y_pred_table[i] == t3[i] and t3[i] == 2):
                    True_Neg += 1
                if(y_pred_table[i] != t3[i] and t3[i] == 4):
                    False_Neg += 1
                if(y_pred_table[i] != t3[i] and t3[i] == 2):
                    False_Pos += 1
            acc_vals.append((True_Pos + True_Neg) / float(numtot))
            sensi_vals.append((True_Pos) / (True_Pos + False_Neg))
            speci_vals.append((True_Neg) / (True_Neg + False_Pos))
            
            
            print('Fourth Fold Validation:')
            numtot = len(p4)
            y_pred_table = []
            True_Pos = 0
            True_Neg = 0
            False_Pos = 0
            False_Neg = 0
            for i in range(len(p4)):
                y_pred = knn_classifier(p4[i], Test4, y4, k, p)
                y_pred_table.append(y_pred)
            for i in range(len(y_pred_table)):
                if(y_pred_table[i] == t4[i]) and t4[i] == 4:
                    True_Pos += 1
                if(y_pred_table[i] == t4[i] and t4[i] == 2):
                    True_Neg += 1
                if(y_pred_table[i] != t4[i] and t4[i] == 4):
                    False_Neg += 1
                if(y_pred_table[i] != t4[i] and t4[i] == 2):
                    False_Pos += 1
            acc_vals.append((True_Pos + True_Neg) / float(numtot))
            sensi_vals.append((True_Pos) / (True_Pos + False_Neg))
            speci_vals.append((True_Neg) / (True_Neg + False_Pos))
            
            
            print('Fifth Fold Validation:')
            numtot = len(p5)
            y_pred_table = []
            True_Pos = 0
            True_Neg = 0
            False_Pos = 0
            False_Neg = 0
            for i in range(len(p5)):
                y_pred = knn_classifier(p5[i], Test5, y5, k, p)
                y_pred_table.append(y_pred)
            for i in range(len(y_pred_table)):
                if(y_pred_table[i] == t5[i]) and t5[i] == 4:
                    True_Pos += 1
                if(y_pred_table[i] == t5[i] and t5[i] == 2):
                    True_Neg += 1
                if(y_pred_table[i] != t5[i] and t5[i] == 4):
                    False_Neg += 1
                if(y_pred_table[i] != t5[i] and t5[i] == 2):
                    False_Pos += 1
            acc_vals.append((True_Pos + True_Neg) / float(numtot))
            sensi_vals.append((True_Pos) / (True_Pos + False_Neg))
            speci_vals.append((True_Neg) / (True_Neg + False_Pos))
            
            
            print('Sixth Fold Validation:')
            numtot = len(p6)
            y_pred_table = []
            True_Pos = 0
            True_Neg = 0
            False_Pos = 0
            False_Neg = 0
            for i in range(len(p6)):
                y_pred = knn_classifier(p6[i], Test6, y6, k, p)
                y_pred_table.append(y_pred)
            for i in range(len(y_pred_table)):
                if(y_pred_table[i] == t6[i]) and t6[i] == 4:
                    True_Pos += 1
                if(y_pred_table[i] == t6[i] and t6[i] == 2):
                    True_Neg += 1
                if(y_pred_table[i] != t6[i] and t6[i] == 4):
                    False_Neg += 1
                if(y_pred_table[i] != t6[i] and t6[i] == 2):
                    False_Pos += 1
            acc_vals.append((True_Pos + True_Neg) / float(numtot))
            sensi_vals.append((True_Pos) / (True_Pos + False_Neg))
            speci_vals.append((True_Neg) / (True_Neg + False_Pos))
            
            
            print('Seventh Fold Validation:')
            numtot = len(p7)
            y_pred_table = []
            True_Pos = 0
            True_Neg = 0
            False_Pos = 0
            False_Neg = 0
            for i in range(len(p7)):
                y_pred = knn_classifier(p7[i], Test7, y7, k, p)
                y_pred_table.append(y_pred)
            for i in range(len(y_pred_table)):
                if(y_pred_table[i] == t7[i]) and t7[i] == 4:
                    True_Pos += 1
                if(y_pred_table[i] == t7[i] and t7[i] == 2):
                    True_Neg += 1
                if(y_pred_table[i] != t7[i] and t7[i] == 4):
                    False_Neg += 1
                if(y_pred_table[i] != t7[i] and t7[i] == 2):
                    False_Pos += 1
            acc_vals.append((True_Pos + True_Neg) / float(numtot))
            sensi_vals.append((True_Pos) / (True_Pos + False_Neg))
            speci_vals.append((True_Neg) / (True_Neg + False_Pos))
            
            
            print('Eighth Fold Validation:')
            numtot = len(p8)
            y_pred_table = []
            True_Pos = 0
            True_Neg = 0
            False_Pos = 0
            False_Neg = 0
            for i in range(len(p8)):
                y_pred = knn_classifier(p8[i], Test8, y8, k, p)
                y_pred_table.append(y_pred)
            for i in range(len(y_pred_table)):
                if(y_pred_table[i] == t8[i]) and t8[i] == 4:
                    True_Pos += 1
                if(y_pred_table[i] == t8[i] and t8[i] == 2):
                    True_Neg += 1
                if(y_pred_table[i] != t8[i] and t8[i] == 4):
                    False_Neg += 1
                if(y_pred_table[i] != t8[i] and t8[i] == 2):
                    False_Pos += 1
            acc_vals.append((True_Pos + True_Neg) / float(numtot))
            sensi_vals.append((True_Pos) / (True_Pos + False_Neg))
            speci_vals.append((True_Neg) / (True_Neg + False_Pos))
            
            
            print('Ninth Fold Validation:')
            numtot = len(p9)
            y_pred_table = []
            True_Pos = 0
            True_Neg = 0
            False_Pos = 0
            False_Neg = 0
            for i in range(len(p9)):
                y_pred = knn_classifier(p9[i], Test9, y9, k, p)
                y_pred_table.append(y_pred)
            for i in range(len(y_pred_table)):
                if(y_pred_table[i] == t9[i]) and t9[i] == 4:
                    True_Pos += 1
                if(y_pred_table[i] == t9[i] and t9[i] == 2):
                    True_Neg += 1
                if(y_pred_table[i] != t9[i] and t9[i] == 4):
                    False_Neg += 1
                if(y_pred_table[i] != t9[i] and t9[i] == 2):
                    False_Pos += 1
            acc_vals.append((True_Pos + True_Neg) / float(numtot))
            sensi_vals.append((True_Pos) / (True_Pos + False_Neg))
            speci_vals.append((True_Neg) / (True_Neg + False_Pos))
            
            
            print('Tenth Fold Validation:')
            numtot = len(p10)
            y_pred_table = []
            True_Pos = 0
            True_Neg = 0
            False_Pos = 0
            False_Neg = 0
            for i in range(len(p10)):
                y_pred = knn_classifier(p10[i], Test10, y10, k, p)
                y_pred_table.append(y_pred)
            for i in range(len(y_pred_table)):
                if(y_pred_table[i] == t10[i]) and t10[i] == 4:
                    True_Pos += 1
                if(y_pred_table[i] == t10[i] and t10[i] == 2):
                    True_Neg += 1
                if(y_pred_table[i] != t10[i] and t10[i] == 4):
                    False_Neg += 1
                if(y_pred_table[i] != t10[i] and t10[i] == 2):
                    False_Pos += 1
            acc_vals.append((True_Pos + True_Neg) / float(numtot))
            sensi_vals.append((True_Pos) / float(True_Pos + False_Neg))
            speci_vals.append((True_Neg) / float(True_Neg + False_Pos))
            #append at the end, mean of the cross-validation's accuracies, sensitivities, specificities.
            acc_mean.append(mean(acc_vals))
            sensi_mean.append(mean(sensi_vals))
            speci_mean.append(mean(speci_vals))
            #append all standard deviations at the end for the three types of data.
            acc_std.append(stat.stdev(acc_vals))
            sensi_std.append(stat.stdev(sensi_vals))
            speci_std.append(stat.stdev(speci_vals))
        
        
        x = np.arange(1, 11)
        #change x/y axes, output 
        #get accuracy graph
        plt.title('Accuracy, p=' + str(p))   
        plt.xlabel('Neighbor Number')
        plt.errorbar(x, y = np.asarray(acc_mean), yerr = np.asarray(acc_std), linestyle='-', marker='o')#x = 1-10, y = values to plot, y_err = std_deviations, linestyles = line exists, marker of circle.
        plt.savefig('Accuracy, p=' + str(p), dpi = 100)
        plt.show()
        #get sensitivity graph.
        plt.title('Sensitivity, p=' + str(p))
        plt.xlabel('Neighbor Number')
        plt.errorbar(x, y = np.asarray(sensi_mean), yerr = np.asarray(sensi_std), linestyle='-', marker='o')
        plt.savefig('Sensitivity, p=' + str(p), dpi = 100)
        plt.show()
        #get specificity graph.
        plt.title('Specificity, p=' + str(p))
        plt.xlabel('Neighbor Number')
        plt.errorbar(x, y = np.asarray(speci_mean), yerr = np.asarray(speci_std), linestyle='-', marker='o')
        plt.savefig('Specificity, p=' + str(p), dpi = 100)
        plt.show()
   
main()