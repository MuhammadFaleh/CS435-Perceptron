import numpy as np
import matplotlib.pyplot as pt
from csv import *


def load_Data(file_name):
    list_all = list()
    list1 = list()
    list2 = list()
    list3 = list()
    with open(file_name, 'r') as input1:
        read = reader(input1)
        for row in read:
            if not row:
                continue
            list_all.append(row)
    length = int(len(list_all) / 3)
    np.reshape(list_all, (5, len(list_all)))
    for i in range(length):
        list1.append(list_all[:][i])
        list2.append(list_all[:][i + length])
        list3.append(list_all[:][i + (2 * length)])
    print(list3)
    list1_2 = np.concatenate((list1, list2))
    list2_3 = np.concatenate((list2, list3))
    list3_1 = np.concatenate((list1, list3))
    return list1_2, list2_3, list3_1


def shuffle(list1_2, list2_3, list3_1):
    np.random.shuffle(list1_2)
    np.random.shuffle(list2_3)
    np.random.shuffle(list3_1)
    list1_2 = replace(list1_2, "class-1", "class-2")
    list2_3 = replace(list2_3, "class-2", "class-3")
    print(list2_3)
    list3_1 = replace(list3_1, "class-1", "class-3")
    return list1_2, list2_3, list3_1


def replace(list1, Str1, Str2):
    list1[list1 == Str1] = 1.0
    list1[list1 == Str2] = -1.0
    return list1


def GetY_X(list1):
    list1N = np.array(list1[:, :4], dtype=float)
    y = np.array(list1[:, -1], dtype=float)
    #list1N = list(np.float_(list1N))
    print(list1N)
    #y = list(np.float_(y))
    return list1N, y


def Perceptron(NumOfIt, list1, yTrain, list_test, yTest):
    W = np.array([0.0, 0.0, 0.0, 0.0])
    bais = 0.0
    Fscore = list()
    error_train = list()
    correct_train = list()
    aList = list()
    error_test = list()
    correct_test = list()
    correct_test_con = list()
    for i in range(NumOfIt):
        error = 0.0
        correct = 0.0
        for x, y in zip(list1, yTrain):
            a = Perdict(x, W, bais)

            if (y * a) <= 0:
                error += 1.0
                W = np.add(W, np.multiply(x, y))
                bais += y
            else:
                correct += 1.0
        predictions = looptest(list_test, yTest, W, bais)
        cur_test_acc, cur_test_err = accuracy(predictions, yTest)
        #cur_train_acc, cur_train_err = accuracy(aList, yTrain)
        tp, fp, tn, fn = getTpnfpn(yTest, predictions)
        correct_test.append(cur_test_acc)
        correct_test_con.append(Accuracy(tp,tn,fp,fn))
        Fscore.append(Percision(tp, fp, fn))
        #tpT, fpT, tnT, fnT = getTpnfpn(yTrain, aList)
        correct_train.append(((correct / len(yTrain)) * 100.0))
        error_test.append(cur_test_err)
        error_train.append(error)

    #print(z)
    return correct_test, error_test, correct_train, error_train, Fscore,  correct_test_con


def looptest(list1, yT, W, bais):
    incorrect = 0.0
    correct = 0.0
    predictions = list()
    z = 0
    for x, y in zip(list1, yT):
        #print(yT)
        predict = Perdict(x, W, bais)
        predictions.append(predict)
        #print(' x = ', list1, "true y = ", y, "preditct", predict, end='')
        if (y == predict):
            correct += 1.0
        else:
            incorrect += 1.0

    total = incorrect + correct
    #print(correct, " - ", incorrect)
    #print("testing z = ", z)
    #print(len(yT))
    return predictions


def Perdict(list1, W, bais):
    a = np.dot(list1, W) + bais
    return np.sign(a)


def Accuracy(tp, tn, fp, fn):
    Accuracy = (tp+tn)/(tp+tn+fp+fn)
    return Accuracy * 100.0


def accuracy(list1, y):
    correct = 0.0
    incorrect = 0.0
    for x, y in zip(list1, y):
        if y == x:
            correct += 1.0
        else:
            incorrect += 1.0
        total = incorrect + correct
    return ((correct / total) * 100.0), incorrect


def Percision(tp, fp, fn):
    if fp == 0 and tp == 0:
        precision = -1
        return -1
    else:
        precision = tp / (tp + fp)
        return Fscore(tp, fn, precision)


def Fscore(tp, fn, Percision):
    Recall = tp / (tp + fn)
    Fscore = (2*Percision*Recall)/(Percision+Recall)
    return Fscore


def getTpnfpn(y, predict):
    tp = 0; fp = 0
    tn = 0; fn = 0
    for x, y in zip(predict, y):
        if y == x and y < 0:
            tn += 1
        if y == x and y > 0:
            tp += 1
        if y != x and x < 0:
            fn += 1
        if y != x and x > 0:
            fp += 1

    return tp, fp, tn, fn


def plot(accuracy1, error1, accuracyTrain1, errorTrain1, Fscore1,
              accuracy2, error2, accuracyTrain2, errorTrain2, Fscore2,
              accuracy3, error3, accuracyTrain3, errorTrain3, Fscore3):

    figure, axis = pt.subplots(3, 3)

    # For First class accuracy
    axis[0, 0].plot(accuracy1, label="testing Accuracy")
    axis[0, 0].plot(accuracyTrain1, color='red', label="Training Accuracy")
    axis[0, 0].set_title("first class accuracy")
    axis[0, 0].legend(loc='upper center')

    # For First class error
    axis[0, 1].plot(error1, color='red', linewidth=1, markersize=4, label="Test error")
    axis[0, 1].plot(errorTrain1, color='blue', linewidth=1, markersize=4, label="Train error")
    axis[0, 1].set_title("first class error")
    axis[0, 1].legend(loc='upper center')

    # For First class Fscore
    axis[0, 2].plot(Fscore1, marker='o', markerfacecolor='black', color='red', linewidth=1, markersize=4)
    axis[0, 2].set_title("first class F-Score(-1 means N/A) ")

    # For second class accuracy
    axis[1, 0].plot(accuracy2, label="testing Accuracy")
    axis[1, 0].plot(accuracyTrain2, color='red', label="Training Accuracy")
    axis[1, 0].set_title("second class accuracy")
    axis[1, 0].legend(loc='upper center')

    # For second class error
    axis[1, 1].plot(error2, color='red', linewidth=1, markersize=4, label="Test error")
    axis[1, 1].plot(errorTrain2, color='blue', linewidth=1, markersize=4, label="Train error")
    axis[1, 1].set_title("second class error")
    axis[1, 1].legend(loc='upper center')

    # For second class Fscore
    axis[1, 2].plot(Fscore2, marker='o', markerfacecolor='black', color='red', linewidth=1, markersize=4)
    axis[1, 2].set_title("second class F-Score(-1 means N/A) ")

    # For third class accuracy
    axis[2, 0].plot(accuracy3, label="testing Accuracy")
    axis[2, 0].plot(accuracyTrain3, color='red', label="Training Accuracy")
    axis[2, 0].set_title("third class accuracy")
    axis[2, 0].legend(loc='upper center')

    # For second class error
    axis[2, 1].plot(error3, color='red', linewidth=1, markersize=4, label="Test error")
    axis[2, 1].plot(errorTrain3, color='blue', linewidth=1, markersize=4, label="Train error")
    axis[2, 1].set_title("third class error")
    axis[2, 1].legend(loc='upper center')

    # For second class Fscore
    axis[2, 2].plot(Fscore3, marker='o', markerfacecolor='black', color='red', linewidth=1, markersize=4)
    axis[2, 2].set_title("third class F-Score(-1 means N/A) ")

    pt.show()


def main():
    list1, list2, list3 = load_Data("train.data")
    listTest1_2, listTest2_3, listTest3_1 = load_Data("test.data")
    list1_2, list2_3, list3_1 = shuffle(list1, list2, list3)

    listTest1_2 = replace(listTest1_2, "class-1", "class-2")
    listTest2_3 = replace(listTest2_3, "class-2", "class-3")
    listTest3_1 = replace(listTest3_1, "class-1", "class-3")

    list1_2te, Y1_2 = GetY_X(list1_2); listTest1_2te, YTest1_2 = GetY_X(listTest1_2)
    list2_3te, Y2_3 = GetY_X(list2_3); listTest2_3te, YTest2_3 = GetY_X(listTest2_3)
    list3_1te, Y3_1 = GetY_X(list3_1); listTest3_1te, YTest3_1 = GetY_X(listTest3_1)

    accuracy1, error1, accuracyTrain1, errorTrain1, Fscore1, accuracy_con1 = Perceptron(30, list1_2te, Y1_2, listTest1_2te, YTest2_3)
    accuracy2, error2, accuracyTrain2, errorTrain2, Fscore2, accuracy_con2 = Perceptron(120, list2_3te, Y2_3, listTest2_3te, YTest2_3)
    accuracy3, error3, accuracyTrain3, errorTrain3, Fscore3, accuracy_con3 = Perceptron(30, list3_1te, Y3_1, listTest3_1te, YTest3_1)

    print("the first class Accuracy Confusion : ", accuracy_con1[-1],"\nthe first class Fscore : "
          , Fscore1[-1],"\n\nthe second class Accuracy Confusion: ", accuracy_con2[-1], "\nthe second class Fscore : ", Fscore2[-1],
          "\n\nthe third class Accuracy Confusion: ", accuracy_con3[-1], "\nthe third class Fscore : ", Fscore3[-1])

    plot(accuracy1, error1, accuracyTrain1, errorTrain1, Fscore1,
         accuracy2, error2, accuracyTrain2, errorTrain2, Fscore2,
         accuracy3, error3, accuracyTrain3, errorTrain3, Fscore3)



if __name__ == '__main__':
    main()
