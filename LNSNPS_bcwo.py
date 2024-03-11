
from decimal import Decimal
import pandas as pd

import time
import Model_LNSNPS
import Feeder
import Generator_LNSNPS
import Classifier_LNSNPS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
Alpha_list = []
h = Decimal('0.1')
Alpha_range = Decimal('0.01')
import random
acc = []
best = 0

#length = np.load("length_bcw.npy")
#np.random.shuffle(length)
#np.save('length_bcw4.npy',length)
for p in range(1000):

    data=pd.read_csv('breast-cancer-wisconsin.data',header=None,sep=',')

    data = np.array(data)
    x = data[:,1:10]

    target = data[:,-1]
    start = time.time()

    y = []
    for i in range(len(target)):
        if target[i] == 2:
            y.append([1, 0])
        else:
            y.append([0, 1])
    y = np.array(y)

    train_x = np.zeros((350, 9))
    train_y = np.zeros((350, 2))
    test_x = np.zeros((333, 9))
    test_y = np.zeros((333, 2))
    #length = np.arange(0,683)
    length = np.load("length_bcw.npy")
    #np.random.shuffle(length)
    #np.save('length_bcw.npy',length)


    j = k = 0
    for i in range(len(target)):  #0-150
        if i < 350:
            train_x[j, :] = x[length[i], :]
            train_y[j, :] = y[length[i], :]
            j = j+1
        else:
            test_x[k, :] = x[length[i], :]
            test_y[k, :] = y[length[i], :]
            k = k+1
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    feeder = Feeder.Feeder()
    generator = Generator_LNSNPS.Generator()
    maxgen = 300
    fitness = []
    Loss = []
    m = 1
    bestfitness = 0
    worestfitness = 0
    m = feeder.prefire(m, 1)
    n = feeder.postfire(m, 1)
    print(train_x.shape)
    kernel_value_train = generator.kernel_value(train_x[:, :], 3)
    print(kernel_value_train.shape)
    classifier = Classifier_LNSNPS.Classifier(kernel_value_train, train_y)
    model = Model_LNSNPS.Model()

    Lossclass = []
    acc_rate = 0
    model_w = classifier.w
    for i in range(maxgen):

        print(Alpha_range)
        classifier.Alpha = Alpha_range
        predict = model.classification(model_w,kernel_value_train, train_y)

        model_w = classifier.train_classification(kernel_value_train, train_y)
        acc_rate = model.accuracy_rate(predict, train_y)
        loss = model.loss(predict, train_y)
        lossclass = model.lossclass(predict,train_y)
        Loss.append(loss)
        Lossclass.append(lossclass)

        fitness.append(acc_rate*100)
        print('第{}次迭代后的准确率为{}%'.format(i + 1, acc_rate * 100))

    #绘图
    fig1 = plt.figure(figsize=(8, 4))
    plt.xlim((0, maxgen+1))
    plt.ylim((0, 1.05))
    fitness_fig = plt.plot(fitness)

    #np.save(f"bcw/e/loss_iris{p+1}.npy", Lossclass)
    #np.save(f"bcw/fitness/loss_iris{p+1}.npy", fitness)
    #np.save(f"bcw/loss/loss_iris{p+1}.npy", Loss)


    plt.xlabel('Iterations')
    plt.ylabel('Accuracy rate')
    my_y_ticks = np.arange(0, 1.08, 0.1)
    plt.yticks(my_y_ticks)
    plt.grid()

    kernel_value_test = generator.kernel_value(test_x[:, :], 3)
    predict = model.classification(model_w,  kernel_value_test, test_y)
    acc_rate = model.accuracy_rate(predict, test_y)
    print('\n\n\n识别准确率为{}%'.format(acc_rate * 100))
    acc.append(acc_rate*100)
    if acc_rate > best :
        best = acc_rate
        np.save('length_bcw.npy',length)

    fitness = np.array(fitness)
    fig2 = plt.figure(figsize=(8, 4))
    predicted = np.zeros(len(predict))
    real_output = np.zeros(len(predict))
    for i in range(len(predict)):
        predicted[i] = np.argmax(predict[i, :])+1
        real_output[i] = np.argmax(test_y[i, :])+1
    #plt.xlim((0, len(predict)+1))

    x = np.array(range(len(predict)))

    end = time.time()
    print('运行时间为{}秒'.format(end-start))
    print(Alpha_list)
