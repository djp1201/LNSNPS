
import pandas as pd


import time
import Model_LSNP
import Feeder
import Generator_LSNP
import Classifier_LSNP
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
import random
from sklearn.datasets import load_iris
acc =[]
for p in range(10):
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

    train_x = np.zeros((500, 9))
    train_y = np.zeros((500, 2))
    test_x = np.zeros((183, 9))
    test_y = np.zeros((183, 2))
    #length = np.arange(0,683)
    length = np.load("length_bcw.npy")
    #np.random.shuffle(length)
    #np.save('length_bcw.npy',length)

    j = k = 0
    for i in range(len(target)):
        if i < 500:
            train_x[j, :] = x[length[i], :]
            train_y[j, :] = y[length[i], :]
            j = j+1
        else:
            test_x[k, :] = x[length[i], :]
            test_y[k, :] = y[length[i], :]
            k = k+1                          #将鸢花数据集打乱
    train_x = np.array(train_x)
    train_y = np.array(train_y)         #训练集标签
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    feeder = Feeder.Feeder()
    generator = Generator_LSNP.Generator()
    maxgen = 300     #设置最大迭代次数
    fitness = []
    Loss = []
    m = 1
    bestfitness = 0
    worestfitness = 0
    m = feeder.prefire(m, 1)
    n = feeder.postfire(m, 1)
    print(train_x.shape)
    kernel_value = generator.kernel_value(train_x[:, :], 3)
    print(kernel_value.shape)
    kernel_value_train = generator.neuron(n, kernel_value)
    classifier = Classifier_LSNP.Classifier(kernel_value_train, train_y)
    model = Model_LSNP.Model()
    Lossclass = []

    model_w = classifier.w
    model_l = classifier.b

    for i in range(maxgen):
        predict = model.classification(model_w,model_l,kernel_value_train, train_y)

        model_w,model_l= classifier.train_classification(kernel_value_train, train_y)
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
    #np.save(f'bcw/fitness/loss_iris{p+1}.npy', fitness)
    #np.save(f'bcw/loss/loss_iris{p+1}.npy', Loss)



    plt.xlabel('Iterations')
    plt.ylabel('Accuracy rate')
    my_y_ticks = np.arange(0, 1.08, 0.1)
    plt.yticks(my_y_ticks)
    plt.grid()

    kernel_value_test = generator.kernel_value(test_x[:, :], 3)
    predict = model.classification(model_w,model_l , kernel_value_test, test_y)
    acc_rate = model.accuracy_rate(predict, test_y)
    print('\n\n\n识别准确率为{}%'.format(acc_rate * 100))
    acc.append(acc_rate*100)


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
