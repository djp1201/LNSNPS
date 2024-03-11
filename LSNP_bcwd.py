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
from sklearn.datasets import load_breast_cancer
acc = []
for p in range(10):
    breast_data = load_breast_cancer()
    x = breast_data.data
    target = breast_data.target
    labels = breast_data.feature_names
    y = []
    for i in range(len(target)):
        if target[i] == 0:
            y.append([0, 1])
        else:
            y.append([1, 0])
    y = np.array(y)

    train_x = np.zeros((285, 30))
    train_y = np.zeros((285, 2))
    test_x = np.zeros((284, 30))
    test_y = np.zeros((284, 2))
    length = np.load("length_wbcd.npy")
    j = k = 0
    for i in range(len(target)):
        if i < 285:
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
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    feeder = Feeder.Feeder()
    generator = Generator_LSNP.Generator()
    maxgen = 200
    fitness = []
    Loss = []
    m = 1
    bestfitness = 0
    worestfitness = 0
    m = feeder.prefire(m, 1)
    n = feeder.postfire(m, 1)
    kernel_value = generator.kernel_value(train_x[:, :], kernel_option=3)
    kernel_value_train = generator.neuron(n, kernel_value)
    classifier = Classifier_LSNP.Classifier(kernel_value_train, train_y)
    model = Model_LSNP.Model()

    Lossclass = []
    model_w = classifier.w
    model_l = classifier.b
    for i in range(maxgen):
        predict = model.classification(model_w, model_l, kernel_value_train, train_y)
        model_w, model_l = classifier.train_classification(kernel_value_train, train_y)
        acc_rate = model.accuracy_rate(predict, train_y)
        loss = model.loss(predict, train_y)
        lossclass = model.lossclass(predict, train_y)

        Loss.append(loss)
        Lossclass.append(lossclass)
        fitness.append(acc_rate)
        print('第{}次迭代后的准确率为{}%'.format(i + 1, acc_rate * 100))

    #绘图
    fig1 = plt.figure(figsize=(8, 4))
    plt.xlim((0, maxgen+1))
    plt.ylim((0, 1.05))
    fitness_fig = plt.plot(fitness)

    #np.save(f"breast/e/loss_iris{p+1}.npy", Lossclass)
    #np.save(f'breast/fitness/loss_iris{p+1}.npy', fitness)
    #np.save(f'breast/loss/loss_iris{p + 1}.npy', Loss)

    plt.xlabel('Iterations')
    plt.ylabel('Accuracy rate')
    my_y_ticks = np.arange(0, 1.08, 0.1)
    plt.yticks(my_y_ticks)
    plt.grid()

    kernel_value_test = generator.kernel_value(test_x[:, :], kernel_option=3)
    predict = model.classification(model_w, model_l, kernel_value_test, test_y)
    acc_rate = model.accuracy_rate(predict, test_y)
    print('\n\n\n识别准确率为{}%'.format(acc_rate*100))
    acc.append(acc_rate*100)

    fig2 = plt.figure(figsize=(8, 4))

    predicted = np.zeros(len(predict))
    real_output = np.zeros(len(predict))
    for i in range(len(predict)):
        predicted[i] = np.argmax(predict[i, :])+1
        real_output[i] = np.argmax(test_y[i, :])+1
    plt.xlim((0, len(predict)+1))
    x = np.array(range(len(predict)))

    end = time.time()
