__author__ = 'JaclynNguyen'
from numpy import random
import numpy as np
import matplotlib.pyplot as plt

def train_data():
    random.seed(1)
    x = np.random.normal(size =(1000,2))
    a = np.array([[3,3],[3,-3]])
    z = np.random.normal(1)
    # True Y
    y = 1/(1 + np.exp(-np.dot(x,a[:,0]))) + np.dot(x,a[:,1])**2 + 0.3*z

    # Generating the random weights


    return x, y

x_train, y_train = train_data()
x, y = x_train[0:900,:], y_train[0:900]
x_t, y_t = x_train[900:1000,], y_train[900:1000]
learning = .0001

train = np.arange(900)
random.shuffle(train)
nodes = 10


lamb_err = []
for lam in [1,5,10,30]:
    epochoverall = []
    testoverall = []
    for epoch in [30,100,200,500,1000, 2000]:
        random.seed(1)
        w1 = np.random.normal(size = (2, nodes))
        b1 = np.ones(10).reshape(1,nodes)
        w2 = np.random.normal(size = (nodes, 1))
        b2 = np.array(1)
        for j in range(epoch):
            random.shuffle(train)
            epoch_error = 0
            for i in train:
                # print x[i]
                a = np.dot(x[i], w1) + b1
                # print a
                z = 1.0/(1 + np.exp(-a))
                # print z
                y_hat = np.dot(z, w2) + b2
                # print y_hat

                delta = y[i] - int(y_hat)
                if j == epoch-1:
                    epoch_error += delta**2


                sigmoid = np.exp(-(np.dot(x[i],  w1) + b1))/(1 + np.exp(-(np.dot(x[i], w1) + b1)))**2
                # print sigmoid

                w1_deriv = (-delta * w2.T * sigmoid).T * x[i] + ((lam/float(len(train))) * w1).T
                w2_deriv = -delta * (1.0/(1 + np.exp(-(np.dot(x[i],  w1) + b1)))) + ((lam/float(len(train))) * w2).T

                b1_deriv = -delta * w2.T * sigmoid
                b2_deriv = -delta

                w1 = w1 - learning*w1_deriv.T
                w2 = w2 - learning*w2_deriv.T


                b1 = b1 - learning*b1_deriv
                b2 = b2 - learning*b2_deriv

        epoch_rmse = np.sqrt(epoch_error/float(len(train)))
        epochoverall.append(epoch_rmse)
        print "Epoch: " + str(epoch) + ", RMSE: " + str(epoch_rmse)

        sum_error = []
        for i in range(len(x_t)):
            a = np.dot(x_t[i], w1) + b1
            # print a
            z = 1.0/(1 + np.exp(-a))
            # print z
            y_hat = np.dot(z, w2) + b2
            error = (y_t[i] - y_hat)**2
            sum_error.append(error)

        testoverall.append(np.sqrt(np.sum(sum_error)/len(y_t)))
        print 'Test RMSE: ' + str(np.sqrt(np.sum(sum_error)/len(y_t)))

    lamb_err.append([epochoverall,testoverall])



# fig = plt.figure()
# plt.title('Neural Networks without Regularization', fontsize = 20, fontweight = 'bold')
# tra, = plt.plot([30,100,200,500,1000, 2000], epochoverall, c = 'red', label = 'Train Error', alpha = .4)
# test, = plt.plot([30,100,200,500,1000, 2000], testoverall, c = 'green', label = 'Test Error', alpha = .4)
# plt.legend(handles = [tra, test])
# plt.savefig('NNwoReg.png', dpi = 1000)


# fig2 = plt.figure()
# # plt.axis([1,30,1,30])
# plt.plot([30,100,200,500,1000, 2000], lamb_err[0][0])
# plt.plot([30,100,200,500,1000, 2000], lamb_err[1][0])
# plt.plot([30,100,200,500,1000, 2000], lamb_err[2][0])
# plt.plot([30,100,200,500,1000, 2000], lamb_err[3][0])
# plt.title('Neural Networks with Regularization \n Training Error', fontsize = 20, fontweight = 'bold')
# plt.show()
# # plt.legend(handles = [tra1, test5, tra10, test30])
# plt.savefig('NNRegTrain.png', dpi = 1000)



fig = plt.figure()
plt.title('Neural Networks with Regularization', fontsize = 20, fontweight = 'bold')
# plt.axis([1,30,1,30])
tra1, = plt.plot([30,100,200,500,1000, 2000], lamb_err[0][0], 'red', ls = 'dashed', label = '1')
test5, = plt.plot([30,100,200,500,1000, 2000], lamb_err[1][0], 'red', ls = 'solid', label = '5')
tra10, = plt.plot([30,100,200,500,1000, 2000], lamb_err[2][0], 'red', ls = 'dashdot', label = '10')
test30, = plt.plot([30,100,200,500,1000, 2000], lamb_err[3][0], 'red', ls = 'dotted',label = '30')
tra1_, = plt.plot([30,100,200,500,1000, 2000], lamb_err[0][1], 'blue', ls = 'dashed', label = '1')
test5_, = plt.plot([30,100,200,500,1000, 2000], lamb_err[1][1], 'blue', ls = 'solid', label = '5')
tra10_, = plt.plot([30,100,200,500,1000, 2000], lamb_err[2][1], 'blue', ls = 'dashdot', label = '10')
test30_, = plt.plot([30,100,200,500,1000, 2000], lamb_err[3][1],'blue', ls= 'dotted',label = '30')
plt.legend(handles = [tra1, test5, tra10, test30, tra1_, test5_, tra10_, test30_])
plt.savefig('NNReg.png', dpi = 1000)


# fig = plt.figure()
# plt.title('Neural Networks without Regularization \n Test Error', fontsize = 20, fontweight = 'bold')
# tra1_, = plt.plot([30,100,200,500,1000, 2000], lamb_err[0][1], 'blue', ls = 'dashed', label = '1')
# test5_, = plt.plot([30,100,200,500,1000, 2000], lamb_err[1][1], 'blue', ls = 'solid', label = '5')
# tra10_, = plt.plot([30,100,200,500,1000, 2000], lamb_err[2][1], 'blue', ls = 'dashdot', label = '10')
# test30_, = plt.plot([30,100,200,500,1000, 2000], lamb_err[3][1],'blue', ls= 'dotted',label = '30')
# plt.legend(handles = [tra1_, test5_, tra10_, test30_])
# plt.savefig('NNRegTest.png', dpi = 1000)
