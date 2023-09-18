
def main():
	print('START Q1_AB\n')


import numpy as np
import math
import matplotlib.pyplot as plt


def clean_data(line):
    return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')


def fetch_data(filename):
    with open(filename, 'r') as f:
        input_data = f.readlines()
        clean_input = list(map(clean_data, input_data))
        f.close()
    return clean_input


def readFile(dataset_path):
    input_data = fetch_data(dataset_path)
    input_np = np.array(input_data)
    return input_np


training_data = '../datasets/Q1_B_train.txt'
# Reading the given training data
train_np = readFile(training_data)

# Retriving X data from training data
x = []
for i in train_np:
    x.append(i[0])

# Retriving Y data from training data
y = []
for j in train_np:
    y.append(j[1])

# Converting string data type into float data type of X
X1 = []
for k in x:
    X1.append(float(k))
X = np.asarray(X1)
X = X[:20]
# Converting string data type into float data type of Y

Y1 = []
for k in y:
    Y1.append(float(k))
Y = np.asarray(Y1)
Y = Y[:20]
Y = Y.reshape((-1, 1))


# Function for getting best fit parameters for given k & d
def pou(x_train, k, d):
    mat = np.array([np.sin(i * k * x_train) * np.sin(i * k * x_train) for i in range(d + 1)]).T
    mat[mat == 0] = 1
    X = np.matrix(mat)
    XT = np.matrix.transpose(X)
    XT_X = np.matmul(XT, X)
    XT_y = np.matmul(XT, Y)
    teta = np.matmul(np.linalg.inv(XT_X), XT_y)
    return teta.tolist()


# parameters calculated
K = [1,2,3,4,5,6,7,8,9,10]
D = [1, 2, 3, 4, 5, 6,7]
for k in K:
    for d in D:
        print('the best fit parameters when k=' + str(k) + ' and d=' + str(d), pou(X, k, d))


# Function to get Y predicted for given X data points
def Y_pred(x_train, k, d):
    teta0 = pou(x_train, k, d)[0]
    temp = []
    for j in range(1, d + 1):
        temp.append(np.sin(k * j * x_train) * np.sin(k * j * x_train) * pou(x_train, k, d)[j])
    t = np.sum(temp, axis=0)
    ytrainpred = [z + teta0 for z in t]
    ytrainpred = np.asarray(ytrainpred)
    return ytrainpred

for k in K:
    for d in D:
        plt.scatter(X, Y_pred(X, k, d), color='orange', marker='o')
        # plt.plot(X, Y_pred(X, k, d), color='blue')
        plt.title('Linear Regression with given basis function when k=' + str(k) + ' and d=' + str(d))
        plt.xlabel('Independent variable')
        plt.ylabel('Dependent variable')
        plt.show()


def mean_squared_error(y_pred, y_real):
    N = y_pred.shape[0]
    error = np.sum((y_pred - y_real) ** 2) / (N)
    return error


for k in K:
    for d in D:
        print('Mean squared error would be for given k=' + str(k) + ' and d=' + str(d), 'is', mean_squared_error(Y_pred(X, k, d), Y))

    print('END Q1_AB\n')


if __name__ == "__main__":
    main()



#With 20 data points, The data is overfitting clearly since minimum error is for k=7 and d=6 0.002928147892915034 which is higher than the minimum error in the
#previous case.