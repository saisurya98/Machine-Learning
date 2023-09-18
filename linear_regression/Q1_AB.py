
import numpy as np
import math
import matplotlib.pyplot as plt

def main():
	print('START Q1_C\n')
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

# Similarly we can output best fitted parameters for any combination of any Frequency increment and function depth
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


def main():
	print('START Q1_AB\n')
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
# Converting string data type into float data type of Y

Y1 = []
for k in y:
    Y1.append(float(k))
Y = np.asarray(Y1)
Y = Y.reshape((-1, 1))

# Closed form approach for getting unknown parameters when k=1,d=1
a = []
for i in X:
    a.append(np.sin(i) * np.sin(i))
a = np.asarray(a)
X_col2 = a.reshape(128, 1)
X_col1 = np.ones((128, 1))
c = np.column_stack((X_col1, a))
X_c = np.matrix(c)
XT = np.matrix.transpose(X_c)
XT_X = np.matmul(XT, X_c)
XT_y = np.matmul(XT, Y)
tetak1d1 = np.matmul(np.linalg.inv(XT_X), XT_y)


# print(tetak1d1)


# parameters calculated
K = [1]
D = [1, 2, 3, 4, 5, 6,7]
for k in K:
    for d in D:
        print('the best fit parameters when k=' + str(k) + ' and d=' + str(d), pou(X, k, d))



K = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
D = [1,2,3,4,5,6,7]
for k in K:
    for d in D:
        plt.scatter(X, Y_pred(X, k, d), color='orange', marker='o')
        plt.title('Linear Regression with given basis function when k=' + str(k) + ' and d=' + str(d))
        plt.xlabel('Independent variable')
        plt.ylabel('Dependent variable')
        plt.show()
print('END Q1_AB\n')


if __name__ == "__main__":
    main()