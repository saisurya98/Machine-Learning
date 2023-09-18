import numpy as np
import math
import matplotlib.pyplot as plt


def read_data(path):
    dataset = np.genfromtxt(path, delimiter=' ',encoding=None)
    dataset = dataset[:, [1, 3]]
    return dataset

# defining weight formula usinng x(i) , x and k
def weights(point, X, k=0.204):
    m, n = np.shape(X)
    I=np.eye((m))
    weights = np.mat(I)
    for j in range(m):
        diff = np.subtract(point,X[j])
        weights[j, j] = np.exp(np.matmul(diff, diff.T) / -(2 * k ** 2))
    return weights

# Calculation for finding unknown parameters using closed form formula
def local_weighted_params(point, X, y, k=0.204):
    y = y.reshape(-1, 1)
    wt = weights(point, X, k)
    X=np.matrix(X)
    XT = np.matrix.transpose(X)
    wt_X=np.matmul(wt,X)
    a=np.matmul(XT,wt_X)
    wt_Y=np.matmul(wt,y)
    b=np.matmul(XT,wt_Y)
    params=np.matmul(np.linalg.inv(a), b)
    return params

#Measure of y predicted function
def local_weighted_linear_regression(point, X, y, k=0.204):
    m = X.shape[0]
    X = np.append(np.ones(m).reshape(m, 1), X, axis=1)
    point = np.array([1, point])
    params = local_weighted_params(point, X, y, k)
    predicted = np.matmul(point, params)
    return predicted


def main():
    print('START Q2_AB\n')
    path_train = '../datasets/Q1_B_train.txt'
    # Reading the input train data
    train_data = read_data(path_train)
    train_data = train_data[train_data[:, 0].argsort()]
# seperating x and y from traindata
    x = train_data[:, 0]
    x = x.reshape(x.shape[0], 1)
    y = train_data[:, 1]
    y.reshape(-1, 1)
    k = 0.204
    parameters = []
    for point in x:
        parameters.append(local_weighted_params(point[0], x, y, k)[0, 0])
    print('the best fit parameters are', parameters)
    y_train_pred = []
    # prediction using local weight regression function using x train
    for point in x:
        y_train_pred.append(local_weighted_linear_regression(point[0], x, y, k)[0, 0])

    # plotting the graph
    plt.scatter(x, y, color='orange')
    plt.plot(x, y_train_pred, color='red')
    plt.title('Locally Weighted Linear Regression')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    print('END Q2_AB\n')


if __name__ == "__main__":
    main()
