import numpy as np


def main():
    print('START Q2_C\n')

    def read_data(path):
        dataset = np.genfromtxt(path, delimiter=' ')
        dataset = dataset[:, [1, 3]]
        return dataset

# Reading train data
    path_train = '../datasets/Q1_B_train.txt'
    # Reading the input train data
    train_data = read_data(path_train)
    train_data = train_data[train_data[:, 0].argsort()]
    # seperating x and y from traindata
    x = train_data[:, 0]
    x = x.reshape(x.shape[0], 1)
    y = train_data[:, 1]
    y.reshape(-1, 1)
# Reading test data
    path_test = '../datasets/Q1_C_test.txt'
    data_test = read_data(path_test)
    # separating x and y from test data
    x_test = data_test[:, 0]
    x_test = x_test.reshape(x_test.shape[0], 1)
    y_test = data_test[:, 1]
    y_test.reshape(x_test.shape[0], 1)

    g = 0.204

    # defining weight formula usinng x(i) , x and k
    def weights(point, X, k=0.204):
        m, n = np.shape(X)
        I = np.eye((m))
        weights = np.mat(I)
        for j in range(m):
            diff = np.subtract(point,X[j])
            weights[j, j] = np.exp(np.matmul(diff, diff.T) / -(2 * k ** 2))
        return weights

    # Calculation for finding unknown parameters using closed form formula
    def local_weighted_params(point, X, y, k=0.204):
        y = y.reshape(-1, 1)
        wt = weights(point, X, k)
        X = np.matrix(X)
        XT = np.matrix.transpose(X)
        wt_X = np.matmul(wt, X)
        a = np.matmul(XT, wt_X)
        wt_Y = np.matmul(wt, y)
        b = np.matmul(XT, wt_Y)
        params = np.matmul(np.linalg.inv(a), b)
        return params

    # Measure of y predicted function
    def local_weighted_linear_regression(point, X, y, k=0.204):
        m = X.shape[0]
        X = np.append(np.ones(m).reshape(m, 1), X, axis=1)
        point = np.array([1, point])
        params = local_weighted_params(point, X, y, k)
        predicted = np.matmul(point, params)
        return predicted
# prediction using local weight regression using train  data
    y_pred_train = []
    for point in x:
            y_pred_train.append(local_weighted_linear_regression(point[0], x, y, g)[0, 0])
    y_pred_train = np.array(y_pred_train)

# prediction using local weight regression using test data
    k = 0.204
    y_pred_test = []
    for point in x_test:
        y_pred_test.append(local_weighted_linear_regression(point[0], x_test, y_test, k)[0, 0])

    y_pred_test = np.array(y_pred_test)
# Mean square error
    print('Mean square error for given test data is', np.mean(np.abs(y_test - y_pred_test)))
    print('Mean square error for given 128 training  data is', np.mean(np.abs(y - y_pred_train)))
    print('END Q2_C\n')


if __name__ == "__main__":
    main()
#Error on train set is 0.64597 which is higher than that of train error from the previous question
#Error on test set is 0.215