
def main():
	print('START Q1_C\n')
	import numpy as np
	import math
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

	# Function for getting best fit parameters for given k & d in train data
	def pou(x_train, k, d):
		mat = np.array([np.sin(i * k * x_train) * np.sin(i * k * x_train) for i in range(d + 1)]).T
		mat[mat == 0] = 1
		X = np.matrix(mat)
		XT = np.matrix.transpose(X)
		XT_X = np.matmul(XT, X)
		XT_y = np.matmul(XT, Y)
		teta = np.matmul(np.linalg.inv(XT_X), XT_y)
		return teta.tolist()

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


	test_data = '../datasets/Q1_C_test.txt'
	# Reading the given test data
	test_np = readFile(test_data)
	# Retriving X data from test data
	x = []
	for i in test_np:
		x.append(i[0])

	# Retriving Y data from test data
	y = []
	for j in test_np:
		y.append(j[1])

	# Converting string data type into float data type of X
	X_test = []
	for k in x:
		X_test.append(float(k))
	X_test = np.asarray(X_test)

	# Converting string data type into float data type of Y

	Y_test = []
	for k in y:
		Y_test.append(float(k))
	Y_test = np.asarray(Y_test)
	Y_test = Y_test.reshape((-1, 1))

	def po(X_test, k, d):
		mat = np.array([np.sin(i * k * X_test) * np.sin(i * k * X_test) for i in range(d + 1)]).T
		mat[mat == 0] = 1
		X = np.matrix(mat)
		XT = np.matrix.transpose(X)
		XT_X = np.matmul(XT, X)
		XT_y = np.matmul(XT, Y_test)
		teta = np.matmul(np.linalg.inv(XT_X), XT_y)
		return teta.tolist()

	# Function to get Y predicted for given X data points
	def Y_pre(X_test, k, d):
		teta0 = po(X_test, k, d)[0]
		temp = []
		for j in range(1, d + 1):
			temp.append(np.sin(k * j * X_test) * np.sin(k * j * X_test) * po(X_test, k, d)[j])
		t = np.sum(temp, axis=0)
		ytrainpred = [z + teta0 for z in t]
		ytrainpred = np.asarray(ytrainpred)
		return ytrainpred

	def mean_squared_error(y_pred, y_real):
		N = y_pred.shape[0]
		error = np.sum((y_pred - y_real) ** 2) / (N)
		return error



	K = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	D = [1, 2, 3, 4, 5, 6, 7]
	for k in K:
		for d in D:
			print('Mean squared error for test set when k=' + str(k) + ' and d=' + str(d),'is',mean_squared_error(Y_pre(X_test,k,d), Y_test))
			print('Mean squared error for train set when k=' + str(k) + ' and d=' + str(d), 'is',mean_squared_error(Y_pred(X, k, d), Y))

	print('END Q1_C\n')

#1Compare the error results and try to determine for what “function depths” overfitting might be a problem
#for k = 10 and any d the data is showing very less error on train set and with test set its also showing error very less. This can be one of
#the cases of overfitting since test set showing error more than tarin set sometimes and adding many features here may lead to overfitting.

#2Which function depth would you consider the best prediction function and why?
#When depth d=7 we could consider it as best prediction function because MSE is minimum in this case
#when compared to other values of d.

#3For which values of k and d do you get minimum error?
#For k =7 , d=7 we got minimum test error which is 4.57e-22
if __name__ == "__main__":
    main()
    