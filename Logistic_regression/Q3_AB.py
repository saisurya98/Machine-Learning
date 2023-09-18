import numpy as np
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
# Sigmoid function definition
def sigmoid(z):
    a= 1 + np.exp(-z)
    output = 1 /(a)
    return output

# optimization function definition
def optimize(x, y, learning_rate, iterations, parameters):
    size = x.shape[0]
    weight = parameters["weight"]
    bias = parameters["bias"]
    for i in range(iterations):
        s = sigmoid(np.matmul(x, weight) + bias)
        loss = -1 / size * np.sum(y * np.log(s) + (1 - y) * np.log(1 - s))
        dW = 1 / size * np.matmul(x.T, (s - y))
        db = 1 / size * np.sum(s - y)
        weight = weight-learning_rate * dW
        bias = bias-learning_rate * db
    parameters["weight"] = weight
    parameters["bias"] = bias
    return parameters

# Getting required paramaters from input train data
def train(x, y, learning_rate,iterations,parameter):
    parameters = optimize(x, y, learning_rate, iterations,parameter)
    return parameters

# Prediction function
def predict(x, parameters):
    z = np.matmul(x, parameters['weight']) + parameters['bias']
    pred = np.array([1 if i > 0.5 else 0 for i in sigmoid(z)])
    return pred

# Defining accuracy
def accuracy(y,ypred):
    correct = 0
    ypred = predict(X, parameters)
    for i in range(len(y)):
        if y[i] == ypred[i]:
            correct = correct+1
    return correct / len(y)
def main():
    print('START Q3_AB\n')
training_data = '../datasets/Q3_data.txt'
# Reading the given training data
train_np = readFile(training_data)
# Reading x input features
X = train_np[:, :-1]
X = X.astype(np.float64)

# Reading Y output variables
Y = train_np[:, -1]
# Setting Women label to zero and Male label to one for classification
Y[Y=='W']=0
Y[Y=='M']=1
Y = Y.astype(np.float64)

#initialize parameters
para = {}
para["weight"] = np.zeros(X.shape[1])
para["bias"] = 0
learning_rate=0.01

# Printing the output label for given X input feature, learning rate, no of iterations
iteration=[5,10,15,20,25,30,35,40,45]
for i in iteration:
        parameters=train(X,Y,0.01,i,para)
        y_pred = predict(X, parameters)
        print(y_pred)
# Printing accuracy for given no of iterations
for i in iteration:
        parameters=train(X,Y,0.01,i,para)
        y_pred = predict(X, parameters)
        print('Accuracy when no of iterations equal to',str(i),'would be ',str(accuracy(Y, y_pred)))

if __name__ == "__main__":
    main()
