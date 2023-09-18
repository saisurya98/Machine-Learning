import numpy as np

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

# Finding accuracy using locv method
def validation(X, y, para, learning_rate=0.01, n_iter=50):
    correct_pred = 0
    for i in range(X.shape[0]):
        row = X[i, :]
        original_label = y[i]
        X_LOCV = np.delete(X, i, axis=0)
        y_LOCV = np.delete(y, i)
        row = row.reshape(1, X.shape[1])
        parameters = train(X_LOCV, y_LOCV, 0.01, n_iter, para)
        predicted_label = predict(row, parameters)
        if original_label == predicted_label[0]:
            correct_pred = correct_pred+1
    return correct_pred / X.shape[0]

def main():
    print('START Q3_D\n')
training_data = '../datasets/Q3_data.txt'
# Reading the given training data
train_np = readFile(training_data)
# Reading x input features
X = train_np[:, :-1]
X = X.astype(np.float64)
X=X[:,[0,1]]
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
print('Leave one out validation score without age column is ', validation(X, Y, para))
if __name__ == "__main__":
    main()

# The accuracy with logistic regression when LOCV is 45% , for KNN and naive bayes algorithm(66% , 70.9%)
# the accuracies are more than this particular value
# Naive bayes outperformed KNN and logistic regression algorithm because navie bayes always measure
#probability at each data point for prediction however logistic regression will only classify points based on a decision boundary
