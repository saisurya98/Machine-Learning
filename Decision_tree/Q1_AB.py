import numpy as np

# definition for reading data set
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


# function for calculating accuracy
def calculate_accuracy(ytrue, ypred):
    acc = []
    x=len(ypred)
    for i in range(0,x):
        if ypred[i] == ytrue[i]:
            acc.append(1)
        else:
            acc.append(0)
    return np.mean(acc)


class Node():
    def __init__(current, features=None, threshold=None, left=None, right=None, infogain=None, value=None):
        current.features = features
        current.threshold = threshold
        current.left = left
        current.right = right
        current.infogain = infogain
        current.value = value

# intializing parameters
dep = 0
min_split = 2
class make_decision_tree():
    def __init__(current, min_split=min_split, dep=dep):
        current.min_split = min_split
        current.dep = dep
        current.root = None

    #  definition for tree building
    def build_tree(self, dataset, currentdepth=0):
        X= dataset[:, :-1]
        Y=dataset[:,-1]
        noof_samples, noof_features = np.shape(X)
        # split until stopping conditions are met
        if noof_samples >= self.min_split and currentdepth <= self.dep:
            # find the best split
            best_split = self.get_best_split(dataset, noof_samples, noof_features)
            # check if information gain is positive
            if best_split["info_gain"] > 0:
                left_subtree = self.build_tree(best_split["dataset_left"], currentdepth + 1)
                right_subtree = self.build_tree(best_split["dataset_right"], currentdepth + 1)
                return Node(best_split["feature_index"], best_split["threshold"], left_subtree, right_subtree,best_split["info_gain"])

        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        return Node(value=leaf_value)

    #  function to find the best split
    def get_best_split(self, dataset,num_samples, noof_features):
        best_split = {}
        max_info_gain = -float("inf")
        for feature_index in range(noof_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y= dataset[:, -1]
                    left_y=dataset_left[:,-1]
                    right_y=dataset_right[:,-1]
                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y)
                    # update the best split if needed
                    if curr_info_gain > max_info_gain:
                        best_split={
                            'feature_index':feature_index,
                            'threshold' : threshold,
                            'dataset_left':dataset_left,
                            'dataset_right':dataset_right,
                            'info_gain' : curr_info_gain
                        }
                        max_info_gain = curr_info_gain
        return best_split

    #  function to split the data
    def split(self, dataset, feature_index, threshold):
        dataset_left = np.array([i for i in dataset if i[feature_index] <= threshold])
        dataset_right = np.array([i for i in dataset if i[feature_index] > threshold])
        return dataset_left, dataset_right

    #  function to compute information gain
    def information_gain(self, parent, leftchild, rightchild):
        weightleft = len(leftchild) / len(parent)
        weightright = len(rightchild) / len(parent)
        gain = self.entropy(parent) - (weightleft * self.entropy(leftchild) + weightright * self.entropy(rightchild))
        return gain

    #  function to compute entropy
    def entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for i in class_labels:
            p = len(y[y == i]) / len(y)
            x=np.log2(p)
            entropy += -p * x
        return entropy

    #  function to compute leaf node
    def calculate_leaf_value(self, Y):
        Y = list(Y)
        return max(Y, key=Y.count)


    #  function to train the decisiontree
    def fit(self, entire_data):
        self.root = self.build_tree(entire_data)

    #  function to predict  dataset
    def predictor_function(self, X):
        return [self.make_prediction(x, self.root) for x in X]

    #  function to predict a data point's
    def make_prediction(self, x, tre):
        if tre.value != None:
            return tre.value
        feature_val = x[tre.features]
        if feature_val <= tre.threshold:
            return self.make_prediction(x, tre.left)
        else:
            return self.make_prediction(x, tre.right)


def main():
    print('START Q1_AB\n')
    '''
    Start writing your code here
    '''

    # Take the dataset and splitting them accordingly
    train_data = readFile('./datasets/Q1_train.txt')
    test_data = readFile('./datasets/Q1_test.txt')
    X_train = train_data[:, :3]
    X_test = test_data[:, :3]

    Y_train=train_data[:, -1]
    Y_train[Y_train == 'W'] = 0
    Y_train[Y_train == 'M'] = 1
    Y_train=Y_train.reshape(-1,1)

    n= np.concatenate((X_train, Y_train), axis=1)

    Y_test = test_data[:, -1]
    Y_test[Y_test == 'W'] = 0
    Y_test[Y_test == 'M'] = 1
    Y_test=Y_test.reshape(-1,1)

    dep = 1
    for i in range(0,5):
        # calling the classifier
        classifier = make_decision_tree(min_split=min_split, dep=dep)
        classifier.fit(n)
        print("Depth = ", dep)
        Y_trainpred = classifier.predictor_function(X_train)
        trainaccuracy = calculate_accuracy(Y_train, Y_trainpred)
        Y_testpred = classifier.predictor_function(X_test)
        testaccuracy = calculate_accuracy(Y_test, Y_testpred)
        print("Train = ", trainaccuracy," Test =  ", testaccuracy)
        dep = dep + 1

    print("Question is For which depths does the result indicate overfitting.Answer :- By analysing the results\n , "
          "we can see that overfitting has occured when depth is incrementing Overfitting occurs when we get high \n"
          "Training accuracy and low test accuracy .This is may be due to more training examples \n."
          "In the output , At depth = 5 , we are getting  Training accuracy which is 1 but the testing accuracy is 0.57.\n"
          "Hence , an exact overfitt has occured at Depth = 5. as depth is increasing a clear observation has been made\n"
          " train accuracy is been gradually increasing where as test accuracy is been slowly decreasing")
    print('END Q1_AB\n')


if __name__ == "__main__":
    main()