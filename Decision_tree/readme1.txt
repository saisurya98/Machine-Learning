Consider the problem from the previous assignments where we want to predict gender from information about height, 
weight, and age. We will use Decision Trees to make this prediction. Note that as the data attributes are continuous numbers 
you have to use the ≥ attribute and determine a threshold for each node in the tree. As a result, you need to solve the 
information gain for each threshold that is halfway between two data points and thus the complexity of the computations 
increases with the number of data items. 
a) Implement a decision tree learner for this particular problem that can derive decision trees with an arbitrary, predetermined depth (up to the maximum depth where all data sets at the leaves are pure) using the information gain 
criterion.
b) Divide the data set from Question 1c) in Project 1 (the large training data set) into a training set comprising the first 
50 data points and a test set consisting of the last 70 data elements. Use the resulting training set to derive trees of 
depths 1 - 5 and evaluate the accuracy of the resulting trees for the 50 training samples and for the test set containing 
the last 70 data items. Compare the classification accuracy on the test set with the one on the training set for each 
tree depth. For which depths does the result indicate overfitting