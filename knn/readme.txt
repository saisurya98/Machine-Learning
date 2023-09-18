

Use the dataset given at the bottom of this file.
K Nearest Neighbor
Q1 Consider the problem where we want to predict the gender of a person from a set of input parameters,
namely height, weight, and age.
a) Using Cartesian distance, Manhattan distance and Minkowski distance of order 3 as the similarity 
measurements show the results of the gender prediction for the Evaluation data that is listed below 
generated training data for values of K of 1, 3, and 7. Include the intermediate steps (i.e., distance
calculation, neighbor selection, and prediction). 
b) Implement the KNN algorithm for this problem. Your implementation should work with different training 
data sets as well as different values of K and allow to input a data point for the prediction.
c) To evaluate the performance of the KNN algorithm (using Euclidean distance metric), implement a leaveone-out evaluation routine for your algorithm. In leave-one-out validation, we repeatedly evaluate the 
algorithm by removing one data point from the training set, training the algorithm on the remaining data set 
and then testing it on the point we removed to see if the label matches or not. Repeating this for each of the 
data points gives us an estimate as to the percentage of erroneous predictions the algorithm makes and
thus a measure of the accuracy of the algorithm for the given data. Apply your leave-one-out validation with 
your KNN algorithm to the dataset for Question 1 c) for values for K of 1, 3, 5, 7, 9, and 11 and report the 
results. For which value of K do you get the best performance?
d) Repeat the prediction and validation you performed in Question 1 c) using KNN when the age data
is removed (i.e. when only the height and weight features are used as part of the distance calculation
in the KNN algorithm). Report the results and compare the performance without the age attribute
with the ones from Question 1 c). Discuss the results. What do the results tell you about the data