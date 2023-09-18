Another way to address nonlinear functions with a lower likelihood of overfitting is the use of locally weighted 
linear regression where the neighborhood function addresses non-linearity and the feature vector stays simple. In this 
case we assume that we will use only the raw feature, x, as well as the bias (i.e. a constant feature 1). Thus the locally 
applied regression function is y = Θ0 + Θ1 ∗ x
As discussed in class, locally weighted linear regression solves a linear regression problem for each query point, deriving 
a local approximation for the shape of the function at that point (as well as for its value). To achieve this, it uses a 
modified error function that applies a weight to each data point’s error that is related to its distance from the query 
point. Here we will assume that the weight function for the ith
 data point and query point x is: 
 Use γ : 0.204
where γ is a measure of the ”locality” of the weight function, indicating how fast the influence of a data
point changes with its distance from the query point.
a. Implement a locally weighted linear regression learner to solve the best fit problem for 1 dimensional data.
b. Apply your locally weighted linear regression learner to the data set that was generated for Question 1b) and 
plot the resulting function together with the data points
c. Evaluate the locally weighted linear regression on the Test data from Question 1 c). How does the performance 
compare to the one for the results from Question 1 c) ?
d. Repeat the experiment and evaluation of part b) and c) using only the first 20 elements of the training data set. 
How does the performance compare to the one for the results from Question 1 d) ? Why might this be the case?
e. Given the results form parts c) and d), do you believe the data set you used was actually derived from a function 
that is consistent with the function format in Question 1 ? Justify your answer.
