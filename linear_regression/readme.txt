1. Consider a simplified fitting problem in the frequency domain where we are looking to find the best fit of data with a 
set of periodic (trigonometric) basis functions of the form 1, sin
2
(x), sin2
(k ∗x), sin2
(2 ∗ k ∗ x),..., where k is effectively the 
frequency increment. The resulting function for a given ”frequency increment”, k, and ”function depth”, d, and 
parameter vector Θ is then:
Try ”frequency increment” k from 1-10
For example, if k = 1 and d = 1, your basis (feature) functions are: 1, sin2
(x) 
 if k = 1 and d = 2, your basis (feature) functions are: 1, sin2
(x), sin2
(2.x) 
 if k = 3 and d = 4, your basis (feature) functions are: 1, sin2
(3*1*x), sin2
(3*2*x) , sin2
(3*3*x), sin2
(3*4*x) 
This means that this problem can be solved using linear regression as the function is linear in terms of the parameters Θ.
Try ”frequency increment” k from 1-10 and thus your basis functions as part of the data generation process described 
above.
a) Implement a linear regression learner to solve this best fit problem for 1 dimensional data. Make sure your 
implementation can handle fits for different ”function depths” (at least to ”depth” 6).
b) Apply your regression learner to the data set that was generated for Question 1b) and plot the resulting function 
for “function depth” 0, 1, 2, 3, 4, 5, and 6. Plot the resulting function together with the data points 
c) Evaluate your regression functions by computing the error on the test data points that were generated for 
Question 1c). Compare the error results and try to determine for what “function depths” overfitting might be a 
problem. Which ”function depth” would you consider the best prediction function and why? For which values of 
k and d do you get minimum error?
d) Repeat the experiment and evaluation of part b) and c) using only the first 20 elements of the training data set 
part b) and the Test set of part c). What differences do you see and why might they occur?
