# Programming Assignments for Machine Learning offered by Coursera                                         
This project is exactly the programming exercises counting towards the success of Machine Learning offered by Coursera.
The algorithms are implemented with Octave (A GNU version of Matlab)
## Programming Exercise 1
This assignment covers the Linear Regression with one variable and multi-variates. The optimization methods involve both gradient descent and normal equation. Importantly, implementation adopts the vectorized version!
## Programming Exercise 2
This assignment explores the Logistic Regression. As opposed to Linear Regression with outputs taking any real value, logistic regression is meant to generate an output ranging between 0 and 1, exactly the same as the range of probability!

Even though we map the 2-D feature vector into all polynomial terms up to sixth power, logistic regression is still able to be treated as the linear regression model applied to sigmoid function. Alternatively, you can image we have a long vector after feature transformation.

A small trick in the vectorization for the L2-Regularization is to introduce a copy of the parameter vector but set the first element to zero because we do not penalize the intercept term.

## Programming Exercise 3
This assignment is composed of two tasks. In the first task, we tackle the multi-class classification using multiple logistic regression unit(one vesus rest). In the decision phase, final label for the given example is the one for which the corresponding logistic classifier has the largest output. The second task is to implement the forward propagation of neural network. As with the previous exercises, vectorized coding is also applied to his assignment. 
