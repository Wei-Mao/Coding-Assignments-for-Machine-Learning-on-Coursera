# Programming Assignments for Machine Learning offered by Coursera                                         
This project is exactly the programming exercises counting towards the sucess of Machine Learning offered by       Coursera.
The algorithms are implemented with Octave (A GNU version of Matlab)
## Programming Exercise 1
This assignment covers the Linear Regression with one variable and multivarites. The optimization methods involve both gradient descent and normal equation. Importantly, implementation adopts the vectorized version!
## Programming Exercise 2
This assighment explores the Logistic Regression. As opposed to Linear Regression with ouputs taking any real value, logistic regression is meant to generate an ouput ranging between 0 and 1, exaclty the same as the range of probability!

Even though we map the 2-D feature vector into all polynomial terms up to sixth power, logistic regresssion is still able to be treated as the linear regerssion model applied to sigmoid function. Alternatively, you can image we have a long vector after feature transformation.

A small trick in the vectoriztion for the L2-Reguralization is to introduce a copy of the parameter vector but set the first element to zero because we do not penalize the intercept term.
