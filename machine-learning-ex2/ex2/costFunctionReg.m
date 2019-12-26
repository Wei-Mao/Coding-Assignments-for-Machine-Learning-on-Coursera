function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 
%   Input:
%       X is a m by (n + 1) matrix of features. (X = [ones(m, 1), X] is already implements in the ex2.m)
%       y is a m by 1 vector of labels taking on {0, 1}
%       theta is a (n+1) dimensional vector
% Mapping  (x1, x2) into all polynomial terms of x1 an x2 up to the sixth power still leads to the lieaner regresssion but in high dimensional space! 
% Initialize some useful values
m = length(y); % number of training examples
theta = theta(:);
y = y(:);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
theta_no_o = zeros(size(theta));
theta_no_o(2:end) = theta(2:end);

h = sigmoid(X * theta);
J = 1 / m * (-y' * log(h) - (1 - y)' *log((1 - h))) + lambda / (2 * m) * theta_no_o' * theta_no_o;
grad = 1 /m * X' * (h - y) + (lambda / m) * theta_no_o;





% =============================================================

end
