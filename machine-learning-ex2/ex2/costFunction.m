function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.
%   Input:
%       X is a m by (n + 1) matrix of features. (X = [ones(m, 1), X] is already implements in the ex2.m)
%       y is a m by 1 vector of labels taking on {0, 1}
%       theta is a (n+1) dimensional vector
% Initialize some useful values
m = length(y); % number of training examples
theta = theta(:); % make sure theta is of the proper shape!
y = y(:);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta

% I implement with vectorized pattern.
h = sigmoid(X * theta);
J = 1/m * (-y' * log(h) - (1 - y)' * log(1 - h));
grad = 1/m * X' * (h - y);








% =============================================================

end
