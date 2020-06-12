function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);
% m is the number of the examples
% n equals the dimension of feature vector

% You should return these values correctly
%mu = zeros(n, 1);
%sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%

mu = mean(X, 1); 
% compute the mean for each column and return it as a row vector
% 1 by n
#mu_matrix = mu(:, ones(1 , n));
mu_matrix = repmat(mu, m, 1); % m by n
covar = 1 / m * (X - mu_matrix)' * (X - mu_matrix); % n by n
sigma2 = diag(covar);











% =============================================================


end
