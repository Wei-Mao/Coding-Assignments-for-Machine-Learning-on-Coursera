function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_try =  [0.01; 0.03; 0.1; 0.3; 1; 3;10; 30];
sigma_try = [0.01; 0.03; 0.1; 0.3; 1; 3;10; 30];
%C_try =  [0.01; 0.03;];
%sigma_try = [0.01; 0.03;];

%num_try = length(C_try);

% keep track of the best learning parameters
% lowest_cross_error = 1000000;
% or Store cross_error for all tries and then select the sigma and C by the index number!

lowest_cross_error = 100;  # initial guess of the lowest_cross_error
C_index = 1;
sigma_index = 1;

for i = 1:length(C_try)
    for j = 1:length(sigma_try)
      model= svmTrain(X, y, C_try(i), @(x1, x2) gaussianKernel(x1, x2, sigma_try(j)));
      predictions = svmPredict(model, Xval);
      cross_error = mean(double(predictions ~= yval));
 
      if cross_error <= lowest_cross_error
        % update the current optimum
        lowest_cross_error = cross_error;
        % update the index of the optimal parameters up to this point
        C_index = i;
        sigma_index = j;
      end
    end
end 

C = C_try(C_index);
sigma = sigma_try(sigma_index);
% =========================================================================

end
