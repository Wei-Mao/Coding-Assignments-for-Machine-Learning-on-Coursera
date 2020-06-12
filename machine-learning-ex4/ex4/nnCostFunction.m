function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
% vector to matrix via arithmetic progression!
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);   # the number of the examples
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m


% Recodeing the labels as one-hot code vector for the purpose of computing the
% cost function

Y = zeros(num_labels, m);
for i = 1:m
Y(y(i), i) = 1;
end
% In the above Y, one column corresponds to  one example,agreeing with the A[L],namely ouput matrix with
% one column for one example!
%Y = Y'; %such that one row is a example!
%disp('Y')
%Y(1000:1004, :)
% Add one row of 1's to account for the bias term 
X = [ones(m, 1), X];  
X = X';
# one column one examples
# the first row of 1's handles the bias term

%disp('Shape of X')
%size(X)
A_1 = sigmoid(Theta1 * X);
%disp("Shape of Theta1")
%size(Theta1)
%disp("Shape of Theta2")
%size(Theta2)
% Add one column of 1's to the top of A_1
A_1 = [ones(1, size(A_1, 2)); A_1];
%disp('Shape of A_1')
%size(A_1)
A_2 = sigmoid(Theta2 * A_1);
%disp('Shape of A_2')
%size(A_2)
K = size(A_2, 1);  # the nunmber of the output units
J = 0;
for k = 1:K
  J = J + (- Y(k, :) * log(A_2(k,:))' - (1 - Y(k, :)) * log(1 - A_2(k, :))');
end

Theta1_penalty = Theta1(:, 2:end);
Theta2_penalty = Theta2(:, 2:end);

J = 1/m * J + lambda / (2 * m) * (Theta1_penalty(:)' * Theta1_penalty(:) + ...
    Theta2_penalty(:)' * Theta2_penalty(:));


%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

sigmoid_grad_1 = A_1(2:end, :) .* (1 - A_1(2:end, :));   # remove  the term correposnding to the bias
dJ_dZ_2 = A_2 - Y;                  # one row for one output unit, one column for one examples
W1 = Theta1(:, 2:end);
b1 = Theta1(:, 1);
W2 = Theta2(:, 2:end);
b2 = Theta2(:, 1);
dJ_W1 = zeros(size(W1));
dJ_b1 = zeros(size(b1));
%disp('Shape of dJ_W1')
%size(dJ_W1)
dJ_W2 = zeros(size(W2));
dJ_b2 = zeros(size(b2));

# only compute the gradient one time 
for k = 1:K
  dJ_b2(k) = dJ_b2(k) + sum(dJ_dZ_2(k, :), 2);
  dJ_W2(k, :) = dJ_W2(k, :) + dJ_dZ_2(k, :) * A_1(2:end, :)';
  
  dJ_dZ_1 = W2(k, :)' * dJ_dZ_2(k, :).* sigmoid_grad_1;
  dJ_b1 = dJ_b1 + sum(dJ_dZ_1, 2);
  dJ_W1 = dJ_W1 + dJ_dZ_1 * X(2:end, :)';
end

# multuply the derivatives by 1/ m 
#  add the derivatives of penalty term to account for the regularization
dJ_W1 = dJ_W1 + lambda * W1;
dJ_W2 = dJ_W2 + lambda * W2;
Theta1_grad = 1/m * [dJ_b1, dJ_W1];
Theta2_grad = 1/ m * [dJ_b2, dJ_W2];

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
