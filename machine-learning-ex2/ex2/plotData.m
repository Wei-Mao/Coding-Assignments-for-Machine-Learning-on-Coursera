function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.
% Input:
%      X is a m by 2 matrix of scores on two exams
%      y is a m by 1 matrix of admission result(0 or 1)
% Create New Figure
figure; hold on;
% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
% y == 1 return a array of bools,
% find returns the linear indices of the non-zeros(logical 1 is viewed as non-zero)
pos_ind = find(y == 1);
neg_ind = find(y == 0);

plot(X(pos_ind, 1), X(pos_ind, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
hold on;
plot(X(neg_ind, 1), X(neg_ind, 2), 'ro', 'LineWidth', 2, 'MarkerSize', 7);










% =========================================================================



hold off;

end
