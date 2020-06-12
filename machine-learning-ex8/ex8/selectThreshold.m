function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

   cv_predictions = pval < epsilon;  # The examlple is considered to be anomaly(positive) if it has a low probability less than epsilon.

   % Logical operators allow a simple and beautiful computation of confusion matrix!
   tp = sum((yval == 1) & (cv_predictions == 1));    % logical and can be interpreted as taking the intesection!
   fp = sum((yval == 0) & (cv_predictions == 1));
   fn = sum((yval == 1) & (cv_predictions == 0));

   % Adding the Octave constant eps to the denominator to avoid the "division by zero error"
   % calculate the Precision and Recall for the current epsilon
   precision = tp / (tp + fp + eps);
   recall = tp / (tp + fn + eps);

   % F1-score(amounts to the harmonic mean of precision and recall)
   F1 = 2 * precision * recall / (precision + recall + eps);

   % =============================================================

    if F1 > bestF1
       bestF1 = F1;  % update the current best F1-score
       bestEpsilon = epsilon; % take note of the current best thereshold
    end
end

end
