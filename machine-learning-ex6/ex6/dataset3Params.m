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
C_vector = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_vector = [0.01 0.03 0.1 0.3 1 3 10 30];

min_loss = inf;

for row = 1:length(C_vector)
  for column = 1:length(sigma_vector)
    % Define the C and sigma values
    C = C_vector(row);
    sigma = sigma_vector(column);
    
    % Train and plot the boundary for C and sigma  
    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    
    predictions = svmPredict(model, Xval);
    val_error = mean(double(predictions ~= yval));
    
     % Update when proper value appears
    if val_error < min_loss
      C_opt = C_vector(row);
      sigma_opt = sigma_vector(column);
      min_loss = val_error;
    endif
    
  endfor
endfor

C = C_opt;
sigma = sigma_opt;

fprintf("Min found at C: %f  Sigma: %f  Validation Loss: %f \n",C ,sigma,min_loss);



% =========================================================================

end
