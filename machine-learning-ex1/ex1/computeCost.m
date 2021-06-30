function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% Compute the predicted values
h_x = X * theta;

% Compute the differences between prediction and ground truth
error_squared = (h_x - y) .^ 2;

% Compute J as the averaged of all squared errors, A.K.A MSE
J = (1 / (2 * m)) * ones(1, m) * error_squared;

% =========================================================================

end
