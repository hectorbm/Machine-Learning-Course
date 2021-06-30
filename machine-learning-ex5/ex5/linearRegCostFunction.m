function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
n = size(theta, 1);

% Hypothesis function term
h_x = X * theta;

% Loss function normal term + reg term
J_term1 = (1 / (2*m)) * sum((h_x - y).^2);
J_term2 = (lambda / (2*m)) * sum(theta(2:n) .^ 2);

% Sum the terms (Loss function)
J = J_term1 + J_term2;


temp_grad = (1/m) * sum(((h_x - y).* X), 1);
temp_grad = transpose(temp_grad);

% Compute the gradient for theta0 and the other regularized terms
grad = temp_grad + [0 ; (lambda / m ) * theta(2:n)];



% =========================================================================

grad = grad(:);

end
