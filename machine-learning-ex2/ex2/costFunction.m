function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

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
%

% Compute the hypothesis function sigmoid
h_x = 1 ./ (1 + exp(-1 * (X * theta)));

% Compute all the elements of the sum in the loss function
J_terms = -(1/m) * (y .* log(h_x) + (1-y) .* log(1-h_x));
% Sum all the elements
J = ones(1,m) * J_terms;

% Compute the grad terms
grad_terms = (1/m) * (h_x - y) .* X;
% Add the grad terms to the grad vector
grad = grad + transpose(grad_terms) * ones(m,1);


% =============================================================

end
