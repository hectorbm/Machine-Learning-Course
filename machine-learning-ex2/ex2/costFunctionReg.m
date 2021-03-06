function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

n = size(theta)(1);

% Compute the hypothesis function sigmoid
h_x = 1 ./ (1 + exp(-1 * (X * theta)));

% Compute all the elements of the sum in the loss function
J_withoutReg = -(1/m) * (y .* log(h_x) + (1-y) .* log(1-h_x));

% Sum all the elements
J = ones(1,m) * J_withoutReg;

J = J + (lambda / (2 * m)) * ones(1,n-1) * (theta(2:n,:).^2);

% Compute the grad terms
grad_terms = (1/m) * (h_x - y) .* X;

% Add the grad terms to the grad vector
grad = transpose(grad_terms) * ones(m,1);
grad(2:n,:) = grad(2:n,:) + (lambda/m) * theta(2:n,:);



% =============================================================

end
