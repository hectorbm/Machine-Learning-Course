function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

# Compute the hypothesis function
h_x = sigmoid(X * theta);

# Compute a non-regularized Cost Function
J_terms = (-1/m) * ((y .* log(h_x)) + ((1-y) .* log(1-h_x)));
# Sum all the terms 
J_nonReg = sum(J_terms);
# Add the Regularization terms
n = length(theta);
J_regTerms  = (lambda / (2 * m)) * sumsq(theta(2:n));
# Sum the raw J and the regularization terms.
J = J_nonReg + J_regTerms;

# Compute the grad terms (sum)
grad_terms = (1/m) * ((h_x - y) .* X);
# Non regularized gradient
grad_nonReg = transpose(sum(grad_terms, 1));

# Regularization term (for gradient)
grad_regTerm = ((lambda / m) * [0;theta(2:n)]);

# Grad = traditional_reg + regterms
grad = grad_nonReg + grad_regTerm;

% =============================================================

grad = grad(:);

end
