function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    
    % Initialize the error matrix
    error = zeros(size(X));
    
    % Evaluate the hypothesis for each training element
    h_x = X * theta;
   
    % Add the same difference vector h(x) - y to all the columns of the error matrix
    error = error .+ (h_x - y);
   
    % Multiply (element-wise) the error matrix for each element of a feature
    error = error .* X;
   
    % Sum the values to obtain the gradient
    grad = ones(1,m) * error;
    grad = transpose(grad);
    
    % Compute all the theta values    
    theta = theta - (alpha/m) * grad;
     
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
