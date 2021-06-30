function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
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
    %       of the cost function (computeCost) and gradient here.
    %
    
    % 1) Compute the h(x) parameterized by theta.
    % 2) Compute the squared error (h_theta(x) - y)^2
    h_x = X * theta;
    %disp(h_x);
    error = (h_x - y);
    % Compute gradient 
    grad0 = (1/m) * ones(1,m) * error;
    grad1 = (1/m) * ones(1,m) * (error .* X(:,2));
    
    % Compute both theta values together, then, update
    temp0 = theta(1) - alpha * grad0;
    temp1 = theta(2) - alpha * grad1;
    %disp(temp0);
    theta(1) = temp0;
    theta(2) = temp1;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
