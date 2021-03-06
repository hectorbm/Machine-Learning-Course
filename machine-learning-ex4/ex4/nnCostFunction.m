function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Number of classes "K"
num_labels = size(Theta2, 1);

% Feedforward step
a1 = [ones(m,1) X];

z2 = a1 * transpose(Theta1);
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];

z3 = a2 * transpose(Theta2); 
a3 = sigmoid(z3);

% Cost function step
JnonReg = 0;
for k = 1:num_labels
  temp = (y==k) .* log(a3(:,k)) + (1-(y == k)) .* log(1 - a3(:,k));
  JnonReg = JnonReg + sum(temp);
endfor
% Cost function without regularization
JnonReg = (-1/m) * JnonReg;

% Exclude the interceptor for each Theta_i(:, 1)
Theta1ToReg = Theta1(:, 2:size(Theta1,2));
Theta2ToReg = Theta2(:, 2:size(Theta2,2));

% Unroll the matrices
ThetaJoinedUnrolled = [Theta1ToReg(:) ; Theta2ToReg(:)];
regularizationTerm = (lambda / (2*m)) * sum(ThetaJoinedUnrolled.^2);

% Compute the regularized cost function
J = JnonReg + regularizationTerm;


% Backpropagation step
delta3 = zeros(size(a3));
% This for-loop is to consider a variable number of labels
for k = 1:num_labels
  delta3(:,k) = a3(:,k) - (y==k);
endfor

% After multiplying delta and theta, remove the first column, 
% in order to match the size of the gradient
delta2 = (delta3 * Theta2)(:,2:end) .* sigmoidGradient(z2);
delta1 = (delta2 * Theta1)(:,2:end) .* sigmoidGradient(X);

% Acumulator is equal to delta_i+1' * a_i 
D1 = transpose(delta2) * a1;
D2 = transpose(delta3) * a2;

% Non-regularized gradient
D1 = (1/m) * D1;
D2 = (1/m) * D2;

D1 = D1 + (lambda / m) * [zeros(size(Theta1,1),1) Theta1(:,2:end)];
D2 = D2 + (lambda / m) * [zeros(size(Theta2,1),1) Theta2(:,2:end)];

% Assign the gradients 
Theta1_grad = D1;
Theta2_grad = D2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
