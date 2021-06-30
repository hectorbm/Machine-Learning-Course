function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

% Separate positive and negative cases

X0 = X(y<0.5,:);
X1 = X(y>=0.5,:);

scatter(X1(:,1), X1(:,2), 'x');
scatter(X0(:,1), X0(:,2), 'o');

% =========================================================================



hold off;

end
