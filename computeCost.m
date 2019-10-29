function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples 
J = 0;
sum = 0;
for i = 1 : m
    diff_i = ((theta(1) * X(i, 1) + theta(2) * X(i, 2)) - y(i)) ^ 2;
    sum = sum + diff_i;
end
J = 1 / (2 * m) * sum;
end
