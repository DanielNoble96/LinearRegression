function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    sum1 = 0;
    sum2 = 0;
    for i = 1 : m
        diff_i = ((theta(1) * X(i, 1) + theta(2) * X(i, 2)) - y(i)) * X(i, 1);
        sum1 = sum1 + diff_i;
    end
    for i = 1 : m
        diff_i = ((theta(1) * X(i, 1) + theta(2) * X(i, 2)) - y(i)) * X(i, 2);
        sum2 = sum2 + diff_i;
    end
    theta(1) = theta(1) - alpha * 1 / m  * sum1;    
    theta(2) = theta(2) - alpha * 1 / m  * sum2;
    
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
end
disp(J_history);
end
