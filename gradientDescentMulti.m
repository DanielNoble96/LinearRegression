function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
n = size(X, 2);
for iter = 1:num_iters
    % create array to store the sums so that thetas are changed
    % simultaneously
    sums = zeros(n, 1);
    % find each of the thetas 
    % create empty vector of thetas of correct size
    for j = 1 : n 
        sum = 0;
        % This loop is the summation and computes sum, to be used for this
        % theta_j
        for i = 1 : m
            h_i = 0;
            % compute the h thing
            for k = 1 : n
                h_i = h_i + theta(k, 1) * X(i, k);
            end
            % h_i for this theta is calculated; now find the sum
            sum = sum + ((h_i - y(i, 1)) * X(i, j));
        end
        % need to update theta outside of this loop, in the outermost one
        % store the current sums in an array then we can update theta when
        % all have been found
        sums(j, 1) = sum; 
    end
    % update thetas simultaneously
    for a = 1 : n
        theta(a, 1) = theta(a, 1) - alpha * (1 / m) * sums(a, 1);
    end
    
    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end
end
